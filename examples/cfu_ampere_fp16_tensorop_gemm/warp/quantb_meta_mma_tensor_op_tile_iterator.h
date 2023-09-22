#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {


////////////////////////////////////////////////////////////////////////////////

// Traits to describe the layout of quantization meta data layout in a MMA fragment
// Since operand B is quantized on a per block basis, it's one meta data per block.

template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads>
class QuantBMetaMmaTile{
public:

  using WarpShapeB = WarpShapeB_;
  using BlockingShape = BlockingShape_;
  using ArchMmaOperator = ArchMmaOperator_;

  static_assert(Threads == 32, "This iterator should work in a warp only.");

  /// Shape of the curresponding operand B tile iterator <instruction_k, warp_n>
  using TileShapeB = MatrixShape<ArchMmaOperator::Shape::kK, WarpShapeB::kColumn>;

  // Tensor core operand B layout is a column major 4x8 tile, divided
  // into 32 threads (T0 ~ T31) as shown below. Each element of the tile is 32b,
  // so for fp16 it becomes 8 x 8, and int8 it becomes 16 x 8. 
  //  T0 |  T4 |  T8 | T12 | T16 | T20 | T24 | T28
  //  T1 |  T5 |  T9 | T13 | T17 | T21 | T25 | T29
  //  T2 |  T6 | T10 | T14 | T18 | T22 | T26 | T30
  //  T3 |  T7 | T11 | T15 | T19 | T23 | T27 | T31
  using CoreTile = layout::PitchLinearShape<4, 8>;

  /// Each thread holds a 32b fragement per tile: for half precision, it's 2 elements, 4 elements for int8
  static int const kNumBsPerCoreTileFragement = 32 / sizeof_bits<typename ArchMmaOperator::ElementB>::value;

  /// Each mma instruction can process either 1 or 2 tensor core operand B tiles (stacked on the k dimension)
  static int const kBTilesPerMma =
      sizeof_bits<typename ArchMmaOperator::ElementB>::value * ArchMmaOperator::FragmentB::kElements / 32;
  static_assert(kBTilesPerMma == 1 || kBTilesPerMma == 2, "Only support 1 or 2 operand B tiles per mma.");

  /// Each operand B tile iterator load covers a number of mma instructions
  static int const kMmaIterationsB = WarpShapeB::kColumn / ArchMmaOperator::Shape::kN;

  /// Number of B elements a fragment of meta data should cover
  static int const kExpandedSize = kNumBsPerCoreTileFragement * kBTilesPerMma * kMmaIterationsB;

  // Now we figure out how many meta data elements to load for each TileShapeB
  
  /// Number of meta elements per CoreTile.
  static int const kCoreTileFragementSize = (kNumBsPerCoreTileFragement + BlockingShape::kRow - 1) / BlockingShape::kRow;

  /// Number of core tiles per mma instruction, different from kBTilesPerMma when blocking size on K dimension
  /// exceeds the tile depth, so two tiles share the same meta data
  static int const kTilesPerMma = ((kBTilesPerMma == 2) && 
                                  (BlockingShape::kRow <= kNumBsPerCoreTileFragement * CoreTile::kContiguous)) 
                                  ? 2 : 1;

  /// stride to reach the meta data for the next CoreTile on the K dimension
  static int const kKTileStride = (kNumBsPerCoreTileFragement * CoreTile::kContiguous + BlockingShape::kRow - 1) / BlockingShape::kRow;

  /// Stride on N dimention should be the tile width, shrunk by blocking size on this dimension.
  static int const kNStride = (CoreTile::kStrided + BlockingShape::kColumn - 1) / BlockingShape::kColumn;

  /// On N dimension, how many tiles share the same meta data
  static int const kNRepeats = (BlockingShape::kColumn + CoreTile::kStrided - 1) / CoreTile::kStrided;

  /// Each fragement should cover kMmaIterationsB number of mma intructions on the N dimension.
  /// When blocking size on this dimension exceeds the tile width, multiple iterations
  /// would share the same data.
  static int const kMmaIterations = (kMmaIterationsB + kNRepeats - 1) / kNRepeats;

  static int const kFragementSize = kCoreTileFragementSize * kTilesPerMma * kMmaIterations;

  CUTLASS_DEVICE
  static MatrixCoord lane_position(int lane_id) {
    return make_Coord((lane_id % CoreTile::kContiguous) * kNumBsPerCoreTileFragement,
         lane_id / CoreTile::kContiguous);
  }
};


////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is to load quantization meta data for operand B from
/// shared memory to fragments (hopefully allocated to registers by compilers).
/// Examples of meta data include scale or offsets. The operand B matrix is
/// quantized on a per block basis, meaning one element of meta data per block.
///
/// This is meant to be used in lock step with the operand B tile iterator.
/// So all parameters are logical positions in the operand B tiles.
/// The goal here is to deliver each meta data element to its corresponding
/// operand B element for dequantization. As a result, we need to figure
/// out the operand B layout in the tensor core.
///
template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Data type of the meta data elements
  typename Element_,
  /// Layout of meta data tensor
  typename Layout_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1>
class QuantBMetaMmaTensorOpTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for column major layout

template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Data type of the meta data elements
  typename Element_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads>
class QuantBMetaMmaTensorOpTileIterator<WarpShapeB_, BlockingShape_,
    Element_, cutlass::layout::ColumnMajor, ArchMmaOperator_,
    Threads, 1>{
public:

  using WarpShapeB = WarpShapeB_;
  using BlockingShape = BlockingShape_;
  using Element = Element_;
  using Layout = cutlass::layout::ColumnMajor;
  using ArchMmaOperator = ArchMmaOperator_;

  using MetaTile = QuantBMetaMmaTile<WarpShapeB, BlockingShape, ArchMmaOperator, Threads>;

  /// Number of MMA instructions for this tile
  static constexpr int kMmaIterationsB = MetaTile::kMmaIterationsB;

  /// Number of B elements per mma tile fragment (32b), 2 for half precision, 4 for int8
  static constexpr int kNumBsPerCoreTileFragement = MetaTile::kNumBsPerCoreTileFragement;

  /// Each mma instruction can process either 1 or 2 operand B tiles (stacked on the k dimension)
  static constexpr int kBTilesPerMma = MetaTile::kBTilesPerMma;

  /// Number of B elements a fragment of meta data should cover
  static constexpr int kExpandedSize = MetaTile::kExpandedSize;

  /// Number of meta elements per core tile fragment
  static constexpr int kCoreTileFragementSize = MetaTile::kCoreTileFragementSize;

  /// stride for reaching the next core tile (if there is one) on the K dimension
  static constexpr int kKTileStride = MetaTile::kKTileStride;

  /// do we need to load meta data for the next core tile on the K dimension?
  static constexpr int kTilesPerMma = MetaTile::kTilesPerMma;

  static constexpr int kNStride = MetaTile::kNStride;
  static constexpr int kNRepeats = MetaTile::kNRepeats;
  static constexpr int kMmaIterations = MetaTile::kMmaIterations;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;
  using Fragment = Array<Element, MetaTile::kFragementSize>;

  using AccessType = Array<Element, kCoreTileFragementSize>;

private:

  Element *pointer_;

  Layout layout_;

  TensorCoord lane_position_;

public:

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator() { }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator(
    TensorRef const &ref, 
    int thread_idx,
    int warp_idx,
    int lane_idx
  ): 
    pointer_(ref.data()),
    layout_(ref.layout()),
    lane_position_(MetaTile::lane_position(lane_idx))
     {}

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    int row = lane_position_.row() / BlockingShape::kRow;
    int column = lane_position_.column() / BlockingShape::kColumn;

    AccessType* dst_ptr = reinterpret_cast<AccessType*>(frag.data());
    CUTLASS_PRAGMA_UNROLL
    for (int n_idx = 0, c = column; n_idx < kMmaIterations; n_idx++, c += kNStride){
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_idx = 0, r = row; mma_tile_idx < kTilesPerMma; mma_tile_idx++, r += kKTileStride){
        AccessType* src_ptr = reinterpret_cast<AccessType*>(pointer_ + layout_({r, c}));
        *dst_ptr = *src_ptr;
        dst_ptr++;
      }
    }
  }

  CUTLASS_HOST_DEVICE
  static Array<Element, kExpandedSize> debug_expand(Fragment const &frag){
    Array<Element, kExpandedSize> ret;
    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      int n_idx = n_out / kNRepeats;
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);
        CUTLASS_PRAGMA_UNROLL
        for (int elem_out_idx = 0; elem_out_idx < kNumBsPerCoreTileFragement; elem_out_idx++){
          int elem_idx = elem_out_idx / BlockingShape::kRow;
          int idx = elem_idx + mma_tile_idx * kCoreTileFragementSize + n_idx * kCoreTileFragementSize * kTilesPerMma;
          ret[out_idx] = frag[idx];
          out_idx++;
        }
      }
    }
    return ret;
  }

  CUTLASS_HOST_DEVICE
  static void dequant(Fragment const &scales, Array<uint8_t,kExpandedSize/2> const &weights, Array<Element, kExpandedSize>& dest){
    static_assert(kNumBsPerCoreTileFragement == 2, "Only for 16b gemm now.");

    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      int n_idx = n_out / kNRepeats;
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);

        // We take advantage of the fact that kNumBsPerCoreTileFragement == 2 to extract
        // 2 int4 weights out of a bytes. Valid for all 16b GEMM.
        int idx = mma_tile_idx * kCoreTileFragementSize + n_idx * kCoreTileFragementSize * kTilesPerMma;

        int w1 = weights[out_idx/2] & 0x0f;
        int w2 = weights[out_idx/2] >> 4;
        Element s = scales[idx];
        dest[out_idx] = Element(s * (w1 - 8));
        if constexpr (BlockingShape::kRow == 1){
          s = scales[idx+1];
        }
        out_idx++;
        dest[out_idx] = Element(s * (w2 - 8));
        out_idx++;
      }
    }
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator++() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(MetaTile::TileShapeB::kRow, 0);
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator--() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(MetaTile::TileShapeB::kRow, 0);
    return *this;
  }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int rows = tile_offset.row() * MetaTile::TileShapeB::kRow;
    int columns = tile_offset.column() * MetaTile::TileShapeB::kColumn;
    lane_position_ += TensorCoord(rows, columns);
    return *this;
  }

};


////////////////////////////////////////////////////////////////////////////////

/// Specialization for row major layout

template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Data type of the meta data elements
  typename Element_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads>
class QuantBMetaMmaTensorOpTileIterator<WarpShapeB_, BlockingShape_,
    Element_, cutlass::layout::RowMajor, ArchMmaOperator_,
    Threads, 1>{
public:

  using WarpShapeB = WarpShapeB_;
  using BlockingShape = BlockingShape_;
  using Element = Element_;
  using Layout = cutlass::layout::RowMajor;
  using ArchMmaOperator = ArchMmaOperator_;

  static_assert(BlockingShape::kColumn == 1 && BlockingShape::kRow > 1,
          "Only support column blocking for row major layout");

  using MetaTile = QuantBMetaMmaTile<WarpShapeB, BlockingShape, ArchMmaOperator, Threads>;

  /// Number of MMA instructions for this tile
  static constexpr int kMmaIterationsB = MetaTile::kMmaIterationsB;

  /// Number of B elements per mma tile fragment (32b), 2 for half precision, 4 for int8
  static constexpr int kNumBsPerCoreTileFragement = MetaTile::kNumBsPerCoreTileFragement;

  /// Each mma instruction can process either 1 or 2 operand B tiles (stacked on the k dimension)
  static constexpr int kBTilesPerMma = MetaTile::kBTilesPerMma;

  /// Number of B elements a fragment of meta data should cover
  static constexpr int kExpandedSize = MetaTile::kExpandedSize;

  /// Number of meta elements per core tile fragment
  static constexpr int kCoreTileFragementSize = MetaTile::kCoreTileFragementSize;

  /// stride for reaching the next core tile (if there is one) on the K dimension
  static constexpr int kKTileStride = MetaTile::kKTileStride;

  /// do we need to load meta data for the next core tile on the K dimension?
  static constexpr int kTilesPerMma = MetaTile::kTilesPerMma;

  static constexpr int kNStride = MetaTile::kNStride;
  static constexpr int kNRepeats = MetaTile::kNRepeats;
  static constexpr int kMmaIterations = MetaTile::kMmaIterations;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;
  using Fragment = Array<Element, MetaTile::kFragementSize>;

//  using AccessType = Array<Element, kMmaIterations>;

private:

  Element *pointer_;

  Layout layout_;

  TensorCoord lane_position_;

public:

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator() { }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator(
    TensorRef const &ref, 
    int thread_idx,
    int warp_idx,
    int lane_idx
  ): 
    pointer_(ref.data()),
    layout_(ref.layout()),
    lane_position_(MetaTile::lane_position(lane_idx))
     {}

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    int row = lane_position_.row() / BlockingShape::kRow;
    int column = lane_position_.column() / BlockingShape::kColumn;
    static_assert(kTilesPerMma * kCoreTileFragementSize == 1, "Only support one meta data per core tile");

    Element* src_ptr = reinterpret_cast<Element*>(pointer_ + layout_({row, column}));
    Element* dst_ptr = reinterpret_cast<Element*>(frag.data());
    CUTLASS_PRAGMA_UNROLL
    for (int n_idx = 0; n_idx < kMmaIterations; n_idx++){
      dst_ptr[n_idx] = src_ptr[n_idx * kNStride];
    }
  }

  CUTLASS_HOST_DEVICE
  static Array<Element, kExpandedSize> debug_expand(Fragment const &frag){
    Array<Element, kExpandedSize> ret;

    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      int n_idx = n_out / kNRepeats;
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);
        CUTLASS_PRAGMA_UNROLL
        for (int elem_out_idx = 0; elem_out_idx < kNumBsPerCoreTileFragement; elem_out_idx++){
          int elem_idx = elem_out_idx / BlockingShape::kRow;
          int col = elem_idx + mma_tile_idx * kCoreTileFragementSize;
          int idx = col * kMmaIterations + n_idx;
          ret[out_idx] = frag[idx];
          out_idx++;
        }
      }
    }
    return ret;
  }

  CUTLASS_HOST_DEVICE
  static void dequant(Fragment const &scales, Array<uint8_t,kExpandedSize/2> const &weights, Array<Element, kExpandedSize>& dest){
    static_assert(kNRepeats == 1, "This is implied by BlockingShape::kColumn == 1");
    static_assert(kNumBsPerCoreTileFragement == 2, "Only for 16b gemm now.");

    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      Element s = scales[n_out];
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        // We take advantage of the fact that kNumBsPerCoreTileFragement == 2 to extract
        // 2 int4 weights out of a bytes. Valid for all 16b GEMM.
        int w1 = weights[out_idx/2] & 0x0f;
        int w2 = weights[out_idx/2] >> 4;
        dest[out_idx] = Element(s * (w1 - 8));
        dest[out_idx + 1] = Element(s * (w2 - 8));
        out_idx += 2;
      }
    }
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator++() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(MetaTile::TileShapeB::kRow, 0);
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator--() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(MetaTile::TileShapeB::kRow, 0);
    return *this;
  }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int rows = tile_offset.row() * MetaTile::TileShapeB::kRow;
    int columns = tile_offset.column() * MetaTile::TileShapeB::kColumn;
    lane_position_ += TensorCoord(rows, columns);
    return *this;
  }

};


////////////////////////////////////////////////////////////////////////////////
}  // namespace warp
}  // namespace gemm
}  // namespace cutlass