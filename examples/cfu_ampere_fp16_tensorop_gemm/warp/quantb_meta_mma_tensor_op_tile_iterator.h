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
class QuantBMetaMmaTensorOpTileIterator{
public:

  using WarpShapeB = WarpShapeB_;
  using BlockingShape = BlockingShape_;
  using Element = Element_;
  using Layout = Layout_;
  using ArchMmaOperator = ArchMmaOperator_;

  static_assert(Threads == 32, "This iterator should work in a warp only.");
  static_assert(PartitionsK_ == 1, "This iterator does not support K partitioning yet.");

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  /// Shape of the curresponding operand B matrix tile iterator
  using TileShapeB = MatrixShape<ArchMmaOperator::Shape::kK, WarpShapeB::kColumn>;

  /// Number of columns (N) this warp needs to iterate over MMA instruction over
  /// So it is the number of mma fragements (element B) we need to load per warp.
  static int const kMmaIterationsB = WarpShapeB::kColumn / ArchMmaOperator::Shape::kN;

  // Base of the tensor core operand layout is a column major 4x8 tile, divided
  // into 32 threads (T0 ~ T31) as shown below. Each element of the tile is 32b,
  // so for fp16 it becomes 8 x 8, and int8 it becomes 16 x 8. 
  //  T0 |  T4 |  T8 | T12 | T16 | T20 | T24 | T28
  //  T1 |  T5 |  T9 | T13 | T17 | T21 | T25 | T29
  //  T2 |  T6 | T10 | T14 | T18 | T22 | T26 | T30
  //  T3 |  T7 | T11 | T15 | T19 | T23 | T27 | T31
  using MmaTileShape = layout::PitchLinearShape<4, 8>;

  /// Number of B elements per mma tile fragment (32b per fragment)
  static int const kNumBsPerMmaTileFragement = 32 / sizeof_bits<typename ArchMmaOperator::ElementB>::value;

  /// Each mma instruction can process either 1 or 2 operand B tiles (stacked on the k dimension)
  static int const kBTilesPerMma =
      sizeof_bits<typename ArchMmaOperator::ElementB>::value * ArchMmaOperator::FragmentB::kElements / 32;
  static_assert(kBTilesPerMma == 1 || kBTilesPerMma == 2, "Only support 1 or 2 operand B tiles per mma.");

  // Since meta data is one per block, we need to figure out how many loads we needs.
  // For each 32b fragment, number of meta data elements to load is number of B elements
  // in 32b divided by block size on the k dimension.
  static int const kMmaTileFragementSize = (kNumBsPerMmaTileFragement + BlockingShape::kRow - 1) / BlockingShape::kRow;

  static int const kKTileStride = (kNumBsPerMmaTileFragement * MmaTileShape::kContiguous + BlockingShape::kRow - 1) / BlockingShape::kRow;
  static int const kTilesPerMma = ((kBTilesPerMma == 2) && 
                                  (BlockingShape::kRow <= kNumBsPerMmaTileFragement * MmaTileShape::kContiguous)) 
                                  ? 2 : 1;

  /// Each fragement should cover kMmaIterationsB number of mma tiles on the N dimension.
  /// Stride on N dimention should be the tile width, shrunk by blocking size on this dimension.
  static int const kNStride = (MmaTileShape::kStrided + BlockingShape::kColumn - 1) / BlockingShape::kColumn;

  /// Number of B elements sharing a meta data element on N dimension
  static int const kNRepeats = (BlockingShape::kColumn + MmaTileShape::kStrided - 1) / MmaTileShape::kStrided;

  /// Each fragement should cover kMmaIterationsB number of mma tiles on the N dimension.
  /// When blocking size on this dimension exceeds the tile width, multiple iterations
  /// would share the same data.
  static int const kMmaIterations = (kMmaIterationsB + kNRepeats - 1) / kNRepeats;

  static int const kExpandedSize = kNumBsPerMmaTileFragement * kBTilesPerMma * kMmaIterationsB;

  using Fragment = Array<Element, kMmaTileFragementSize * kTilesPerMma * kMmaIterations>;

  using AccessType = Array<Element, kMmaTileFragementSize>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;


private:

  CUTLASS_DEVICE
  static TensorCoord lane_position(int lane_id) {
    return make_Coord((lane_id % MmaTileShape::kContiguous) * kNumBsPerMmaTileFragement,
         lane_id / MmaTileShape::kContiguous);
  }

  Element *pointer_;

  Layout layout_;

  TensorCoord lane_position_;

  /// Stride quantity
  StrideIndex stride_;

  // Debug
  int thread_idx_;
  int warp_idx_;
  int lane_idx_;

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
    lane_position_(lane_position(lane_idx)),
    stride_(ref.stride()[0]),
    thread_idx_(thread_idx),
    warp_idx_(warp_idx),
    lane_idx_(lane_idx)
     {}

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) {
    int row = lane_position_.row() / BlockingShape::kRow;
    int column = lane_position_.column() / BlockingShape::kColumn;

    int load_idx = 0;
    for (int n_idx = 0; n_idx < kMmaIterations; n_idx++){
      for (int mma_tile_idx = 0; mma_tile_idx < kTilesPerMma; mma_tile_idx++){
        AccessType* src_ptr = reinterpret_cast<AccessType*>(pointer_ + layout_({row + mma_tile_idx * kKTileStride, column + n_idx * kNStride}));
        AccessType* dst_ptr = reinterpret_cast<AccessType*>(frag.data() + load_idx);
        load_idx += kMmaTileFragementSize;
        *dst_ptr = *src_ptr;
      }
    }
  }

  CUTLASS_HOST_DEVICE
  static Array<Element, kExpandedSize> debug_expand(Fragment const &frag){
    Array<Element, kExpandedSize> ret;
    int out_idx = 0;
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      int n_idx = n_out / kNRepeats;
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);
        for (int elem_out_idx = 0; elem_out_idx < kNumBsPerMmaTileFragement; elem_out_idx++){
          int elem_idx = elem_out_idx / BlockingShape::kRow;
          int idx = elem_idx + mma_tile_idx * kMmaTileFragementSize + n_idx * kMmaTileFragementSize * kTilesPerMma;
          ret[out_idx] = frag[idx];
          out_idx++;
        }
      }
    }
    return ret;
  }

  CUTLASS_HOST_DEVICE
  static void dequant(Fragment const &scales, Array<uint8_t,kExpandedSize/2> const &weights, Array<Element, kExpandedSize>& dest){
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kExpandedSize/2; ++i) {
      dest[i * 2] = static_cast<Element>(int(weights[i] & 0x0f) - 8);
      dest[i * 2 + 1] = static_cast<Element>(int(weights[i] >> 4) - 8);
    }

    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      int n_idx = n_out / kNRepeats;
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);
        CUTLASS_PRAGMA_UNROLL
        for (int elem_out_idx = 0; elem_out_idx < kNumBsPerMmaTileFragement; elem_out_idx++){
          int elem_idx = elem_out_idx / BlockingShape::kRow;
          int idx = elem_idx + mma_tile_idx * kMmaTileFragementSize + n_idx * kMmaTileFragementSize * kTilesPerMma;
          dest[out_idx] = dest[out_idx] * scales[idx];
          out_idx++;
        }
      }
    }
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator++() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(TileShapeB::kRow, 0);
    return *this;
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator--() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(TileShapeB::kRow, 0);
    return *this;
  }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int rows = tile_offset.row() * TileShapeB::kRow;
    int columns = tile_offset.column() * TileShapeB::kColumn;
    lane_position_ += TensorCoord(rows, columns);
    return *this;
  }

};

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass