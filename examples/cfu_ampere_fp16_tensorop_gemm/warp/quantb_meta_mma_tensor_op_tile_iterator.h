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

  using Fragment = Array<Element, kNumBsPerMmaTileFragement * kBTilesPerMma * kMmaIterationsB>;

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
    int b_row = lane_position_.row();
    int b_column = lane_position_.column();
    int meta_column = b_column / BlockingShape::kColumn;

    // if (warp_idx_ == 1 && thread_idx_ != 0 && lane_idx_ == 0){
    //     printf("b_row: %d, b_column: %d, meta_column: %d\n", b_row, b_column, meta_column);
    // }
    
    int load_idx = 0;
    int mma_tile_k_offset = kNumBsPerMmaTileFragement * 4;
    for (int i = 0; i < kMmaIterationsB; i++){
        for (int mma_tile_idx = 0; mma_tile_idx < kBTilesPerMma; mma_tile_idx++){
            int row = b_row + mma_tile_idx * mma_tile_k_offset;
            for (int elem_idx = 0; elem_idx < kNumBsPerMmaTileFragement; elem_idx++){
                frag[load_idx] = pointer_[layout_({(row + elem_idx)/BlockingShape::kRow, meta_column})];
                load_idx++;
            }
        }
        b_column += 8;
        meta_column = b_column / BlockingShape::kColumn;
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