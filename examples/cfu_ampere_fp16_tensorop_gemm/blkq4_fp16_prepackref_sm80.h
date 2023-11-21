/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *
 * Licensed under the MIT License.
 * 
 * Module Name:
 *    blkq4_fp16_prepackref_sm80.h
 *
 * Abstract:
 *   Utils for prepacking quantized weights in block-wise q4 x fp16 gemm,
 *   specialized for sm80.
 */

#include "matrix_layout.h"

namespace onnxruntime {
namespace cuda {

/**
 * @brief Prepack weight matrix to facilitate matrix loading, depending on MMA
 * instruction layout.
 * 
 * The weight matrix is int4, yet we want to leverage existing fp16/bf16
 * tile loading and MMA layout code in CUTLASS. So we group 4 int4 into 2
 * bytes, pretending it's fp16. This grouping must be done in a way to be
 * easily unpacked into tiles that match the MMA instruction layout.
 * For MMA instruction <16, 8, 16>, each instruction processes 2 8x8 tiles,
 * vertically stacked on the K dimension. And MmaTensorOpMultiplicandTileIterator
 * loads a <InstructionShape::kK, WarpShape::kN> tile.
 * 
 * So we stack 2x2 tiles on a 3rd dimeansion, and reshape them in a HWC fashion:
 * T0, T2
 * T1, T3
 * ==>
 * T0[0, 0], T1[0, 0], T2[0, 0], T3[0, 0]
 * T0[1, 0], T1[1, 0], T2[1, 0], T3[1, 0]
 * T0[2, 0], T1[2, 0], T2[2, 0], T3[2, 0]
 * T0[3, 0], T1[3, 0], T2[3, 0], T3[3, 0]
 * ...
 * T0[0, 7], T1[0, 7], T2[0, 7], T3[0, 7]
 * T0[1, 7], T1[1, 7], T2[1, 7], T3[1, 7]
 * T0[2, 7], T1[2, 7], T2[2, 7], T3[2, 7]
 * T0[3, 7], T1[3, 7], T2[3, 7], T3[3, 7]
 *
 * This pack a 8x16 int8 tile into a 16x8 int8 tile, i.e. a 8x8 16b tile
*/
void prepack_weights_ref(
    int rows,
    int columns,
    const MatrixRef<uint8_t const, ColumnMajorLayout, true>& tensor_weight,
    const MatrixRef<uint8_t, ColumnMajorLayout, true>& tensor_weight_prepacked) {
  Expects(tensor_weight.shape()[0] == rows / 2 && tensor_weight.shape()[1] == columns);
  Expects(tensor_weight_prepacked.shape()[0] == rows && tensor_weight_prepacked.shape()[1] == columns / 2);

  auto t0_base = make_Position(0, 0);
  auto t1_base = make_Position(4, 0);
  auto t2_base = make_Position(0, 8);
  auto t3_base = make_Position(4, 8);
  for (int col_dtile = 0; col_dtile < columns / 16; ++col_dtile) {
    for (int row_dtile = 0; row_dtile < rows / 16; ++row_dtile) {
      // Packing from a 8x16 tile to a 16x8 tile
      auto dtile_base = make_Position(row_dtile * 8, col_dtile * 16);
      auto packed_tile_base = make_Position(row_dtile * 16, col_dtile * 8);
      for (int col = 0; col < 8; ++col) {
        for (int row = 0; row < 4; ++row) {
          auto cord = make_Position(row, col);
          auto packed_cord = packed_tile_base + make_Position(row * 4, col); // packed tile is 16x8
          uint8_t buf[4];
          buf[0] = tensor_weight.at(dtile_base + t0_base + cord);
          buf[1] = tensor_weight.at(dtile_base + t1_base + cord);
          buf[2] = tensor_weight.at(dtile_base + t2_base + cord);
          buf[3] = tensor_weight.at(dtile_base + t3_base + cord);
          
          // [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7] so that each pair of adjacent weights
          // are in different b16 register at the same positions. This makes it easier to convert to
          // fp16x2 format in a b32 register

          tensor_weight_prepacked.at(packed_cord) = (buf[0] & 0x0f) | ((buf[1] & 0x0f) << 4);
          tensor_weight_prepacked.at(packed_cord + make_Position(1, 0)) = (buf[2] & 0x0f) | ((buf[3] & 0x0f) << 4);
          tensor_weight_prepacked.at(packed_cord + make_Position(2, 0)) = ((buf[0] & 0xf0) >> 4) | (buf[1] & 0xf0);
          tensor_weight_prepacked.at(packed_cord + make_Position(3, 0)) = ((buf[2] & 0xf0) >> 4) | (buf[3] & 0xf0);
        }
      }
    }
  }
}

template <
    typename ScaleElementT,
    typename Layout,
    typename QuantBlocking>
void prepack_quant_scales_ref(
    int rows,
    int columns,
    const MatrixRef<ScaleElementT const, Layout, true>& tensor_scale,
    const MatrixRef<ScaleElementT, Layout, true> tensor_scale_prepacked) {
  Expects(tensor_scale.shape()[0] == (rows / QuantBlocking::kRow) && tensor_scale.shape()[1] == (columns / QuantBlocking::kColumn));
  Expects(tensor_scale_prepacked.shape() == tensor_scale.shape());

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr (sizeof(ScaleElementT) == 2 && QuantBlocking::kRow == 1){
      // In Ampere tensor op, each operand B tile is 8 x 8, in a warp of 32 threads, each thread
      // holds a fragement of the tile containing 2 elements in the k dimension. Most often we use
      // mma instruction shape of 16x8x16, which means 2 B tiles are stacked in the k dimension,
      // as shown below (T stands for thread):
      // T0, T4, T8, T12
      // T1, T5, T9, T13
      // T2, T6, T10, T14
      // T3, T7, T11, T15
      // T0, T4, T8, T12
      // T1, T5, T9, T13
      // T2, T6, T10, T14
      // T3, T7, T11, T15
      //
      // We need to deliver quantization scale and offset elements to the corresponding threads,
      // so we can perform dequantization efficiently. With a column major layout, each thread
      // needs two seperate loads for a mma instruction, due to the tile fragement layout shown
      // above. To reduce the number of loads, we rearrange each column as below, so we can use
      // a single load to load fragements for two tiles:
      // T0        T0
      // T1        T0
      // T2        T1
      // T3   =>   T1
      // T0        T2
      // T1        T2
      // T2        T3
      // T3        T3
      
      for (int col = 0; col < tensor_scale.shape()[1]; ++col){
        for (int row_blk = 0; row_blk < tensor_scale.shape()[0]; row_blk += 16){
          for (int thread_id = 0; thread_id < 4; thread_id++){
            const int dst_idx = row_blk + thread_id * 4;
            const int src_idx = row_blk + thread_id * 2;
            tensor_scale_prepacked.at(dst_idx + 0, col) = tensor_scale.at(src_idx + 0, col);
            tensor_scale_prepacked.at(dst_idx + 1, col) = tensor_scale.at(src_idx + 1, col);
            tensor_scale_prepacked.at(dst_idx + 2, col) = tensor_scale.at(src_idx + 8, col);
            tensor_scale_prepacked.at(dst_idx + 3, col) = tensor_scale.at(src_idx + 9, col);
          }
        }
      }
  } else {
    // In all other cases, we don't prepack scale or offset
    std::copy(tensor_scale.data().begin(), tensor_scale.data().end(), tensor_scale_prepacked.data().begin());
  }
}


template <typename Layout, typename QuantBlocking>
void prepack_quant_offsets_ref(
    size_t rows,
    size_t columns,
    MatrixRef<uint8_t const, Layout, true> tensor_offset,
    MatrixRef<uint8_t, Layout, true> tensor_offset_prepacked) {
  Expects(tensor_offset.shape()[0] == (rows / QuantBlocking::kRow) && tensor_offset.shape()[1] == (columns / QuantBlocking::kColumn));
  Expects(tensor_offset_prepacked.shape() == tensor_offset.shape());

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr(QuantBlocking::kRow == 1){
      // In Ampere tensor op, each operand B tile is 8 x 8, in a warp of 32 threads, each thread
      // holds a fragement of the tile containing 2 elements in the k dimension. Most often we use
      // mma instruction shape of 16x8x16, which means 2 B tiles are stacked in the k dimension,
      // as shown below (T stands for thread):
      // T0, T4, T8, T12
      // T1, T5, T9, T13
      // T2, T6, T10, T14
      // T3, T7, T11, T15
      // T0, T4, T8, T12
      // T1, T5, T9, T13
      // T2, T6, T10, T14
      // T3, T7, T11, T15
      //
      // We need to deliver quantization scale and offset elements to the corresponding threads,
      // so we can perform dequantization efficiently. With a column major layout, each thread
      // needs two seperate loads for a mma instruction, due to the tile fragement layout shown
      // above. To reduce the number of loads, we rearrange each column as below, so we can use
      // a single load to load fragements for two tiles:
      // T0        T0
      // T1        T0
      // T2        T1
      // T3   =>   T1
      // T0        T2
      // T1        T2
      // T2        T3
      // T3        T3
    if (tensor_offset_prepacked.good()){
      for (int col = 0; col < tensor_offset.shape()[1]; ++col){
        for (int row_blk = 0; row_blk < tensor_offset.shape()[0]; row_blk += 16){
          for (int thread_id = 0; thread_id < 4; thread_id++){
            const int dst_idx = row_blk + thread_id * 4;
            const int src_idx = row_blk + thread_id * 2;
            // [a, b, c, d] => [a, c, b, d] so that adjacent weights are in their own
            // 16b element: [a, x, b, x] and [x, c, x, d], which makes it easier to
            // convert to fp16x2 format in a b32 register
            tensor_offset_prepacked.at(dst_idx + 0, col) = tensor_offset.at(src_idx + 0, col);
            tensor_offset_prepacked.at(dst_idx + 1, col) = tensor_offset.at(src_idx + 8, col);
            tensor_offset_prepacked.at(dst_idx + 2, col) = tensor_offset.at(src_idx + 1, col);
            tensor_offset_prepacked.at(dst_idx + 3, col) = tensor_offset.at(src_idx + 9, col);
          }
        }
      }
    }
  } else {
    // In all other cases, we don't prepack scale or offset
    std::copy(tensor_offset.data().begin(), tensor_offset.data().end(), tensor_offset_prepacked.data().begin());
  }
}


}  // namespace cuda
}  // namespace onnxruntime
