/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <iostream>
#include <variant>

#include "cutlass/cutlass.h"
#include "device/quantb_gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "gsl/gsl"

#include "helper.h"

#define USE_QUANT_OFFSET 1

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// Duplicate cutlass tensor utils so that it can be used in onnxruntime CPU code.
// Simple appearance of cutlass includes seems to cause strange compilation with RE2

/// Position<> is a simple structure used to represent tensor coordinates
template <
  int Rank_,                          ///< Logical rank of coordinate
  typename Index_ = int,              ///< Index type used for each dimension
  typename LongIndex_ = int64_t       ///< Long index type used for linear offsets
>
struct Position {

public:

  //
  // Type and constant definitions
  //

  /// Number of elements in Position
  static int const kRank = Rank_;

  /// Index type used to store elements
  using Index = Index_;

  /// Type used to represent linear offsets
  using LongIndex = LongIndex_;

private:

  //
  // Data members
  //

  /// Indices
  Index idx[kRank];

public:

  //
  // Methods
  //

  /// Default ctor initializes uniformly
  __forceinline__
  explicit Position(Index value = Index(0)) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = value;
    }
  }

  /// Constructs from an array of integers
  __forceinline__
  Position(Index const (&_idx)[kRank]) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = _idx[i];
    }
  }

  /// Constructs from some other Position
  template <int R, typename I, typename L>
  __forceinline__
  Position(Position<R, I, L> other) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = other[i];
    }
  }

  /// Element-wise addition
  __forceinline__
  Position operator+(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] + b.idx[i];
    }
    return c;
  }

  /// Element-wise subtraction
  __forceinline__
  Position operator-(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] - b.idx[i];
    }
    return c;
  }

  /// Element-wise multiplication
  __forceinline__
  Position operator*(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] * b.idx[i];
    }
    return c;
  }

  /// Element-wise division
  __forceinline__
  Position operator/(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] / b.idx[i];
    }
    return c;
  }

  /// In-place addition
  __forceinline__
  Position& operator+=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] += b.idx[i];
    }
    return *this;
  }

  /// In-place subtraction
  __forceinline__
  Position& operator-=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] -= b.idx[i];
    }
    return *this;
  }

  /// In-place multiplication
  __forceinline__
  Position& operator*=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] *= b.idx[i];
    }
    return *this;
  }

  /// In-place division
  __forceinline__
  Position& operator/=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] /= b.idx[i];
    }
    return *this;
  }

  /// Member access operator
  __forceinline__ Index& operator[](int dim) { return idx[dim]; }

  /// Member access operator
  __forceinline__ Index const& operator[](int dim) const { return idx[dim]; }

  /// Determines if two Position<> objects are equal
  __forceinline__
  bool operator==(Position const& b) const {
    bool equal = true;
    for (int i = 0; equal && i < kRank; ++i) {
      equal = (idx[i] == b.idx[i]);
    }
    return equal;
  }

  /// Not equal
  __forceinline__
  bool operator!=(Position const& b) const { return !(*this == b); }

  /// Clamps a coordinate to a range specified by maximum and minimum values
  __forceinline__
  Position& clamp(Position const& max, Position const& min = Position()) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = __NV_STD_MAX(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
    }
    return *this;
  }

  /// Returns the sum of all elements
  __forceinline__
  Index sum() const {
    Index sum_(idx[0]);
    for (int i = 1; i < kRank; ++i) {
      sum_ += idx[i];
    }
    return sum_;
  }

  /// Returns the product of all elements
  __forceinline__
  LongIndex product() const {
    LongIndex product_(idx[0]);
    for (int i = 1; i < kRank; ++i) {
      product_ *= idx[i];
    }
    return product_;
  }

};

template <typename T, typename L=int64_t> 
Position<2, T, L> make_Position(T _0, T _1) {
  T values[2] = {_0, _1};
  return Position<2, T, L>(values);
}

template <typename T, typename L=int64_t> 
Position<3, T, L> make_Position(T _0, T _1, T _2) {
  T values[3] = {_0, _1, _2};
  return Position<2, T, L>(values);
}


class RowMajorLayout {
public:

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using MatCoord = Position<2, Index, LongIndex>;

private:
  //
  // Data members
  //

  /// Stride data member
  Index stride_;

public:
  //
  // Methods
  //

  /// Constructor
  __forceinline__
  RowMajorLayout(Index ldm = 0): stride_(ldm) { }

  /// Helper returns a layout to a tightly packed tensor
  __forceinline__
  static RowMajorLayout packed(MatCoord const &extent) {
    return RowMajorLayout(extent[1]);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (row, column)
  __forceinline__
  LongIndex operator()(MatCoord const &coord) const {
    return LongIndex(coord[0]) * stride_ + coord[1];
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  __forceinline__
  MatCoord inverse(LongIndex offset) const {
    return make_Position(Index(offset / stride_), Index(offset % stride_));
  }

  /// Returns the stride of the layout
  __forceinline__
  Index stride() const {
    return stride_;
  }
};

class ColumnMajorLayout {
public:

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using MatCoord = Position<2, Index, LongIndex>;

private:
  //
  // Data members
  //

  /// Stride data member
  Index stride_;

public:
  //
  // Methods
  //

  /// Ctor
  __forceinline__
  ColumnMajorLayout(Index ldm = 0): stride_(ldm) { }


  /// Helper returns a layout to a tightly packed tensor
  __forceinline__
  static ColumnMajorLayout packed(MatCoord const &extent) {
    return ColumnMajorLayout(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (row, column)
  __forceinline__
  LongIndex operator()(MatCoord const &coord) const {
    return LongIndex(coord[1]) * LongIndex(stride_) + coord[0];
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  __forceinline__
  MatCoord inverse(LongIndex offset) const {
    return make_Position(Index(offset % stride_), Index(offset / stride_));
  }

  /// Returns the stride of the layout
  __forceinline__
  Index stride() const {
    return stride_;
  }

};

template <
  /// Data type of element stored within tensor (concept: NumericType)
  typename Element_,
  /// Defines a mapping from logical coordinate to linear memory (concept: Layout)
  typename Layout_
>
class MatrixRef {
 public:
  /// Data type of individual access
  using Element = Element_;

  using Reference = Element &;

  /// Mapping function from logical coordinate to linear memory
  using Layout = Layout_;

  /// Index type
  using Index = typename Layout::Index;

  /// Long index used for pointer offsets
  using LongIndex = typename Layout::LongIndex;

  /// Coordinate in logical tensor space
  using MatCoord = Position<2>;// typename Layout::MatCoord;

  /// MatrixRef to constant data
  using ConstMatrixRef = MatrixRef<
    typename std::remove_const<Element>::type const,
    Layout>;

  /// MatrixRef to non-constant data
  using NonConstMatrixRef = MatrixRef<
    typename std::remove_const<Element>::type,
    Layout>;

 private:

  /// Pointer
  gsl::span<Element> data_;

  /// Shape of matrix
  MatCoord shape_;

  /// Layout object maps logical coordinates to linear offsets
  Layout layout_;

 public:

  //
  // Methods
  //

  /// Constructs a MatrixRef with a pointer and layout object.
  __forceinline__
  MatrixRef(): data_() {}

  /// Constructs a MatrixRef with a span and layout object.
  __forceinline__
  MatrixRef(
    gsl::span<Element> const &data, ///< pointer to start of tensor
    MatCoord const &shape           ///< shape of tensor
  ):
    data_(data), shape_(shape), layout_(Layout::packed(shape)) {
    Expects(data_.size() >= shape_.product());
  }

  /// Constructs a MatrixRef with a pointer and layout object.
  __forceinline__
  MatrixRef(
    Element * ptr,                  ///< pointer to start of tensor
    LongIndex size,                 ///< size of tensor in elements
    MatCoord const &shape           ///< shape of tensor
  ):
    data_(ptr, size), shape_(shape), layout_(Layout::packed(shape)) {
    Expects(data_.size() >= shape_.product());
  }

  /// Converting constructor from MatrixRef to non-constant data.
  template<typename _Magic = int>
  __forceinline__
  MatrixRef(
    NonConstMatrixRef const &ref,              ///< MatrixRef to non-const data
    ///SFINAE trick to avoid creating a copy-constructor when Element_ is already non-const
    _Magic magic = (typename std::enable_if< !std::is_same<NonConstMatrixRef, MatrixRef<Element_, Layout_> >::value, _Magic>::type)0
  ):
    data_(ref.data()), shape_(ref.shape()), layout_(Layout::packed(ref.shape())) { }

  /// Returns a reference to constant-valued tensor.
  __forceinline__
  ConstMatrixRef const_ref() const {
    return ConstMatrixRef(data_, shape_);
  }

  __forceinline__
  NonConstMatrixRef non_const_ref() const {
    return NonConstMatrixRef(const_cast<typename std::remove_const<Element>::type *>(data_.data()), data_.size(), shape_);
  }

  /// Returns true if the MatrixRef is non-null
  __forceinline__
  bool good() const {
    return !data_.empty();
  }

  /// Returns the pointer to referenced data
  __forceinline__
  gsl::span<Element> const& data() const {
    return data_;
  }

  /// Returns the shape of the tensor
  __forceinline__
  MatCoord const& shape() const {
    return shape_;
  }

  /// Returns the layout object
  __forceinline__
  Layout & layout() {
    return layout_;
  }

  /// Returns the layout object
  __forceinline__
  Layout layout() const {
    return layout_;
  }

  /// Returns the layout object's stride vector
  __forceinline__
  Index stride() const {
    return layout_.stride();
  }

  /// Returns the layout object's stride vector
  __forceinline__
  Index & stride() {
    return layout_.stride();
  }

  /// Computes the offset of an index from the origin of the tensor
  __forceinline__
  LongIndex offset(MatCoord const& coord) const {
    return layout_(coord);
  }

  /// Returns a reference to the element at a given Coord
  __forceinline__
  Reference at(MatCoord const& coord) const {
    return data_[offset(coord)];
  }

  __forceinline__
  Reference at(int row, int col) const {
    return data_[offset(make_Position(row, col))];
  }

  /// Returns a reference to the element at a given Coord
  __forceinline__
  Reference operator[](MatCoord const& coord) const {
    return data_[offset(coord)];
  }

};

/// Constructs a MatrixRef, deducing types from arguments.
template <
  typename Element,
  typename Layout
>
__forceinline__
MatrixRef<Element, Layout> make_MatrixRef(Element *ptr, int64_t size, Layout const &layout, typename Layout::MatCoord const &shape) {
  return MatrixRef<Element, Layout>(ptr, size, layout, shape);
}

template <
  typename Element,
  typename LayoutCutlass,
  typename Layout = std::conditional_t<std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value, ColumnMajorLayout, RowMajorLayout>
  >
__forceinline__
MatrixRef<Element, Layout> make_MatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
  static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value
                || std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
  auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
  auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.host_data());
  return MatrixRef<Element, Layout>(ptr, tensor.capacity(), shape);
}

template <
  typename Element,
  typename LayoutCutlass,
  typename Layout = std::conditional_t<std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value, ColumnMajorLayout, RowMajorLayout>
  >
__forceinline__
MatrixRef<Element const, Layout> make_ConstMatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
  static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value
                || std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
  auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
  return MatrixRef<Element const, Layout>(tensor.host_data(), tensor.capacity(), shape);
}


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
void prepack_weights(size_t rows,
                     size_t columns,
                     MatrixRef<uint8_t const, ColumnMajorLayout> tensor_weight,
                     MatrixRef<uint8_t, ColumnMajorLayout> tensor_weight_prepacked) {
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

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_size({1024, 2048, 1024}),
    batch_count(1),
    reference_check(true),
    iterations(200),
    alpha(1),
    beta() { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "14_ampere_tf32_tensorop_gemm example\n\n"
      << "  This example uses the CUTLASS Library to execute TF32 tensorop GEMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

// Quantization parameter for B matrix
using ElementW = uint8_t;                           // <- Weight is int4, uint8 for two of them
// We pack 4 weights into one 16b element, so we can leverage cutlass tile iterators
// for async share memory loading, and minimizing bank conflict during matrix loading
using ElementWPack = cutlass::half_t;
using LayoutInputWPack = cutlass::layout::ColumnMajor;  // <- layout of packed weight, must be column major
using ElementQScale = cutlass::half_t;              // <- data type of quantization scale
#ifdef USE_QUANT_OFFSET
using ElementQOffset = uint8_t;                     // <- data type of quantization offset
#endif
using QuantBlocking = cutlass::MatrixShape<1,32>;   // <- weights block per scale (1,16/32/64), (16/32/64,1)
using LayoutInputQScale = 
    std::conditional<QuantBlocking::kRow == 1,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;  // <- layout of quantization scale

template <typename Layout>
void prepack_quant_scales(
    size_t rows,
    size_t columns,
    MatrixRef<cutlass::half_t const, Layout> tensor_scale,
    MatrixRef<cutlass::half_t, Layout> tensor_scale_prepacked) {
  Expects(tensor_scale.shape()[0] == (rows / QuantBlocking::kRow) && tensor_scale.shape()[1] == (columns / QuantBlocking::kColumn));
  Expects(tensor_scale_prepacked.shape() == tensor_scale.shape());

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


template <typename Layout>
void prepack_quant_offsets(
    size_t rows,
    size_t columns,
    MatrixRef<uint8_t const, Layout> tensor_offset,
    MatrixRef<uint8_t, Layout> tensor_offset_prepacked) {
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

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 64>; // <64, 128, 64>
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>; // <64, 32, 64>
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::QuantBGemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementQScale,
#ifdef USE_QUANT_OFFSET
                                         ElementQOffset,
#else
                                         std::monostate,  // <- no quantization offset
#endif
                                         LayoutInputQScale,
                                         QuantBlocking,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

int run(Options &options) {

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = options.problem_size;

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K

  // Create weight matrix with dimensions K x N.
  // Actual weight type is int4, we use ElementW = uint8 to avoid possible compilation
  // troubles. Since the layout is column major, we are packing 2 weights in a column
  // into one int8
  cutlass::HostTensor<ElementW, LayoutInputB> tensor_weight(
      {problem_size.k()/2, problem_size.n()});
  // Create weight quantization scale and offset with dimensions K x N
  cutlass::HostTensor<ElementQScale, LayoutInputQScale> tensor_scale(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});
#ifdef USE_QUANT_OFFSET
  cutlass::HostTensor<ElementQOffset, LayoutInputQScale> tensor_offset(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});
#endif

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      2);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_weight.host_view(),
      1,
      ElementW(0),
      ElementW(255),
      0);  // <- Fill weights on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_scale.host_view(),
      1,
      ElementQScale(1.5),
      ElementQScale(-1.5),
      5);  // <- Fill weight scales on host with uniform-distribution random data
#ifdef USE_QUANT_OFFSET
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_offset.host_view(),
      1,
      ElementQOffset(0),
      ElementQOffset(15),
      0);  // <- Fill weight offsets on host with uniform-distribution random data
#endif
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros

//   // Fill tensor_weight with the patterned data, so that we can use
//   // print to make sure the layout matches after loaded to registers
//   int loop_val = 0;
//   int offset = 3;
//   for (int col_tile = 0; col_tile < tensor_weight.extent().column()/8; ++col_tile) {
//     for (int row_tile = 0; row_tile < tensor_weight.extent().row()/4; ++row_tile) {
//       for (int col = 0; col < 8; ++col) {
//         for (int row = 0; row < 4; ++row) {
//           auto weight_cord = cutlass::make_Coord(row_tile * 4 + row, col_tile * 8 + col);
//           auto val = (loop_val + offset) % 256;
//           tensor_weight.at(weight_cord) = ElementW(val);
//           loop_val++;
//           if (loop_val == 256) {
//             loop_val = 0;
//             offset += 11;
//           }
//         }
//       }
//     }
//   }
//   for (int col = 0; col < tensor_scale.extent().column(); ++col){
//     int c =  col * QuantBlocking::kColumn;
//     for (int row = 0; row < tensor_scale.extent().row(); ++row){
//       int r = row * QuantBlocking::kRow;
//       auto weight_cord = cutlass::make_Coord(r/2, c);
//       int w = 0;
//       if (r % 2 == 0) {
//         w = int(tensor_weight.at(weight_cord) & 0x0f);
//       } else {
//         w = int(tensor_weight.at(weight_cord) >> 4);
//       }
//       tensor_scale.at({row, col}) = w;
// #ifdef USE_QUANT_OFFSET
//       tensor_offset.at({row, col}) = ElementQOffset(w);
// #endif
//     }
//   }

  // int fill_val = -512;
  // int factor = 1;
  // for (int col = 0; col < tensor_scale.extent().column(); ++col){
  //   for (int row = 0; row < tensor_scale.extent().row(); ++row){
  //     tensor_scale.at({row, col}) = ElementQScale((float)fill_val * float(factor));
  //     fill_val++;
  //     if (fill_val == 512) {
  //       fill_val = -512;
  //       factor += 1;
  //     }
  //   }
  // }

  // std::cout << "Matrix Weight:\n" << tensor_weight.host_view() << "\n";

  std::cout << "Prepacking weight matrix and quantization meta data ...\n";

  cutlass::HostTensor<ElementW, LayoutInputB> tensor_weight_prepacked(
    cutlass::make_Coord(problem_size.k(), problem_size.n()/2));
  prepack_weights(problem_size.k(), problem_size.n(),
                  make_ConstMatrixRef(tensor_weight),
                  make_MatrixRef(tensor_weight_prepacked));

  // std::cout << "Matrix Weight Prepacked:\n" << tensor_weight_prepacked.host_view() << "\n";

  cutlass::HostTensor<ElementQScale, LayoutInputQScale> tensor_scale_prepacked(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});
#ifdef USE_QUANT_OFFSET
  cutlass::HostTensor<ElementQOffset, LayoutInputQScale> tensor_offset_prepacked(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});
#endif

  prepack_quant_scales(problem_size.k(), problem_size.n(),
                       make_ConstMatrixRef(tensor_scale),
                       make_MatrixRef(tensor_scale_prepacked));
#ifdef USE_QUANT_OFFSET
  prepack_quant_offsets(problem_size.k(), problem_size.n(),
                        make_ConstMatrixRef(tensor_offset),
                        make_MatrixRef(tensor_offset_prepacked));
#endif

  // std::cout << "================== Matrix Scale ==========================\n";
  // for (int row = 0; row < tensor_scale_prepacked.extent().row(); ++row){
  //   for (int col = 0; col < tensor_scale_prepacked.extent().column(); ++col){
  //     printf("%.0f, ", float(tensor_scale_prepacked.at({row, col})));
  //   }
  //   printf("\n");
  // }

  std::cout << "Copy data from host to GPU...\n";
  tensor_a.sync_device();
  tensor_weight_prepacked.sync_device();
  tensor_scale_prepacked.sync_device();
#ifdef USE_QUANT_OFFSET
  tensor_offset_prepacked.sync_device();
#endif
  tensor_c.sync_device();
  tensor_d.sync_device();
  cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_W(
    reinterpret_cast<ElementWPack const *>(tensor_weight_prepacked.device_data()),
    LayoutInputWPack::packed({problem_size.k()/2, problem_size.n()/2}));

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(options.alpha);
  ElementComputeEpilogue beta = ElementComputeEpilogue(options.beta);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     ref_W,                  // <- reference to packed weights on device
                                     tensor_scale_prepacked.device_ref(),  // <- reference to quant scale on device
#ifdef USE_QUANT_OFFSET
                                     tensor_offset_prepacked.device_ref(),  // <- reference to quant offset on device
#endif
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Result structure
  Result result;

  //
  // Construct events
  //

  cudaEvent_t events[2];

  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
  }

  std::cout << "Running Quantized Gemm...\n";

  // Record an event at the start of a series of GEMMs
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  //
  // Run profiling loop
  //

  for (int iter = 0; iter < options.iterations; ++iter) {
    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);
  }

  //
  // Stop profiling loop
  //

  // Record an event when the GEMMs are complete
  result.error = cudaEventRecord(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Compute average runtime and GFLOPs.
  result.runtime_ms = double(runtime_ms) / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  // Preparing reference kernel arguments
  std::cout << "Dequantizing weights and running reference kernel...\n";

  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create dequantized matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Dequantize weights and save into matrix B for reference
  for (int col = 0; col < tensor_b.extent().column(); ++col){
    for (int row = 0; row < tensor_b.extent().row(); ++row) {
      auto weight_cord = cutlass::make_Coord(row/2, col);
      auto scale_cord = cutlass::make_Coord(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
#ifdef USE_QUANT_OFFSET
      const uint8_t offset = tensor_offset.at(scale_cord);
#else
      const uint8_t offset = 8;
#endif
      int w = 0;
      if (row % 2 == 0) {
        w = int(tensor_weight.at(weight_cord) & 0x0f) - offset;
      } else {
        w = int(tensor_weight.at(weight_cord) >> 4) - offset;
      }
      auto scale = tensor_scale.at(scale_cord);
      tensor_b.at({row, col}) = scale * float(w);
    }
  }
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  tensor_b.sync_device();
  tensor_ref_d.sync_device();

  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device;

  // Launch device reference gemm kernel
  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  if (passed) {
    std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << " GFLOPs: " << result.gflops << std::endl;
  }

  std::cout << (passed ? "Passed" : "Failed") << std::endl;

  return (passed ? 0  : -1);
}

int main(int argc, const char **argv) {
  
  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  std::cout << "Device: " << props.name << " with " << props.multiProcessorCount << " SMs" << std::endl;
  std::cout << "Device compute capability: " << props.major << "." << props.minor << std::endl;

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d TF32 tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}
