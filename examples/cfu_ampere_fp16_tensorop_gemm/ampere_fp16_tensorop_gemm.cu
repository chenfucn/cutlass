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

#include "cutlass/cutlass.h"
#include "device/quantb_gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

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
    problem_size({128, 256, 1024}),
    batch_count(1),
    reference_check(true),
    iterations(20),
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
using ElementOutput = float;                        // <- data type of elements in output matrix D

// Quantization parameter for B matrix
using ElementW = uint8_t;                           // <- Weight is int4, uint8 for two of them
// We pack 4 weights into one 16b element, so we can leverage cutlass tile iterators
// for async share memory loading, and minimizing bank conflict during matrix loading
using ElementWPack = cutlass::half_t;
using LayoutInputWPack = cutlass::layout::ColumnMajor;  // <- layout of packed weight, must be column major
using ElementQScale = cutlass::half_t;              // <- data type of quantization scale
using QuantBlocking = cutlass::MatrixShape<32,1>;   // <- weights block per scale (1,16/32/64), (16/32/64,1)
using LayoutInputQScale = 
    std::conditional<QuantBlocking::kRow == 1,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;  // <- layout of quantization scale

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
    cutlass::gemm::GemmShape<32, 128, 64>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 64>;
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
constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::QuantBGemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementQScale,
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
  // Create weight quantization scale with dimensions K x N
  cutlass::HostTensor<ElementQScale, LayoutInputQScale> tensor_scale(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});

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
      ElementW(1),
      ElementW(254),
      0);  // <- Fill weights on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_scale.host_view(),
      1,
      ElementQScale(1.5),
      ElementQScale(-1.5),
      3);  // <- Fill weight scales on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros

  // // Fill tensor_weight with the patterned data, so that we can use
  // // print to make sure the layout matches after loaded to registers
  // int loop_val = 0;
  // int offset = 3;
  // for (int col_tile = 0; col_tile < tensor_weight.extent().column()/8; ++col_tile) {
  //   for (int row_tile = 0; row_tile < tensor_weight.extent().row()/4; ++row_tile) {
  //     for (int col = 0; col < 8; ++col) {
  //       for (int row = 0; row < 4; ++row) {
  //         auto weight_cord = cutlass::make_Coord(row_tile * 4 + row, col_tile * 8 + col);
  //         auto val = (loop_val + offset) % 256;
  //         tensor_weight.at(weight_cord) = ElementW(val);
  //         loop_val++;
  //         if (loop_val == 256) {
  //           loop_val = 0;
  //           offset += 5;
  //         }
  //       }
  //     }
  //   }
  // }

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
  std::cout << "================== Matrix Scale ==========================\n";
  for (int row = 0; row < tensor_scale.extent().row(); ++row){
    for (int col = 0; col < tensor_scale.extent().column(); ++col){
      printf("%.3f, ", float(tensor_scale.at({row, col})));
    }
    printf("\n");
  }

  /////////////////////////////////////////////////////////////////////////////
  // Prepack weight matrix to facilitate matrix loading, depending on MMA
  // instruction layout.
  //
  // The weight matrix is int4, yet we want to leverage existing fp16/bf16
  // tile loading and MMA layout code in CUTLASS. So we group 4 int4 into 2
  // bytes, pretending it's fp16. This grouping must be done in a way to be
  // easily unpacked into tiles that match the MMA instruction layout.
  // For MMA instruction <16, 8, 16>, each instruction processes 2 8x8 tiles,
  // vertically stacked on the K dimension. And MmaTensorOpMultiplicandTileIterator
  // loads a <InstructionShape::kK, WarpShape::kN> tile. 
  //
  // So we stack 2x2 tiles on a 3rd dimeansion, and reshape them in a HWC layout:
  // T0, T2
  // T1, T3
  // ==>
  // T0[0, 0], T1[0, 0], T2[0, 0], T3[0, 0]
  // T0[1, 0], T1[1, 0], T2[1, 0], T3[1, 0]
  // T0[2, 0], T1[2, 0], T2[2, 0], T3[2, 0]
  // T0[3, 0], T1[3, 0], T2[3, 0], T3[3, 0]
  // ...
  // T0[0, 7], T1[0, 7], T2[0, 7], T3[0, 7]
  // T0[1, 7], T1[1, 7], T2[1, 7], T3[1, 7]
  // T0[2, 7], T1[2, 7], T2[2, 7], T3[2, 7]
  // T0[3, 7], T1[3, 7], T2[3, 7], T3[3, 7]
  //
  // This pack a 8x16 int8 tile into a 16x8 int8 tile, i.e. a 8x8 16b tile
  /////////////////////////////////////////////////////////////////////////////
  cutlass::HostTensor<ElementW, LayoutInputB> tensor_weight_prepacked(
    cutlass::make_Coord(problem_size.k(), problem_size.n()/2));
  auto t0_base = cutlass::make_Coord(0, 0);
  auto t1_base = cutlass::make_Coord(4, 0);
  auto t2_base = cutlass::make_Coord(0, 8);
  auto t3_base = cutlass::make_Coord(4, 8);
  for (int col_dtile = 0; col_dtile < tensor_weight.extent().column()/16; ++col_dtile) {
    for (int row_dtile = 0; row_dtile < tensor_weight.extent().row()/8; ++row_dtile) {
      // Packing from a 8x16 tile to a 16x8 tile
      auto dtile_base = cutlass::make_Coord(row_dtile * 8, col_dtile * 16);
      auto packed_tile_base = cutlass::make_Coord(row_dtile * 16, col_dtile * 8);
      for (int col = 0; col < 8; ++col) {
        for (int row = 0; row < 4; ++row) {
          auto cord = cutlass::make_Coord(row, col);
          auto packed_cord = packed_tile_base + cutlass::make_Coord(row * 4, col); // packed tile is 16x8
          tensor_weight_prepacked.at(packed_cord) = tensor_weight.at(dtile_base + t0_base + cord);
          tensor_weight_prepacked.at(packed_cord + cutlass::make_Coord(1, 0)) = tensor_weight.at(dtile_base + t1_base + cord);
          tensor_weight_prepacked.at(packed_cord + cutlass::make_Coord(2, 0)) = tensor_weight.at(dtile_base + t2_base + cord);
          tensor_weight_prepacked.at(packed_cord + cutlass::make_Coord(3, 0)) = tensor_weight.at(dtile_base + t3_base + cord);
        }
      }
    }
  }

  // std::cout << "Matrix Weight Prepacked:\n" << tensor_weight_prepacked.host_view() << "\n";

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_weight_prepacked.sync_device();
  tensor_scale.sync_device();
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
                                     tensor_scale.device_ref(),  // <- reference to matrix Scale on device
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

  // Record an event at the start of a series of GEMMs
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);

  // //
  // // Run profiling loop
  // //

  // for (int iter = 0; iter < options.iterations; ++iter) {
  //   // Launch initialized CUTLASS kernel
  //   status = gemm_op();
  //   CUTLASS_CHECK(status);
  // }

  // //
  // // Stop profiling loop
  // //

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

  // // Measure elapsed runtime
  // float runtime_ms = 0;
  // result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  // if (result.error != cudaSuccess) {
  //   std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
  //   return -1;
  // }

  // // Compute average runtime and GFLOPs.
  // result.runtime_ms = double(runtime_ms) / double(options.iterations);
  // result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  // Preparing reference kernel arguments
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create dequantized matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Dequantize weights and save into matrix B for reference
  for (int col = 0; col < tensor_b.extent().column(); ++col){
    for (int row = 0; row < tensor_b.extent().row(); ++row) {
      auto weight_cord = cutlass::make_Coord(row/2, col);
      int w = 0;
      if (row % 2 == 0) {
        w = int(tensor_weight.at(weight_cord) & 0x0f) - 8;
      } else {
        w = int(tensor_weight.at(weight_cord) >> 4) - 8;
      }
      auto scale_cord = cutlass::make_Coord(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
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
