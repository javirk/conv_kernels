#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/device_memory.h"

using DataType = cutlass::tfloat32_t;

using ElementInputA = DataType;
using ElementInputB = DataType;
using ElementOutput = DataType;
using ElementAccumulator = float;
using ElementComputeEpilogue = float;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

// Float16 version
// using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
// using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
// using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// Float32 version
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue>;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemmConvolution = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

float profile_cutlass_conv2d_implementation(size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats{100};
    constexpr int num_warmups{10};
    cudaStream_t stream;
    size_t const out_H{H - 2};
    size_t const out_W{W - 2};
    size_t const input_size{C_in * H * W};
    size_t const filter_size{C_out * C_in * 9};
    size_t const output_size{C_out * out_H * out_W};
    size_t const padding{0};

    ElementInputA* d_input;
    ElementInputB* d_filter;
    ElementOutput* d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(ElementInputA)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size * sizeof(ElementOutput)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filter_size * sizeof(ElementInputB)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    int split_k_slices = 1;

    cutlass::conv::Conv2dProblemSize problem_size(
        {1, (int)H, (int)W, (int)C_in},
        {(int)C_out, 3, 3, (int)C_in},  // KRSC
        {(int)padding, (int)padding, (int)padding, (int)padding},
        {1, 1},  // stride
        {1, 1},  // dilation
        {1, (int)out_H, (int)out_W, (int)C_out},  // output size NPQK
        mode,
        split_k_slices);

    cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a(
        d_input, LayoutInputA::packed({1, (int)H, (int)W, (int)C_in}));
    cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b(
        d_filter, LayoutInputB::packed({(int)C_out, 3, 3, (int)C_in}));
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c(
        d_output, LayoutOutput::packed({1, (int)out_H, (int)out_W, (int)C_out}));

    typename ImplicitGemmConvolution::Arguments arguments{
        problem_size,
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_c,
        {1.0f, 0.0f},
    };

    ImplicitGemmConvolution implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = implicit_gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS cannot implement this problem" << std::endl;
        return -1.0f;
    }

    status = implicit_gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS initialization failed" << std::endl;
        return -1.0f;
    }

    std::function<cutlass::Status(cudaStream_t)> conv_fn{
        [&](cudaStream_t s) { return implicit_gemm_op.run(s); }};
    float const latency{
        measure_performance(conv_fn, stream, num_repeats, num_warmups)};

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_filter));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}
