#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

constexpr size_t div_up(size_t a, size_t b) { return (a + b - 1) / b; }

void print_latency(std::string const& kernel_name, float latency, float tflops)
{
    std::cout << kernel_name << " - Time: " << std::fixed << std::setprecision(2)
              << latency << " ms, " << tflops << " TFLOPS" << std::endl;
}

// Include all kernels (same as main.cu)
#include "kernels/cpu.cu"
#include "kernels/0_cutlass.cu"
#include "kernels/1_naiveconv.cu"
#include "automatic_kernels/2_filter_registers.cu"
#include "automatic_kernels/4_more_oc_per_thread.cu"
#include "automatic_kernels/5_vectorized_input.cu"
#include "automatic_kernels/6_wmma_implicit_gemm.cu"
#include "automatic_kernels/7_wmma_smem_tiled.cu"
#include "automatic_kernels/9_wmma_double_buffer.cu"
#include "automatic_kernels/10_wmma_2d_spatial.cu"
#include "automatic_kernels/16_wmma_tuned.cu"
#include "automatic_kernels/18_wmma_nhwc.cu"
#include "automatic_kernels/22_wmma_nhwc_occupancy.cu"
#include "automatic_kernels/23_wmma_nhwc_fewer_warps.cu"
#include "automatic_kernels/25_wmma_nhwc_padded.cu"
#include "automatic_kernels/26_wmma_nhwc_vec4.cu"
#include "automatic_kernels/28_wmma_nhwc_vec4_filter.cu"
#include "automatic_kernels/29_wmma_nhwc_large_tile.cu"
#include "automatic_kernels/32_wmma_nhwc_transposed_filter.cu"
#include "automatic_kernels/33_wmma_nhwc_2x_n_tile.cu"
#include "automatic_kernels/36_wmma_nhwc_fp16.cu"
#include "automatic_kernels/37_wmma_nhwc_fp16_global.cu"
#include "automatic_kernels/38_wmma_nhwc_fp16_vec8.cu"
#include "automatic_kernels/39_wmma_nhwc_fp16_bk32.cu"
#include "automatic_kernels/40_wmma_nhwc_fp16_bk64.cu"
#include "automatic_kernels/43_wmma_nhwc_fp16_fused_fx.cu"

template <typename T>
float profile_conv2d_implementation(
    std::function<void(T*, T const*, T const*, size_t, size_t, size_t, size_t,
                       cudaStream_t)>
        conv_function,
    size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats{100};
    constexpr int num_warmups{10};
    cudaStream_t stream;
    size_t const out_H{H - 2};
    size_t const out_W{W - 2};
    size_t const input_size{C_in * H * W};
    size_t const filter_size{C_out * C_in * 9};
    size_t const output_size{C_out * out_H * out_W};

    T* d_input;
    T* d_output;
    T* d_filter;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filter_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const conv_wrapped{
        std::bind(conv_function, d_output, d_input, d_filter, C_in, C_out, H,
                  W, std::placeholders::_1)};
    float const latency{
        measure_performance(conv_wrapped, stream, num_repeats, num_warmups)};

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_filter));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

float calculate_tflops(size_t C_in, size_t C_out, size_t H, size_t W, float latency)
{
    size_t const R{3};
    size_t const S{3};
    size_t const out_size{(H - R + 1) * (W - S + 1) * C_out};
    float const tflops{2.0f * out_size * C_in * R * S / latency / 1e12};
    return tflops;
}

// Kernel descriptor
struct KernelEntry {
    std::string name;
    std::function<float(size_t, size_t, size_t, size_t)> profile_fn;
    size_t min_channels;  // minimum C_in/C_out (must be multiple of this)
};

int main()
{
    // Build the kernel table
    std::vector<KernelEntry> kernels;

    // CUTLASS reference
    kernels.push_back({"CUTLASS",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_cutlass_conv2d_implementation(ci, co, h, w);
        }, 1});

    // Naive
    kernels.push_back({"01_Naive",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_conv2d_implementation<float>(&launch_naive_conv2d_3x3<float>, ci, co, h, w);
        }, 1});

    // Filter registers
    kernels.push_back({"02_FilterReg",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_conv2d_implementation<float>(&launch_filter_reg_conv2d_3x3<float>, ci, co, h, w);
        }, 1});

    // More OC per thread
    kernels.push_back({"04_MoreOC",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_conv2d_implementation<float>(&launch_more_oc_conv2d_3x3<float>, ci, co, h, w);
        }, 1});

    // Vectorized input
    kernels.push_back({"05_VecInput",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_conv2d_implementation<float>(&launch_vec_input_conv2d_3x3<float>, ci, co, h, w);
        }, 1});

    // WMMA implicit gemm
    kernels.push_back({"06_WMMA",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_conv2d_implementation<float>(&launch_wmma_conv2d_3x3<float>, ci, co, h, w);
        }, 16});

    // WMMA SMEM tiled
    kernels.push_back({"07_WMMA_SMEM",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_conv2d_implementation<float>(&launch_wmma_smem_tiled_conv2d_3x3<float>, ci, co, h, w);
        }, 16});

    // NHWC kernels with custom profile functions
    kernels.push_back({"18_NHWC",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"22_NHWC_Occupancy",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_occupancy_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"23_NHWC_FewerWarps",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_fewer_warps_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"25_NHWC_Padded",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_padded_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"26_NHWC_Vec4",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_vec4_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"28_NHWC_Vec4Filter",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_vec4_filter_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"29_NHWC_LargeTile",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_large_tile_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"32_NHWC_TransFilter",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_transposed_filter_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"33_NHWC_2xN",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_2x_n_tile_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"36_NHWC_FP16",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_fp16_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"37_NHWC_FP16Global",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_fp16_global_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"38_NHWC_FP16Vec8",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_fp16_vec8_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"39_NHWC_FP16_BK32",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_fp16_bk32_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"40_NHWC_FP16_BK64",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_fp16_bk64_conv2d(ci, co, h, w);
        }, 16});

    kernels.push_back({"43_NHWC_FP16_FusedFx",
        [](size_t ci, size_t co, size_t h, size_t w) {
            return profile_nhwc_fp16_fused_fx_conv2d(ci, co, h, w);
        }, 16});

    // Sweep parameters
    std::vector<size_t> resolutions = {128, 256, 512, 768, 1024};
    std::vector<size_t> channels = {8, 16, 32, 64, 128, 256};

    // Print TSV header
    std::cout << "resolution\tC_in\tC_out\tkernel\ttime_ms\ttflops" << std::endl;

    for (size_t res : resolutions) {
        for (size_t ch : channels) {
            size_t C_in = ch;
            size_t C_out = ch;
            size_t H = res;
            size_t W = res;

            for (auto const& k : kernels) {
                // Skip if channels don't meet minimum requirement
                if (k.min_channels > 1 && (C_in % k.min_channels != 0 || C_out % k.min_channels != 0)) {
                    continue;
                }

                float latency = k.profile_fn(C_in, C_out, H, W);
                float tflops = calculate_tflops(C_in, C_out, H, W, latency);

                std::cout << res << "\t" << C_in << "\t" << C_out << "\t"
                          << k.name << "\t"
                          << std::fixed << std::setprecision(4) << latency << "\t"
                          << std::setprecision(4) << tflops << std::endl;

                // Flush after each result so we can monitor progress
                std::cout.flush();
            }
        }
    }

    return 0;
}
