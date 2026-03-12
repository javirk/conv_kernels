#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
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

// These functions use CHECK_CUDA_ERROR and CHECK_LAST_CUDA_ERROR macros.
#include "kernels/cpu.cu"  // This is for unit testing
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
    size_t const out_size{(H - R + 1) * (W - S + 1) * C_out};  // H W K
    float const tflops{2.0f * out_size * C_in * R * S / latency / 1e12};
    return tflops;
}

int main()
{
    size_t const C_in{64};
    size_t const C_out{64};
    size_t const H{1024};
    size_t const W{1024};

    // Unit tests.
    for (size_t h{3}; h <= 16; ++h)
    {
        for (size_t w{3}; w <= 16; ++w)
        {
            assert(verify_conv2d_implementation<float>(
                &launch_wmma_tuned_conv2d_3x3<float>, 1, 1, h, w));
        }
    }
    assert(verify_conv2d_implementation<float>(&launch_wmma_tuned_conv2d_3x3<float>, C_in, C_out, 32, 32));
    std::cout << "Unit tests passed." << std::endl;

    // Profiling CUTLASS convolution for reference.
    float const latency_cutlass{profile_cutlass_conv2d_implementation(C_in, C_out, H, W)};
    float const tflops_cutlass{calculate_tflops(C_in, C_out, H, W, latency_cutlass)};
    print_latency("CUTLASS 3x3 Conv2D", latency_cutlass, tflops_cutlass);

    // Profiling.
    // std::cout << C_in << " -> " << C_out << " channels, " << H << " x " << W
    //           << " spatial" << std::endl;
    float const latency_naive{profile_conv2d_implementation<float>(&launch_naive_conv2d_3x3<float>, C_in, C_out, H, W)};
    float const tflops_naive{calculate_tflops(C_in, C_out, H, W, latency_naive)};
    print_latency("1. Naive 3x3 Conv2D", latency_naive, tflops_naive);

    float const latency2{profile_conv2d_implementation<float>(&launch_filter_reg_conv2d_3x3<float>, C_in, C_out, H, W)};
    float const tflops2{calculate_tflops(C_in, C_out, H, W, latency2)};
    print_latency("2. Filter Reg 3x3 Conv2D", latency2, tflops2);

    float const latency4{profile_conv2d_implementation<float>(&launch_more_oc_conv2d_3x3<float>, C_in, C_out, H, W)};
    float const tflops4{calculate_tflops(C_in, C_out, H, W, latency4)};
    print_latency("4. More OC/Thread 3x3 Conv2D", latency4, tflops4);

    float const latency5{profile_conv2d_implementation<float>(&launch_vec_input_conv2d_3x3<float>, C_in, C_out, H, W)};
    float const tflops5{calculate_tflops(C_in, C_out, H, W, latency5)};
    print_latency("5. Vec Input 3x3 Conv2D", latency5, tflops5);

    float const latency6{profile_conv2d_implementation<float>(&launch_wmma_conv2d_3x3<float>, C_in, C_out, H, W)};
    float const tflops6{calculate_tflops(C_in, C_out, H, W, latency6)};
    print_latency("6. WMMA Conv2D", latency6, tflops6);

    float const latency7{profile_conv2d_implementation<float>(&launch_wmma_smem_tiled_conv2d_3x3<float>, C_in, C_out, H, W)};
    float const tflops7{calculate_tflops(C_in, C_out, H, W, latency7)};
    print_latency("7. WMMA SMEM Tiled Conv2D", latency7, tflops7);

    float const latency{profile_conv2d_implementation<float>(&launch_wmma_tuned_conv2d_3x3<float>, C_in, C_out, H, W)};
    float const tflops{calculate_tflops(C_in, C_out, H, W, latency)};
    print_latency("16. WMMA Tuned Conv2D", latency, tflops);



    return 0;
}
