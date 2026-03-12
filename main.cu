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
                &launch_wmma_nhwc_fp16_bk64_conv2d_3x3<float>, 1, 1, h, w));
        }
    }
    assert(verify_conv2d_implementation<float>(&launch_wmma_nhwc_fp16_bk64_conv2d_3x3<float>, C_in, C_out, 32, 32));
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

    float const latency{profile_nhwc_conv2d(C_in, C_out, H, W)};
    float const tflops{calculate_tflops(C_in, C_out, H, W, latency)};
    print_latency("18. WMMA NHWC Conv2D", latency, tflops);

    float const latency22{profile_nhwc_occupancy_conv2d(C_in, C_out, H, W)};
    float const tflops22{calculate_tflops(C_in, C_out, H, W, latency22)};
    print_latency("22. WMMA NHWC Occupancy Conv2D", latency22, tflops22);

    float const latency23{profile_nhwc_fewer_warps_conv2d(C_in, C_out, H, W)};
    float const tflops23{calculate_tflops(C_in, C_out, H, W, latency23)};
    print_latency("23. WMMA NHWC FewerWarps Conv2D", latency23, tflops23);

    float const latency25{profile_nhwc_padded_conv2d(C_in, C_out, H, W)};
    float const tflops25{calculate_tflops(C_in, C_out, H, W, latency25)};
    print_latency("25. WMMA NHWC Padded Conv2D", latency25, tflops25);

    float const latency26{profile_nhwc_vec4_conv2d(C_in, C_out, H, W)};
    float const tflops26{calculate_tflops(C_in, C_out, H, W, latency26)};
    print_latency("26. WMMA NHWC Vec4 Conv2D", latency26, tflops26);

    float const latency28{profile_nhwc_vec4_filter_conv2d(C_in, C_out, H, W)};
    float const tflops28{calculate_tflops(C_in, C_out, H, W, latency28)};
    print_latency("28. WMMA NHWC Vec4Filter Conv2D", latency28, tflops28);

    float const latency29{profile_nhwc_large_tile_conv2d(C_in, C_out, H, W)};
    float const tflops29{calculate_tflops(C_in, C_out, H, W, latency29)};
    print_latency("29. WMMA NHWC LargeTile Conv2D", latency29, tflops29);

    float const latency32{profile_nhwc_transposed_filter_conv2d(C_in, C_out, H, W)};
    float const tflops32{calculate_tflops(C_in, C_out, H, W, latency32)};
    print_latency("32. WMMA NHWC TransFilter Conv2D", latency32, tflops32);

    float const latency33{profile_nhwc_2x_n_tile_conv2d(C_in, C_out, H, W)};
    float const tflops33{calculate_tflops(C_in, C_out, H, W, latency33)};
    print_latency("33. WMMA NHWC 2xN Tile Conv2D", latency33, tflops33);

    float const latency36{profile_nhwc_fp16_conv2d(C_in, C_out, H, W)};
    float const tflops36{calculate_tflops(C_in, C_out, H, W, latency36)};
    print_latency("36. WMMA NHWC FP16 Conv2D", latency36, tflops36);

    float const latency37{profile_nhwc_fp16_global_conv2d(C_in, C_out, H, W)};
    float const tflops37{calculate_tflops(C_in, C_out, H, W, latency37)};
    print_latency("37. WMMA NHWC FP16Global Conv2D", latency37, tflops37);

    float const latency38{profile_nhwc_fp16_vec8_conv2d(C_in, C_out, H, W)};
    float const tflops38{calculate_tflops(C_in, C_out, H, W, latency38)};
    print_latency("38. WMMA NHWC FP16Vec8 Conv2D", latency38, tflops38);

    float const latency39{profile_nhwc_fp16_bk32_conv2d(C_in, C_out, H, W)};
    float const tflops39{calculate_tflops(C_in, C_out, H, W, latency39)};
    print_latency("39. WMMA NHWC FP16 BK32 Conv2D", latency39, tflops39);

    float const latency40{profile_nhwc_fp16_bk64_conv2d(C_in, C_out, H, W)};
    float const tflops40{calculate_tflops(C_in, C_out, H, W, latency40)};
    print_latency("40. WMMA NHWC FP16 BK64 Conv2D", latency40, tflops40);

    return 0;
}
