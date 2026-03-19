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
#include "automatic_kernels/1_wmma_implicit_gemm_nhwc.cu"
#include "automatic_kernels/2_wmma_large_tile_bk32.cu"
#include "automatic_kernels/7_wmma_double_buffer.cu"
#include "automatic_kernels/9_wmma_native_half.cu"
#include "automatic_kernels/11_wmma_half_coalesced.cu"
#include "automatic_kernels/12_wmma_half_reg_precomp.cu"
#include "automatic_kernels/15_wmma_half_padded_ldg.cu"
#include "automatic_kernels/26_parallel_warp_store.cu"
#include "automatic_kernels/27_coalesced_B_load.cu"
#include "automatic_kernels/29_precomp_k_offsets.cu"
#include "automatic_kernels/33_int_addr_no_spill.cu"

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

// Half-precision verification: uses float CPU reference, half GPU kernel
bool verify_half_conv2d(
    std::function<void(half*, half const*, half const*, size_t, size_t, size_t, size_t, cudaStream_t)> conv_function,
    size_t C_in, size_t C_out, size_t H, size_t W)
{
    std::mt19937 gen{0};
    cudaStream_t stream;
    size_t const out_H{H - 2};
    size_t const out_W{W - 2};
    size_t const input_size{C_in * H * W};
    size_t const filter_size{C_out * C_in * 9};
    size_t const output_size{C_out * out_H * out_W};

    std::vector<float> input_f(input_size);
    std::vector<float> filter_f(filter_size);
    std::vector<float> output_ref(output_size, 0.0f);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : input_f) v = dist(gen);
    for (auto& v : filter_f) v = dist(gen);

    // Convert to half for GPU
    std::vector<half> input_h(input_size);
    std::vector<half> filter_h(filter_size);
    for (size_t i = 0; i < input_size; i++) input_h[i] = __float2half(input_f[i]);
    for (size_t i = 0; i < filter_size; i++) filter_h[i] = __float2half(filter_f[i]);

    // CPU reference using half-converted values for fair comparison
    for (size_t oc = 0; oc < C_out; oc++)
        for (size_t r = 0; r < out_H; r++)
            for (size_t c = 0; c < out_W; c++) {
                float sum = 0.0f;
                for (size_t ic = 0; ic < C_in; ic++)
                    for (int fy = 0; fy < 3; fy++)
                        for (int fx = 0; fx < 3; fx++)
                            sum += __half2float(input_h[ic * H * W + (r + fy) * W + (c + fx)]) *
                                   __half2float(filter_h[oc * C_in * 9 + ic * 9 + fy * 3 + fx]);
                output_ref[oc * out_H * out_W + r * out_W + c] = sum;
            }

    half *d_input, *d_output, *d_filter;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filter_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input_h.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filter, filter_h.data(), filter_size * sizeof(half), cudaMemcpyHostToDevice));

    conv_function(d_output, d_input, d_filter, C_in, C_out, H, W, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<half> output_h(output_size);
    CHECK_CUDA_ERROR(cudaMemcpy(output_h.data(), d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost));

    bool correct = true;
    for (size_t i = 0; i < output_size; i++) {
        float diff = std::abs(__half2float(output_h[i]) - output_ref[i]);
        float ref_abs = std::abs(output_ref[i]);
        // Use relative tolerance for FP16
        if (diff > 0.05f * ref_abs + 0.01f) {
            correct = false;
            break;
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_filter));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return correct;
}

// Half-precision profiling
float profile_half_conv2d(
    std::function<void(half*, half const*, half const*, size_t, size_t, size_t, size_t, cudaStream_t)> conv_function,
    size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats{10};
    constexpr int num_warmups{5};
    cudaStream_t stream;
    size_t const out_H{H - 2};
    size_t const out_W{W - 2};
    size_t const input_size{C_in * H * W};
    size_t const filter_size{C_out * C_in * 9};
    size_t const output_size{C_out * out_H * out_W};

    half *d_input, *d_output, *d_filter;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filter_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const conv_wrapped{
        std::bind(conv_function, d_output, d_input, d_filter, C_in, C_out, H, W, std::placeholders::_1)};
    float const latency{measure_performance(conv_wrapped, stream, num_repeats, num_warmups)};

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
    size_t const C_in{256};
    size_t const C_out{256};
    size_t const H{512};
    size_t const W{512};

    std::cout << "Profiling " << C_in << " -> " << C_out << " channels, " << H << " x " << W << std::endl;

    // Unit tests (half precision).
    for (size_t h{3}; h <= 16; ++h)
        for (size_t w{3}; w <= 16; ++w)
            assert(verify_half_conv2d(&launch_wmma_half_padded_ldg_conv2d_3x3, 1, 1, h, w));
    assert(verify_half_conv2d(&launch_wmma_half_padded_ldg_conv2d_3x3, C_in, C_out, 32, 32));
    for (size_t h{3}; h <= 16; ++h)
        for (size_t w{3}; w <= 16; ++w)
            assert(verify_half_conv2d(&launch_wmma_parallel_store_conv2d_3x3, 1, 1, h, w));
    assert(verify_half_conv2d(&launch_wmma_parallel_store_conv2d_3x3, C_in, C_out, 32, 32));
    for (size_t h{3}; h <= 16; ++h)
        for (size_t w{3}; w <= 16; ++w)
            assert(verify_half_conv2d(&launch_wmma_precomp_k_offsets_conv2d_3x3, 1, 1, h, w));
    assert(verify_half_conv2d(&launch_wmma_precomp_k_offsets_conv2d_3x3, C_in, C_out, 32, 32));
    for (size_t h{3}; h <= 16; ++h)
        for (size_t w{3}; w <= 16; ++w)
            assert(verify_half_conv2d(&launch_wmma_int_addr_ns_conv2d_3x3, 1, 1, h, w));
    assert(verify_half_conv2d(&launch_wmma_int_addr_ns_conv2d_3x3, C_in, C_out, 32, 32));
    std::cout << "Unit tests passed." << std::endl;

    // Profiling CUTLASS convolution for reference.
    float const latency_cutlass{profile_cutlass_conv2d_implementation(C_in, C_out, H, W)};
    float const tflops_cutlass{calculate_tflops(C_in, C_out, H, W, latency_cutlass)};
    print_latency("CUTLASS 3x3 Conv2D", latency_cutlass, tflops_cutlass);

    // Profiling latest kernel.
    float const latency_15{profile_half_conv2d(&launch_wmma_half_padded_ldg_conv2d_3x3, C_in, C_out, H, W)};
    float const tflops_15{calculate_tflops(C_in, C_out, H, W, latency_15)};
    print_latency("15. WMMA Half Padded+LDG", latency_15, tflops_15);

    float const latency_26{profile_half_conv2d(&launch_wmma_parallel_store_conv2d_3x3, C_in, C_out, H, W)};
    float const tflops_26{calculate_tflops(C_in, C_out, H, W, latency_26)};
    print_latency("26. Parallel Warp Store", latency_26, tflops_26);

    float const latency_27{profile_half_conv2d(&launch_wmma_coalesced_B_conv2d_3x3, C_in, C_out, H, W)};
    float const tflops_27{calculate_tflops(C_in, C_out, H, W, latency_27)};
    print_latency("27. Coalesced B Load", latency_27, tflops_27);

    float const latency_29{profile_half_conv2d(&launch_wmma_precomp_k_offsets_conv2d_3x3, C_in, C_out, H, W)};
    float const tflops_29{calculate_tflops(C_in, C_out, H, W, latency_29)};
    print_latency("29. Precomp K Offsets", latency_29, tflops_29);

    float const latency_latest{profile_half_conv2d(&launch_wmma_int_addr_ns_conv2d_3x3, C_in, C_out, H, W)};
    float const tflops_latest{calculate_tflops(C_in, C_out, H, W, latency_latest)};
    print_latency("33. Int Addr NoSpill", latency_latest, tflops_latest);

    return 0;
}
