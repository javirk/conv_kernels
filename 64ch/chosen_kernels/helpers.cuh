#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr size_t div_up(size_t a, size_t b) { return (a + b - 1) / b; }

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error at %s:%d — %s %s\n", file, line,
                cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
inline void check_cuda_last(char const* file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error at %s:%d — %s\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ---- Layout conversion helpers ----

__global__ void chw_to_hwc_kernel(float* __restrict__ dst, float const* __restrict__ src,
                                   int C, int H, int W)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * H * W) return;
    int const c = idx / (H * W);
    int const hw = idx % (H * W);
    dst[hw * C + c] = src[idx];
}

__global__ void hwc_to_chw_kernel(float* __restrict__ dst, float const* __restrict__ src,
                                   int C, int H, int W)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * H * W) return;
    int const c = idx % C;
    int const hw = idx / C;
    dst[c * H * W + hw] = src[idx];
}

__global__ void reorder_filter_nhwc_kernel(float* __restrict__ dst, float const* __restrict__ src,
                                            int C_out, int C_in)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * C_in * 9) return;
    int const oc = idx / (C_in * 9);
    int const rem = idx % (C_in * 9);
    int const ic = rem / 9;
    int const fpos = rem % 9;
    dst[oc * 9 * C_in + fpos * C_in + ic] = src[idx];
}

__global__ void transpose_filter_kernel(
    float* __restrict__ filter_t,     // [3, 3, C_in, C_out]
    float const* __restrict__ filter, // [C_out, 3, 3, C_in]
    int C_in, int C_out)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const total = 9 * C_in * C_out;
    if (idx >= total) return;
    int const oc = idx % C_out;
    int const ic = (idx / C_out) % C_in;
    int const fx = (idx / (C_out * C_in)) % 3;
    int const fy = idx / (C_out * C_in * 3);
    filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc] =
        filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic];
}

__global__ void float_to_half_kernel(__half* __restrict__ out,
                                      float const* __restrict__ in, int n)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}
