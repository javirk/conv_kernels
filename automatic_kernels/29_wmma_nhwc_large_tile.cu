#pragma once

#include <mma.h>
using namespace nvcuda;

// Experiment 29: Larger spatial tile BM=128 (16x8) to amortize filter loads.
//
// Base: exp 28 (vec4 for both As and Bs).
// Change: BM 64→128. Each block covers 16 rows x 8 cols of output.
// This means the same Bs (filter) tile serves 2x more spatial positions.
// 8 warps: 2 along M (warp_m=0..1), 4 along N (warp_n=0..3).
// Each warp_m handles 64 rows → 4 WMMA 16x16 fragments along M.
// SMEM: As[2][128][17]=8.5KB, Bs[2][16][65]=4.1KB ≈ 12.6KB.

constexpr int BM29_H = 16;
constexpr int BM29_W = 8;
constexpr int BM29 = BM29_H * BM29_W;  // 128
constexpr int BN29 = 64;
constexpr int BK29 = 16;
constexpr int WM29 = 16;
constexpr int WN29 = 16;
constexpr int WK29 = 8;

constexpr int BK29_PAD = BK29 + 1;  // 17
constexpr int BN29_PAD = BN29 + 1;  // 65

__device__ __forceinline__ void load_As_tile_29(
    float As[][BK29_PAD], float const* __restrict__ input,
    int tile_row, int tile_col, int fy, int fx, int ic_base,
    int C_in, int H, int W, int tid, int total_threads)
{
    constexpr int vec4_count = (BM29 * BK29) / 4;  // 512
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        int const elem = vi * 4;
        int const m = elem / BK29;
        int const k = elem % BK29;
        int const row = tile_row + (m / BM29_W) + fy;
        int const col = tile_col + (m % BM29_W) + fx;
        int const ic = ic_base + k;
        float4 val;
        if (row < H && col < W && ic + 3 < C_in) {
            val = *reinterpret_cast<float4 const*>(
                &input[row * W * C_in + col * C_in + ic]);
        } else {
            val.x = (row < H && col < W && ic < C_in) ?
                input[row * W * C_in + col * C_in + ic] : 0.0f;
            val.y = (row < H && col < W && ic + 1 < C_in) ?
                input[row * W * C_in + col * C_in + ic + 1] : 0.0f;
            val.z = (row < H && col < W && ic + 2 < C_in) ?
                input[row * W * C_in + col * C_in + ic + 2] : 0.0f;
            val.w = (row < H && col < W && ic + 3 < C_in) ?
                input[row * W * C_in + col * C_in + ic + 3] : 0.0f;
        }
        As[m][k]     = val.x;
        As[m][k + 1] = val.y;
        As[m][k + 2] = val.z;
        As[m][k + 3] = val.w;
    }
}

__device__ __forceinline__ void load_Bs_tile_29(
    float Bs[][BN29_PAD], float const* __restrict__ filter,
    int block_n, int fy, int fx, int ic_base,
    int C_in, int C_out, int tid, int total_threads)
{
    constexpr int vec4_count = (BK29 * BN29) / 4;  // 256
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        int const n = vi / (BK29 / 4);
        int const k4 = vi % (BK29 / 4);
        int const k = k4 * 4;
        int const oc = block_n + n;
        int const ic = ic_base + k;
        float4 val;
        if (ic + 3 < C_in && oc < C_out) {
            val = *reinterpret_cast<float4 const*>(
                &filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic]);
        } else {
            val.x = (ic < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
            val.y = (ic + 1 < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic + 1] : 0.0f;
            val.z = (ic + 2 < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic + 2] : 0.0f;
            val.w = (ic + 3 < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic + 3] : 0.0f;
        }
        Bs[k][n]     = val.x;
        Bs[k + 1][n] = val.y;
        Bs[k + 2][n] = val.z;
        Bs[k + 3][n] = val.w;
    }
}

__launch_bounds__(256, 2)
__global__ void wmma_nhwc_large_tile_conv2d_kernel(
    float* __restrict__ output,
    float const* __restrict__ input,
    float const* __restrict__ filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM29_H;
    int const tile_col = blockIdx.y * BM29_W;
    int const block_n = blockIdx.z * BN29;

    __shared__ float As[2][BM29][BK29_PAD];
    __shared__ float Bs[2][BK29][BN29_PAD];

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = BN29 / WN29;  // 4
    int const warp_m = warp_id / warps_n;  // 0 or 1
    int const warp_n = warp_id % warps_n;  // 0..3

    // 4 accumulator fragments per warp along M (64 rows / 16 = 4)
    wmma::fragment<wmma::accumulator, WM29, WN29, WK29, float> acc0, acc1, acc2, acc3;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);
    wmma::fill_fragment(acc3, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;
    int const num_ic_iters = (C_in + BK29 - 1) / BK29;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            load_As_tile_29(As[0], input, tile_row, tile_col, fy, fx, 0,
                           C_in, H, W, tid, total_threads);
            load_Bs_tile_29(Bs[0], filter, block_n, fy, fx, 0,
                           C_in, C_out, tid, total_threads);
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK29;
                    load_As_tile_29(As[next_buf], input, tile_row, tile_col,
                                   fy, fx, next_ic, C_in, H, W, tid, total_threads);
                    load_Bs_tile_29(Bs[next_buf], filter, block_n, fy, fx,
                                   next_ic, C_in, C_out, tid, total_threads);
                }

                #pragma unroll
                for (int kk = 0; kk < BK29; kk += WK29)
                {
                    wmma::fragment<wmma::matrix_a, WM29, WN29, WK29,
                                   wmma::precision::tf32, wmma::row_major> a_frag0, a_frag1, a_frag2, a_frag3;
                    wmma::fragment<wmma::matrix_b, WM29, WN29, WK29,
                                   wmma::precision::tf32, wmma::row_major> b_frag;

                    int const m_base = warp_m * 64;
                    wmma::load_matrix_sync(a_frag0, &As[cur_buf][m_base][kk], BK29_PAD);
                    wmma::load_matrix_sync(a_frag1, &As[cur_buf][m_base + 16][kk], BK29_PAD);
                    wmma::load_matrix_sync(a_frag2, &As[cur_buf][m_base + 32][kk], BK29_PAD);
                    wmma::load_matrix_sync(a_frag3, &As[cur_buf][m_base + 48][kk], BK29_PAD);
                    wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][warp_n * WN29], BN29_PAD);
                    wmma::mma_sync(acc0, a_frag0, b_frag, acc0);
                    wmma::mma_sync(acc1, a_frag1, b_frag, acc1);
                    wmma::mma_sync(acc2, a_frag2, b_frag, acc2);
                    wmma::mma_sync(acc3, a_frag3, b_frag, acc3);
                }

                __syncthreads();
            }
        }
    }

    // Store in 2 passes (one per warp_m), reusing first 64*64=4096 floats of SMEM
    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);

    // Pass 1: warp_m=0 stores to Cs, all threads write to global
    if (warp_m == 0)
    {
        wmma::store_matrix_sync(&Cs[0 * BN29 + warp_n * WN29], acc0, BN29, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[16 * BN29 + warp_n * WN29], acc1, BN29, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[32 * BN29 + warp_n * WN29], acc2, BN29, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[48 * BN29 + warp_n * WN29], acc3, BN29, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN29; idx += total_threads)
    {
        int const m = idx / BN29;
        int const n = idx & (BN29 - 1);
        int const out_row = tile_row + (m / BM29_W);
        int const out_col = tile_col + (m % BM29_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN29 + n];
        }
    }
    __syncthreads();

    // Pass 2: warp_m=1 stores to Cs, all threads write to global
    if (warp_m == 1)
    {
        wmma::store_matrix_sync(&Cs[0 * BN29 + warp_n * WN29], acc0, BN29, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[16 * BN29 + warp_n * WN29], acc1, BN29, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[32 * BN29 + warp_n * WN29], acc2, BN29, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[48 * BN29 + warp_n * WN29], acc3, BN29, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN29; idx += total_threads)
    {
        int const m = idx / BN29;
        int const n = idx & (BN29 - 1);
        int const m_global = m + 64;
        int const out_row = tile_row + (m_global / BM29_W);
        int const out_col = tile_col + (m_global % BM29_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN29 + n];
        }
    }
}

float profile_nhwc_large_tile_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats = 100;
    constexpr int num_warmups = 10;
    int const out_H = H - 2;
    int const out_W = W - 2;

    float *d_input_hwc, *d_filter_nhwc, *d_output_hwc;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM29_H - 1) / BM29_H,
        (out_W + BM29_W - 1) / BM29_W,
        (C_out + BN29 - 1) / BN29);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_large_tile_conv2d_kernel<<<grid, block, 0, s>>>(
            d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    };

    std::function<void(cudaStream_t)> fn = kernel_fn;
    float latency = measure_performance(fn, stream, num_repeats, num_warmups);

    CHECK_CUDA_ERROR(cudaFree(d_input_hwc));
    CHECK_CUDA_ERROR(cudaFree(d_filter_nhwc));
    CHECK_CUDA_ERROR(cudaFree(d_output_hwc));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

template <typename T>
void launch_wmma_nhwc_large_tile_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                              size_t C_in, size_t C_out, size_t H, size_t W,
                                              cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    T *d_input_hwc, *d_filter_nhwc, *d_output_hwc;
    cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(T));
    cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(T));
    cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(T));

    int const n1 = C_in * H * W;
    chw_to_hwc_kernel<<<(n1+255)/256, 256, 0, stream>>>(d_input_hwc, d_input, C_in, H, W);
    int const n2 = C_out * C_in * 9;
    reorder_filter_nhwc_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_nhwc, d_filter, C_out, C_in);

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM29_H - 1) / BM29_H,
        (out_W + BM29_W - 1) / BM29_W,
        (C_out + BN29 - 1) / BN29);
    wmma_nhwc_large_tile_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_output_hwc);
}
