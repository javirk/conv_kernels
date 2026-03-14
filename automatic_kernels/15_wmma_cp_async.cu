#pragma once

#include <mma.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

// WMMA with cp.async for truly asynchronous global-to-shared memory copies.
// Based on experiment 10 (best so far: 11.60 ms) with cp.async replacing manual loads.

constexpr int BM15_H = 8;
constexpr int BM15_W = 8;
constexpr int BM15 = BM15_H * BM15_W;  // 64
constexpr int BN15 = 64;
constexpr int BK15 = 16;
constexpr int WM15 = 16;
constexpr int WN15 = 16;
constexpr int WK15 = 8;

__global__ void wmma_cp_async_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const K_gemm = C_in * 9;

    int const tile_row = blockIdx.x * BM15_H;
    int const tile_col = blockIdx.y * BM15_W;
    int const block_n = blockIdx.z * BN15;

    __shared__ float As[2][BM15][BK15];
    __shared__ float Bs[2][BK15][BN15];

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN15 / WN15;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM15, WN15, WK15, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    // Helper to issue cp.async loads for a tile
    auto load_tile_async = [&](int buf, int k_start) {
        // Load A: im2col
        for (int idx = tid; idx < BM15 * BK15; idx += total_threads)
        {
            int const m = idx / BK15;
            int const kk = idx % BK15;
            int const local_row = m / BM15_W;
            int const local_col = m % BM15_W;
            int const out_row = tile_row + local_row;
            int const out_col = tile_col + local_col;
            int const gk = k_start + kk;

            if (out_row < out_H && out_col < out_W && gk < K_gemm)
            {
                int const ic = gk / 9;
                int const rem = gk % 9;
                int const fy = rem / 3;
                int const fx = rem % 3;
                float const* src = &input[ic * H * W + (out_row + fy) * W + (out_col + fx)];
                __pipeline_memcpy_async(&As[buf][m][kk], src, sizeof(float));
            }
            else
            {
                As[buf][m][kk] = 0.0f;
            }
        }
        // Load B: filter
        for (int idx = tid; idx < BK15 * BN15; idx += total_threads)
        {
            int const kk = idx / BN15;
            int const n = idx % BN15;
            int const gk = k_start + kk;
            int const gn = block_n + n;

            if (gk < K_gemm && gn < C_out)
            {
                float const* src = &filter[gn * K_gemm + gk];
                __pipeline_memcpy_async(&Bs[buf][kk][n], src, sizeof(float));
            }
            else
            {
                Bs[buf][kk][n] = 0.0f;
            }
        }
        __pipeline_commit();
    };

    // Load first tile
    load_tile_async(0, 0);
    __pipeline_wait_prior(0);
    __syncthreads();

    int const num_k_iters = (K_gemm + BK15 - 1) / BK15;

    for (int ki = 0; ki < num_k_iters; ++ki)
    {
        int const cur_buf = ki % 2;
        int const next_buf = 1 - cur_buf;

        // Issue async load for next tile
        if (ki + 1 < num_k_iters)
        {
            load_tile_async(next_buf, (ki + 1) * BK15);
        }

        // Compute from current buffer
        int const tile_m_off = warp_m * WM15;
        int const tile_n_off = warp_n * WN15;

        #pragma unroll
        for (int kk = 0; kk < BK15; kk += WK15)
        {
            wmma::fragment<wmma::matrix_a, WM15, WN15, WK15,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM15, WN15, WK15,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As[cur_buf][tile_m_off][kk], BK15);
            wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][tile_n_off], BN15);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // Wait for next tile to be ready
        if (ki + 1 < num_k_iters)
        {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    // Store
    __shared__ float Cs[BM15][BN15];
    wmma::store_matrix_sync(&Cs[warp_m * WM15][warp_n * WN15], acc, BN15, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM15 * BN15; idx += total_threads)
    {
        int const m = idx / BN15;
        int const n = idx % BN15;
        int const local_row = m / BM15_W;
        int const local_col = m % BM15_W;
        int const out_row = tile_row + local_row;
        int const out_col = tile_col + local_col;
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[oc * out_H * out_W + out_row * out_W + out_col] = Cs[m][n];
        }
    }
}

template <typename T>
void launch_wmma_cp_async_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                       size_t C_in, size_t C_out, size_t H, size_t W,
                                       cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    constexpr int num_warps = (BM15 / WM15) * (BN15 / WN15);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (out_H + BM15_H - 1) / BM15_H,
        (out_W + BM15_W - 1) / BM15_W,
        (C_out + BN15 - 1) / BN15);

    wmma_cp_async_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
