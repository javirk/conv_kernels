#pragma once

#include <mma.h>
using namespace nvcuda;

// Warp-level tiling: each warp computes a 2x2 grid of WMMA 16x16 tiles.
// With BM=64, BN=64: 2 warps in M, 2 warps in N = 4 warps = 128 threads.
// Each warp computes 32x32 output, giving 4x more compute per warp.
// Much better occupancy (128 threads/block vs 512) and better compute/memory ratio.

constexpr int BM13 = 64;
constexpr int BN13 = 64;
constexpr int BK13 = 16;

constexpr int WM13 = 16;
constexpr int WN13 = 16;
constexpr int WK13 = 8;

// Each warp computes WARP_TILES_M x WARP_TILES_N WMMA tiles
constexpr int WARP_TILES_M13 = 2;
constexpr int WARP_TILES_N13 = 2;

// Warps per dimension
constexpr int WARPS_M13 = BM13 / (WM13 * WARP_TILES_M13);  // 64/32 = 2
constexpr int WARPS_N13 = BN13 / (WN13 * WARP_TILES_N13);  // 64/32 = 2
constexpr int NUM_WARPS13 = WARPS_M13 * WARPS_N13;  // 4

__global__ void wmma_warp_tile_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const K_gemm = C_in * 9;

    int const block_row = blockIdx.x * (BM13 / 8);  // BM13_H = 8
    int const block_col = blockIdx.y * (BM13 / 8);  // BM13_W = 8, so BM13 = 8*8 = 64
    // Actually, let me use linear spatial indexing for simplicity
    int const N = out_H * out_W;
    int const block_m = blockIdx.x * BM13;
    int const block_n = blockIdx.y * BN13;

    __shared__ float As[2][BM13][BK13];   // 2*64*16 = 2048 = 8KB
    __shared__ float Bs[2][BK13][BN13];   // 2*16*64 = 2048 = 8KB
    // Total: 16KB for AB, 16KB for Cs = 32KB < 48KB

    int const warp_id = threadIdx.x / 32;
    int const warp_m = warp_id / WARPS_N13;
    int const warp_n = warp_id % WARPS_N13;

    // 2x2 accumulators per warp
    wmma::fragment<wmma::accumulator, WM13, WN13, WK13, float> acc[WARP_TILES_M13][WARP_TILES_N13];
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M13; ++wm)
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N13; ++wn)
            wmma::fill_fragment(acc[wm][wn], 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;  // 128

    auto load_tile = [&](int buf, int k_start) {
        // A: im2col
        for (int idx = tid; idx < BM13 * BK13; idx += total_threads)
        {
            int const m = idx / BK13;
            int const kk = idx % BK13;
            int const gm = block_m + m;
            int const gk = k_start + kk;
            if (gm < N && gk < K_gemm)
            {
                int const out_row = gm / out_W;
                int const out_col = gm % out_W;
                int const ic = gk / 9;
                int const rem = gk % 9;
                int const fy = rem / 3;
                int const fx = rem % 3;
                As[buf][m][kk] = input[ic * H * W + (out_row + fy) * W + (out_col + fx)];
            }
            else
            {
                As[buf][m][kk] = 0.0f;
            }
        }
        // B: filter
        for (int idx = tid; idx < BK13 * BN13; idx += total_threads)
        {
            int const kk = idx / BN13;
            int const n = idx % BN13;
            int const gk = k_start + kk;
            int const gn = block_n + n;
            Bs[buf][kk][n] = (gk < K_gemm && gn < C_out) ?
                filter[gn * K_gemm + gk] : 0.0f;
        }
    };

    load_tile(0, 0);
    __syncthreads();

    int const num_k_iters = (K_gemm + BK13 - 1) / BK13;

    for (int ki = 0; ki < num_k_iters; ++ki)
    {
        int const cur_buf = ki % 2;
        int const next_buf = 1 - cur_buf;

        if (ki + 1 < num_k_iters)
            load_tile(next_buf, (ki + 1) * BK13);

        // Each warp computes 2x2 WMMA tiles
        #pragma unroll
        for (int kk = 0; kk < BK13; kk += WK13)
        {
            // Load A fragments for this warp's M tiles
            wmma::fragment<wmma::matrix_a, WM13, WN13, WK13,
                           wmma::precision::tf32, wmma::row_major> a_frags[WARP_TILES_M13];
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M13; ++wm)
            {
                int const m_off = (warp_m * WARP_TILES_M13 + wm) * WM13;
                wmma::load_matrix_sync(a_frags[wm], &As[cur_buf][m_off][kk], BK13);
            }

            // Load B fragments for this warp's N tiles
            wmma::fragment<wmma::matrix_b, WM13, WN13, WK13,
                           wmma::precision::tf32, wmma::row_major> b_frags[WARP_TILES_N13];
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N13; ++wn)
            {
                int const n_off = (warp_n * WARP_TILES_N13 + wn) * WN13;
                wmma::load_matrix_sync(b_frags[wn], &Bs[cur_buf][kk][n_off], BN13);
            }

            // MMA
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M13; ++wm)
                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N13; ++wn)
                    wmma::mma_sync(acc[wm][wn], a_frags[wm], b_frags[wn], acc[wm][wn]);
        }

        __syncthreads();
    }

    // Store results
    __shared__ float Cs[BM13][BN13];  // 64*64 = 16KB
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M13; ++wm)
    {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N13; ++wn)
        {
            int const m_off = (warp_m * WARP_TILES_M13 + wm) * WM13;
            int const n_off = (warp_n * WARP_TILES_N13 + wn) * WN13;
            wmma::store_matrix_sync(&Cs[m_off][n_off], acc[wm][wn], BN13, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < BM13 * BN13; idx += total_threads)
    {
        int const m = idx / BN13;
        int const n = idx % BN13;
        int const gm = block_m + m;
        int const gn = block_n + n;
        if (gm < N && gn < C_out)
        {
            output[gn * N + gm] = Cs[m][n];
        }
    }
}

template <typename T>
void launch_wmma_warp_tile_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                        size_t C_in, size_t C_out, size_t H, size_t W,
                                        cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;

    dim3 const block(NUM_WARPS13 * 32);  // 128 threads
    dim3 const grid(
        (N + BM13 - 1) / BM13,
        (C_out + BN13 - 1) / BN13);

    wmma_warp_tile_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
