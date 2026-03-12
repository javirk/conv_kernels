#pragma once

#include <mma.h>
using namespace nvcuda;

// Double-buffered WMMA: overlap global memory loads with tensor core compute.
// Uses two sets of SMEM buffers, loading next tile while computing current.

constexpr int BM9 = 64;
constexpr int BN9 = 64;
constexpr int BK9 = 16;
constexpr int WM9 = 16;
constexpr int WN9 = 16;
constexpr int WK9 = 8;

__device__ __forceinline__ float im2col_load9(
    float const* input, int n_idx, int k_idx,
    int out_W, int C_in, int H, int W)
{
    int const out_row = n_idx / out_W;
    int const out_col = n_idx % out_W;
    int const ic = k_idx / 9;
    int const rem = k_idx % 9;
    int const fy = rem / 3;
    int const fx = rem % 3;
    return input[ic * H * W + (out_row + fy) * W + (out_col + fx)];
}

__global__ void wmma_double_buffer_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;
    int const K_gemm = C_in * 9;

    int const block_m = blockIdx.x * BM9;
    int const block_n = blockIdx.y * BN9;

    // Double buffers: 2 x (As[BM9][BK9] + Bs[BK9][BN9])
    __shared__ float As[2][BM9][BK9];
    __shared__ float Bs[2][BK9][BN9];

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN9 / WN9;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM9, WN9, WK9, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    // Load first tile into buffer 0
    for (int idx = tid; idx < BM9 * BK9; idx += total_threads)
    {
        int const m = idx / BK9;
        int const kk = idx % BK9;
        int const gm = block_m + m;
        As[0][m][kk] = (gm < N && kk < K_gemm) ?
            im2col_load9(input, gm, kk, out_W, C_in, H, W) : 0.0f;
    }
    for (int idx = tid; idx < BK9 * BN9; idx += total_threads)
    {
        int const kk = idx / BN9;
        int const n = idx % BN9;
        int const gn = block_n + n;
        Bs[0][kk][n] = (kk < K_gemm && gn < C_out) ?
            filter[gn * K_gemm + kk] : 0.0f;
    }
    __syncthreads();

    int const num_k_iters = (K_gemm + BK9 - 1) / BK9;

    for (int ki = 0; ki < num_k_iters; ++ki)
    {
        int const cur_buf = ki % 2;
        int const next_buf = 1 - cur_buf;
        int const next_k = (ki + 1) * BK9;

        // Prefetch next tile into next_buf (if there is one)
        if (ki + 1 < num_k_iters)
        {
            for (int idx = tid; idx < BM9 * BK9; idx += total_threads)
            {
                int const m = idx / BK9;
                int const kk = idx % BK9;
                int const gm = block_m + m;
                int const gk = next_k + kk;
                As[next_buf][m][kk] = (gm < N && gk < K_gemm) ?
                    im2col_load9(input, gm, gk, out_W, C_in, H, W) : 0.0f;
            }
            for (int idx = tid; idx < BK9 * BN9; idx += total_threads)
            {
                int const kk = idx / BN9;
                int const n = idx % BN9;
                int const gk = next_k + kk;
                int const gn = block_n + n;
                Bs[next_buf][kk][n] = (gk < K_gemm && gn < C_out) ?
                    filter[gn * K_gemm + gk] : 0.0f;
            }
        }

        // Compute from current buffer
        int const tile_m_off = warp_m * WM9;
        int const tile_n_off = warp_n * WN9;

        #pragma unroll
        for (int kk = 0; kk < BK9; kk += WK9)
        {
            wmma::fragment<wmma::matrix_a, WM9, WN9, WK9,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM9, WN9, WK9,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As[cur_buf][tile_m_off][kk], BK9);
            wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][tile_n_off], BN9);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    // Store results
    __shared__ float Cs[BM9][BN9];
    wmma::store_matrix_sync(&Cs[warp_m * WM9][warp_n * WN9], acc, BN9, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM9 * BN9; idx += total_threads)
    {
        int const m = idx / BN9;
        int const n = idx % BN9;
        int const gm = block_m + m;
        int const gn = block_n + n;
        if (gm < N && gn < C_out)
        {
            output[gn * N + gm] = Cs[m][n];
        }
    }
}

template <typename T>
void launch_wmma_double_buffer_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                            size_t C_in, size_t C_out, size_t H, size_t W,
                                            cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;

    constexpr int num_warps = (BM9 / WM9) * (BN9 / WN9);  // 4*4 = 16 warps = 512 threads
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (N + BM9 - 1) / BM9,
        (C_out + BN9 - 1) / BN9);

    wmma_double_buffer_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
