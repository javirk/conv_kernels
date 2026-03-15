#pragma once

#include <mma.h>
using namespace nvcuda;

// WMMA with 2D spatial grid to avoid expensive divmod in im2col.
// Instead of linearizing spatial dims, use 3D grid: (tile_row, tile_col, oc_tile)
// and compute im2col from known (row, col) directly.

constexpr int BM10_H = 8;   // spatial rows per block
constexpr int BM10_W = 8;   // spatial cols per block
constexpr int BM10 = BM10_H * BM10_W;  // 64 spatial positions per block
constexpr int BN10 = 64;    // output channels per block
constexpr int BK10 = 16;    // K-dim chunk

constexpr int WM10 = 16;
constexpr int WN10 = 16;
constexpr int WK10 = 8;

__global__ void wmma_2d_spatial_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;
    int const K_gemm = C_in * 9;

    // 2D spatial tiling
    int const tile_row_start = blockIdx.x * BM10_H;
    int const tile_col_start = blockIdx.y * BM10_W;
    int const block_n = blockIdx.z * BN10;

    // Double buffers
    __shared__ float As[2][BM10][BK10];
    __shared__ float Bs[2][BK10][BN10];

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN10 / WN10;  // 4
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM10, WN10, WK10, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    auto load_tile = [&](int buf, int k_start) {
        // Load A: im2col without divmod
        for (int idx = tid; idx < BM10 * BK10; idx += total_threads)
        {
            int const m = idx / BK10;
            int const kk = idx % BK10;
            int const local_row = m / BM10_W;
            int const local_col = m % BM10_W;
            int const out_row = tile_row_start + local_row;
            int const out_col = tile_col_start + local_col;
            int const gk = k_start + kk;

            if (out_row < out_H && out_col < out_W && gk < K_gemm)
            {
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

        // Load B: filter tile
        for (int idx = tid; idx < BK10 * BN10; idx += total_threads)
        {
            int const kk = idx / BN10;
            int const n = idx % BN10;
            int const gk = k_start + kk;
            int const gn = block_n + n;
            Bs[buf][kk][n] = (gk < K_gemm && gn < C_out) ?
                filter[gn * K_gemm + gk] : 0.0f;
        }
    };

    // Load first tile
    load_tile(0, 0);
    __syncthreads();

    int const num_k_iters = (K_gemm + BK10 - 1) / BK10;

    for (int ki = 0; ki < num_k_iters; ++ki)
    {
        int const cur_buf = ki % 2;
        int const next_buf = 1 - cur_buf;

        if (ki + 1 < num_k_iters)
        {
            load_tile(next_buf, (ki + 1) * BK10);
        }

        int const tile_m_off = warp_m * WM10;
        int const tile_n_off = warp_n * WN10;

        #pragma unroll
        for (int kk = 0; kk < BK10; kk += WK10)
        {
            wmma::fragment<wmma::matrix_a, WM10, WN10, WK10,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM10, WN10, WK10,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As[cur_buf][tile_m_off][kk], BK10);
            wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][tile_n_off], BN10);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    // Store results
    __shared__ float Cs[BM10][BN10];
    wmma::store_matrix_sync(&Cs[warp_m * WM10][warp_n * WN10], acc, BN10, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM10 * BN10; idx += total_threads)
    {
        int const m = idx / BN10;
        int const n = idx % BN10;
        int const local_row = m / BM10_W;
        int const local_col = m % BM10_W;
        int const out_row = tile_row_start + local_row;
        int const out_col = tile_col_start + local_col;
        int const gn = block_n + n;
        if (out_row < out_H && out_col < out_W && gn < C_out)
        {
            output[gn * out_H * out_W + out_row * out_W + out_col] = Cs[m][n];
        }
    }
}

template <typename T>
void launch_wmma_2d_spatial_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                         size_t C_in, size_t C_out, size_t H, size_t W,
                                         cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    constexpr int num_warps = (BM10 / WM10) * (BN10 / WN10);  // 4*4=16
    dim3 const block(num_warps * 32);  // 512 threads
    dim3 const grid(
        (out_H + BM10_H - 1) / BM10_H,
        (out_W + BM10_W - 1) / BM10_W,
        (C_out + BN10 - 1) / BN10);

    wmma_2d_spatial_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
