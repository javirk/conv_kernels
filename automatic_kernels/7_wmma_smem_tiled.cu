#pragma once

#include <mma.h>
using namespace nvcuda;

// WMMA implicit GEMM with proper shared memory tiling.
// Block-level shared memory tiles for A (im2col) and B (filter),
// cooperatively loaded by all threads. Each warp computes its own
// output sub-tile using WMMA.

// Tile sizes
constexpr int BM7 = 64;   // block tile M (output spatial positions)
constexpr int BN7 = 64;   // block tile N (output channels)
constexpr int BK7 = 16;   // block tile K (inner dim chunk)
constexpr int WM7 = 16;   // WMMA tile M
constexpr int WN7 = 16;   // WMMA tile N
constexpr int WK7 = 8;    // WMMA tile K

__device__ __forceinline__ float im2col_load7(
    float const* input, int n_idx, int k_idx,
    int out_H, int out_W, int C_in, int H, int W)
{
    int const out_row = n_idx / out_W;
    int const out_col = n_idx % out_W;
    int const ic = k_idx / 9;
    int const rem = k_idx % 9;
    int const fy = rem / 3;
    int const fx = rem % 3;
    return input[ic * H * W + (out_row + fy) * W + (out_col + fx)];
}

__global__ void wmma_smem_tiled_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;
    int const K_gemm = C_in * 9;

    int const block_m = blockIdx.x * BM7;
    int const block_n = blockIdx.y * BN7;

    // Shared memory for A[BM7][BK7] and B[BK7][BN7]
    __shared__ float As[BM7][BK7];
    __shared__ float Bs[BK7][BN7];

    // Warp arrangement: 4x4 warps covering BM7/WM7 x BN7/WN7
    int const warp_id = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const warps_per_row = BN7 / WN7;  // 4
    int const warp_m = warp_id / warps_per_row;
    int const warp_n = warp_id % warps_per_row;

    // Accumulator
    wmma::fragment<wmma::accumulator, WM7, WN7, WK7, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const total_threads = blockDim.x;
    int const tid = threadIdx.x;

    for (int k = 0; k < K_gemm; k += BK7)
    {
        // Cooperatively load A[BM7][BK7] = im2col tile
        for (int idx = tid; idx < BM7 * BK7; idx += total_threads)
        {
            int const m = idx / BK7;
            int const kk = idx % BK7;
            int const gm = block_m + m;
            int const gk = k + kk;
            As[m][kk] = (gm < N && gk < K_gemm) ?
                im2col_load7(input, gm, gk, out_H, out_W, C_in, H, W) : 0.0f;
        }

        // Cooperatively load B[BK7][BN7] = filter tile
        // filter layout: [C_out, C_in*9] so B[kk][n] = filter[n * K_gemm + (k+kk)]
        for (int idx = tid; idx < BK7 * BN7; idx += total_threads)
        {
            int const kk = idx / BN7;
            int const n = idx % BN7;
            int const gk = k + kk;
            int const gn = block_n + n;
            Bs[kk][n] = (gk < K_gemm && gn < C_out) ?
                filter[gn * K_gemm + gk] : 0.0f;
        }

        __syncthreads();

        // Each warp does WMMA on its sub-tiles, iterating over BK7 in chunks of WK7
        int const tile_m_offset = warp_m * WM7;
        int const tile_n_offset = warp_n * WN7;

        for (int kk = 0; kk < BK7; kk += WK7)
        {
            wmma::fragment<wmma::matrix_a, WM7, WN7, WK7,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM7, WN7, WK7,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            // Load from shared memory
            // A sub-tile: As[tile_m_offset..+WM7][kk..+WK7], stride = BK7
            wmma::load_matrix_sync(a_frag, &As[tile_m_offset][kk], BK7);
            // B sub-tile: Bs[kk..+WK7][tile_n_offset..+WN7], stride = BN7
            wmma::load_matrix_sync(b_frag, &Bs[kk][tile_n_offset], BN7);

            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    // Store results: output[oc][spatial] = output[oc * N + spatial_idx]
    __shared__ float Cs[BM7][BN7];
    int const tile_m_offset = warp_m * WM7;
    int const tile_n_offset = warp_n * WN7;
    wmma::store_matrix_sync(&Cs[tile_m_offset][tile_n_offset], acc, BN7, wmma::mem_row_major);

    __syncthreads();

    // Write from shared to global cooperatively
    for (int idx = tid; idx < BM7 * BN7; idx += total_threads)
    {
        int const m = idx / BN7;
        int const n = idx % BN7;
        int const gm = block_m + m;
        int const gn = block_n + n;
        if (gm < N && gn < C_out)
        {
            output[gn * N + gm] = Cs[m][n];
        }
    }
}

template <typename T>
void launch_wmma_smem_tiled_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                         size_t C_in, size_t C_out, size_t H, size_t W,
                                         cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;

    // 16 warps = 512 threads
    constexpr int num_warps = (BM7 / WM7) * (BN7 / WN7);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (N + BM7 - 1) / BM7,
        (C_out + BN7 - 1) / BN7);

    wmma_smem_tiled_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
