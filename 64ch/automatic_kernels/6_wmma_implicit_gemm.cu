#pragma once

#include <mma.h>
using namespace nvcuda;

// Implicit GEMM conv2d using WMMA tensor core operations.
// Reshape conv2d as: output[N, K] = im2col[N, C_in*9] x filter[C_in*9, K]
// where N = out_H * out_W, K = C_out.
// Use WMMA 16x16x8 tiles with TF32 precision for Ampere.

// im2col helper: given a linear output index and a column in the im2col matrix,
// return the corresponding input value.
__device__ __forceinline__ float im2col_load(
    float const* input, int n_idx, int k_idx,
    int out_H, int out_W, int C_in, int H, int W)
{
    // n_idx is the output spatial index: row * out_W + col
    // k_idx is ic * 9 + fy * 3 + fx
    int const out_row = n_idx / out_W;
    int const out_col = n_idx % out_W;
    int const ic = k_idx / 9;
    int const rem = k_idx % 9;
    int const fy = rem / 3;
    int const fx = rem % 3;
    int const in_row = out_row + fy;
    int const in_col = out_col + fx;
    return input[ic * H * W + in_row * W + in_col];
}

// WMMA tile dimensions for TF32 on Ampere
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// Each warp computes a WMMA_M x WMMA_N tile of the output.
// Block processes BLOCK_M x BLOCK_N output tile.
constexpr int BLOCK_M = 64;  // output spatial positions per block
constexpr int BLOCK_N = 64;  // output channels per block
constexpr int WARPS_M = BLOCK_M / WMMA_M;  // 4
constexpr int WARPS_N = BLOCK_N / WMMA_N;  // 4

__global__ void wmma_conv2d_3x3_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;  // total output spatial positions
    int const K_gemm = C_in * 9;  // im2col inner dimension

    // Block tile origin in the output matrix
    int const block_m = blockIdx.x * BLOCK_M;
    int const block_n = blockIdx.y * BLOCK_N;

    // Warp index within the block
    int const warp_id = threadIdx.x / 32;
    int const warp_m = warp_id / WARPS_N;  // which M-tile this warp handles
    int const warp_n = warp_id % WARPS_N;  // which N-tile this warp handles

    // The output tile this warp computes
    int const tile_m = block_m + warp_m * WMMA_M;
    int const tile_n = block_n + warp_n * WMMA_N;

    // Accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Loop over K dimension in chunks of WMMA_K
    for (int k = 0; k < K_gemm; k += WMMA_K)
    {
        // Load A fragment (im2col): WMMA_M x WMMA_K
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> a_frag;

        // Load B fragment (filter reshaped): WMMA_K x WMMA_N
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> b_frag;

        // Fill A from im2col (on the fly)
        for (int i = 0; i < a_frag.num_elements; ++i)
        {
            // Map fragment element to matrix coordinates
            // For row_major A[WMMA_M, WMMA_K]: element i maps to (i / WMMA_K, i % WMMA_K)
            // But fragment layout is opaque. We need to use load_matrix_sync with a temp buffer.
            // Let's use a register buffer approach instead.
        }

        // Since fragment layout is opaque, we need to stage through shared memory
        // Use a small staging buffer in shared memory
        __shared__ float a_stage[WARPS_M * WARPS_N][WMMA_M * WMMA_K];
        __shared__ float b_stage[WARPS_M * WARPS_N][WMMA_K * WMMA_N];

        // Each warp fills its own staging area
        int const lane = threadIdx.x % 32;

        // Fill A stage: im2col values
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += 32)
        {
            int const m = idx / WMMA_K;
            int const kk = idx % WMMA_K;
            int const global_m = tile_m + m;
            int const global_k = k + kk;
            if (global_m < N && global_k < K_gemm)
                a_stage[warp_id][idx] = im2col_load(input, global_m, global_k,
                                                     out_H, out_W, C_in, H, W);
            else
                a_stage[warp_id][idx] = 0.0f;
        }

        // Fill B stage: filter[K_gemm, C_out] column-major = filter[oc * K_gemm + k]
        // But our filter is [C_out, C_in, 3, 3] = [C_out, K_gemm]
        // So B[k, n] = filter[n * K_gemm + k] — but we need row_major B[WMMA_K, WMMA_N]
        for (int idx = lane; idx < WMMA_K * WMMA_N; idx += 32)
        {
            int const kk = idx / WMMA_N;
            int const n = idx % WMMA_N;
            int const global_k = k + kk;
            int const global_n = tile_n + n;
            if (global_k < K_gemm && global_n < C_out)
                b_stage[warp_id][idx] = filter[global_n * K_gemm + global_k];
            else
                b_stage[warp_id][idx] = 0.0f;
        }

        __syncwarp();

        // Load fragments from staging
        wmma::load_matrix_sync(a_frag, a_stage[warp_id], WMMA_K);
        wmma::load_matrix_sync(b_frag, b_stage[warp_id], WMMA_N);

        // MMA
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Store accumulator to output
    // Output layout: [C_out, out_H, out_W] = [C_out, N]
    // acc represents output[tile_m..tile_m+WMMA_M, tile_n..tile_n+WMMA_N] in row-major
    // But we need to store as output[oc * N + spatial_idx]
    // Store to shared memory first, then write to global
    __shared__ float c_stage[WARPS_M * WARPS_N][WMMA_M * WMMA_N];
    wmma::store_matrix_sync(c_stage[warp_id], acc, WMMA_N, wmma::mem_row_major);

    __syncwarp();

    int const lane = threadIdx.x % 32;
    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32)
    {
        int const m = idx / WMMA_N;
        int const n = idx % WMMA_N;
        int const global_m = tile_m + m;
        int const global_n = tile_n + n;
        if (global_m < N && global_n < C_out)
        {
            output[global_n * N + global_m] = c_stage[warp_id][idx];
        }
    }
}

template <typename T>
void launch_wmma_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                             size_t C_in, size_t C_out, size_t H, size_t W,
                             cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;

    dim3 const grid(
        (N + BLOCK_M - 1) / BLOCK_M,
        (C_out + BLOCK_N - 1) / BLOCK_N);
    dim3 const block(WARPS_M * WARPS_N * 32);  // 16 warps * 32 = 512 threads

    wmma_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
