#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA Implicit GEMM for Conv2D
// Maps conv2d to GEMM: M=out_H*out_W, N=C_out, K=C_in*9
// Tile: 64x64, BK=16, 4 warps (128 threads)
// Each warp computes 32x32 (2x2 WMMA 16x16 tiles)
// smem: A[64][16]=2KB + B[16][64]=2KB = 4KB (fits easily)

static constexpr int TILE_M_1 = 64;
static constexpr int TILE_N_1 = 64;
static constexpr int BK_1 = 16;

template <typename T>
__global__ void wmma_implicit_gemm_nhwc_conv2d_3x3_kernel(
    T* __restrict__ output,
    T const* __restrict__ input,
    T const* __restrict__ filter,
    size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;
    size_t const N = C_out;
    size_t const K = C_in * 9;

    size_t const block_m = blockIdx.x * TILE_M_1;
    size_t const block_n = blockIdx.y * TILE_N_1;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;  // 0..1
    int const warp_col = warp_id % 2;  // 0..1

    __shared__ half smem_A[TILE_M_1][BK_1];     // 2KB
    __shared__ half smem_B[BK_1][TILE_N_1];      // 2KB

    // Each warp: 2x2 WMMA tiles = 32x32
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    for (size_t k_offset = 0; k_offset < K; k_offset += BK_1) {
        // Load A tile: TILE_M x BK = 64*16 = 1024 elements, 128 threads => 8 each
        for (int idx = threadIdx.x; idx < TILE_M_1 * BK_1; idx += blockDim.x) {
            int const tile_m = idx / BK_1;
            int const tile_k = idx % BK_1;
            size_t const m = block_m + tile_m;
            size_t const k = k_offset + tile_k;

            half val = __float2half(0.0f);
            if (m < M && k < K) {
                size_t const oh = m / out_W;
                size_t const ow = m % out_W;
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                size_t const fy = fpos / 3;
                size_t const fx = fpos % 3;
                val = __float2half(input[ic * H * W + (oh + fy) * W + (ow + fx)]);
            }
            smem_A[tile_m][tile_k] = val;
        }

        // Load B tile: BK x TILE_N = 16*64 = 1024 elements
        for (int idx = threadIdx.x; idx < BK_1 * TILE_N_1; idx += blockDim.x) {
            int const tile_k = idx / TILE_N_1;
            int const tile_n = idx % TILE_N_1;
            size_t const k = k_offset + tile_k;
            size_t const n = block_n + tile_n;

            half val = __float2half(0.0f);
            if (k < K && n < N) {
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                size_t const fy = fpos / 3;
                size_t const fx = fpos % 3;
                val = __float2half(filter[n * C_in * 9 + ic * 9 + fy * 3 + fx]);
            }
            smem_B[tile_k][tile_n] = val;
        }

        __syncthreads();

        int const wm = warp_row * 32;
        int const wn = warp_col * 32;

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(a_frag[mi], &smem_A[wm + mi * 16][0], BK_1);
        #pragma unroll
        for (int ni = 0; ni < 2; ni++)
            wmma::load_matrix_sync(b_frag[ni], &smem_B[0][wn + ni * 16], TILE_N_1);

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);

        __syncthreads();
    }

    // Store results: each warp stores its 2x2 tiles through smem (reuse smem_A area)
    // We need 32x32 floats = 4KB per warp. Do it in 16x32 chunks (2KB each).
    __shared__ float smem_store[32][32];  // 4KB - reused by warps sequentially

    int const wm = warp_row * 32;
    int const wn = warp_col * 32;

    for (int w = 0; w < 4; w++) {
        if (warp_id == w) {
            // Store 2x2 fragments
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    wmma::store_matrix_sync(
                        &smem_store[mi * 16][ni * 16],
                        c_frag[mi][ni], 32, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        if (warp_id == w) {
            int const w_row = w / 2;
            int const w_col = w % 2;
            int const base_m = w_row * 32;
            int const base_n = w_col * 32;
            int const lane = threadIdx.x % 32;
            // Each lane writes a row (32 elements)
            for (int r = lane; r < 32; r += 32) {
                size_t const m = block_m + base_m + r;
                if (m < M) {
                    size_t const oh = m / out_W;
                    size_t const ow = m % out_W;
                    for (int c = 0; c < 32; c++) {
                        size_t const n = block_n + base_n + c;
                        if (n < N) {
                            output[n * out_H * out_W + oh * out_W + ow] =
                                static_cast<T>(smem_store[r][c]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
void launch_wmma_implicit_gemm_nhwc_conv2d_3x3(
    T* d_output, T const* d_input, T const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TILE_M_1 - 1) / TILE_M_1),
        (unsigned int)((C_out + TILE_N_1 - 1) / TILE_N_1));
    dim3 const block(128);  // 4 warps

    wmma_implicit_gemm_nhwc_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
