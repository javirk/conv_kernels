#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 7: Double buffering to overlap compute with memory loads
// Two smem buffers: while computing from buffer[i], load into buffer[1-i]
// 128x128 tile, BK=32, 8 warps
// Fused filter loop (fy,fx outer)
// smem: 2 * (A[128][32] + B[32][128]) = 2 * 16KB = 32KB

static constexpr int TM7 = 128;
static constexpr int TN7 = 128;
static constexpr int BK7 = 32;
static constexpr int NWARPS7 = 8;

template <typename T>
__global__ void wmma_double_buffer_conv2d_3x3_kernel(
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

    size_t const block_m = blockIdx.x * TM7;
    size_t const block_n = blockIdx.y * TN7;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;  // 0..3
    int const warp_col = warp_id % 2;  // 0..1

    // Double buffer
    __shared__ half smem_A[2][TM7][BK7];    // 2 * 8KB = 16KB
    __shared__ half smem_B[2][BK7][TN7];    // 2 * 8KB = 16KB
    // Total: 32KB

    // Each warp: 32x64 (2x4 WMMA tiles)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int const nthreads = NWARPS7 * 32;
    size_t const total_k_iters = (K + BK7 - 1) / BK7;

    // Load first tile into buffer 0
    {
        size_t k_offset = 0;
        for (int idx = threadIdx.x; idx < TM7 * BK7; idx += nthreads) {
            int const tm = idx / BK7;
            int const tk = idx % BK7;
            size_t const m = block_m + tm;
            size_t const k = k_offset + tk;

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
            smem_A[0][tm][tk] = val;
        }
        for (int idx = threadIdx.x; idx < BK7 * TN7; idx += nthreads) {
            int const tk = idx / TN7;
            int const tn = idx % TN7;
            size_t const k = k_offset + tk;
            size_t const n = block_n + tn;

            half val = __float2half(0.0f);
            if (k < K && n < N) {
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                size_t const fy = fpos / 3;
                size_t const fx = fpos % 3;
                val = __float2half(filter[n * C_in * 9 + ic * 9 + fy * 3 + fx]);
            }
            smem_B[0][tk][tn] = val;
        }
    }
    __syncthreads();

    int buf = 0;

    for (size_t ki = 0; ki < total_k_iters; ki++) {
        int const next_buf = 1 - buf;
        size_t const next_k_offset = (ki + 1) * BK7;

        // Prefetch next tile into next_buf (if not last iteration)
        if (ki + 1 < total_k_iters) {
            for (int idx = threadIdx.x; idx < TM7 * BK7; idx += nthreads) {
                int const tm = idx / BK7;
                int const tk = idx % BK7;
                size_t const m = block_m + tm;
                size_t const k = next_k_offset + tk;

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
                smem_A[next_buf][tm][tk] = val;
            }
            for (int idx = threadIdx.x; idx < BK7 * TN7; idx += nthreads) {
                int const tk = idx / TN7;
                int const tn = idx % TN7;
                size_t const k = next_k_offset + tk;
                size_t const n = block_n + tn;

                half val = __float2half(0.0f);
                if (k < K && n < N) {
                    size_t const ic = k / 9;
                    size_t const fpos = k % 9;
                    size_t const fy = fpos / 3;
                    size_t const fx = fpos % 3;
                    val = __float2half(filter[n * C_in * 9 + ic * 9 + fy * 3 + fx]);
                }
                smem_B[next_buf][tk][tn] = val;
            }
        }

        // Compute from current buffer
        int const wm = warp_row * 32;
        int const wn = warp_col * 64;

        for (int kk = 0; kk < BK7; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[buf][wm + mi * 16][kk], BK7);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[buf][kk][wn + ni * 16], TN7);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        __syncthreads();
        buf = next_buf;
    }

    // Store results (sequential per-warp)
    __shared__ float smem_store[16][16];
    int const wm = warp_row * 32;
    int const wn = warp_col * 64;

    for (int w = 0; w < NWARPS7; w++) {
        if (warp_id == w) {
            for (int mi = 0; mi < 2; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    wmma::store_matrix_sync(&smem_store[0][0], c_frag[mi][ni], 16, wmma::mem_row_major);
                    __syncwarp();
                    int const lane = threadIdx.x % 32;
                    for (int idx = lane; idx < 256; idx += 32) {
                        int const r = idx / 16;
                        int const c = idx % 16;
                        size_t const gm = block_m + wm + mi * 16 + r;
                        size_t const gn = block_n + wn + ni * 16 + c;
                        if (gm < M && gn < N) {
                            size_t const oh = gm / out_W;
                            size_t const ow = gm % out_W;
                            output[gn * out_H * out_W + oh * out_W + ow] = static_cast<T>(smem_store[r][c]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
void launch_wmma_double_buffer_conv2d_3x3(
    T* d_output, T const* d_input, T const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM7 - 1) / TM7),
        (unsigned int)((C_out + TN7 - 1) / TN7));
    dim3 const block(NWARPS7 * 32);

    wmma_double_buffer_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
