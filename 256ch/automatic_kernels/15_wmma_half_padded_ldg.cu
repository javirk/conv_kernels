#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 15: Smem padding + __ldg for read-only cache
// Padding smem_A columns to avoid bank conflicts
// __ldg uses the texture cache path for read-only data
// Based on exp 12 (best so far): 128x128, BK=32, 8 warps, reg precomp

static constexpr int TM15 = 128;
static constexpr int TN15 = 128;
static constexpr int BK15 = 32;
static constexpr int PAD15 = 8;   // pad to shift bank alignment
static constexpr int NWARPS15 = 8;

__global__ void wmma_half_padded_ldg_conv2d_3x3_kernel(
    half* __restrict__ output,
    half const* __restrict__ input,
    half const* __restrict__ filter,
    size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;
    size_t const N = C_out;
    size_t const K = C_in * 9;

    size_t const block_m = blockIdx.x * TM15;
    size_t const block_n = blockIdx.y * TN15;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS15 * 32;

    // Register precomputed addresses
    int const my_tm = threadIdx.x % TM15;
    size_t const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (int)(my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (int)(my_m % out_W) : -1;

    // Padded smem to reduce bank conflicts
    __shared__ half smem_A[2][TM15][BK15 + PAD15];   // padded
    __shared__ half smem_B[2][BK15][TN15 + PAD15];   // padded

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    size_t const total_k_iters = (K + BK15 - 1) / BK15;

    // Load first tile with __ldg
    {
        for (int idx = threadIdx.x; idx < TM15 * BK15; idx += nthreads) {
            int const tm = idx % TM15;
            int const tk = idx / TM15;
            size_t const k = tk;

            half val = __float2half(0.0f);
            if (my_oh >= 0 && k < K) {
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                size_t const addr = ic * H * W + (size_t)(my_oh + fpos / 3) * W + (size_t)(my_ow + fpos % 3);
                val = __ldg(&input[addr]);
            }
            smem_A[0][tm][tk] = val;
        }
        for (int idx = threadIdx.x; idx < BK15 * TN15; idx += nthreads) {
            int const tn = idx % TN15;
            int const tk = idx / TN15;
            size_t const n = block_n + tn;

            half val = __float2half(0.0f);
            if (tk < K && n < N) {
                size_t const ic = tk / 9;
                size_t const fpos = tk % 9;
                val = __ldg(&filter[n * C_in * 9 + ic * 9 + fpos]);
            }
            smem_B[0][tk][tn] = val;
        }
    }
    __syncthreads();

    int buf = 0;

    for (size_t ki = 0; ki < total_k_iters; ki++) {
        int const next_buf = 1 - buf;
        size_t const next_k_offset = (ki + 1) * BK15;

        if (ki + 1 < total_k_iters) {
            for (int idx = threadIdx.x; idx < TM15 * BK15; idx += nthreads) {
                int const tm = idx % TM15;
                int const tk = idx / TM15;
                size_t const k = next_k_offset + tk;

                half val = __float2half(0.0f);
                if (my_oh >= 0 && k < K) {
                    size_t const ic = k / 9;
                    size_t const fpos = k % 9;
                    size_t const addr = ic * H * W + (size_t)(my_oh + fpos / 3) * W + (size_t)(my_ow + fpos % 3);
                    val = __ldg(&input[addr]);
                }
                smem_A[next_buf][tm][tk] = val;
            }
            for (int idx = threadIdx.x; idx < BK15 * TN15; idx += nthreads) {
                int const tn = idx % TN15;
                int const tk = idx / TN15;
                size_t const k = next_k_offset + tk;
                size_t const n = block_n + tn;

                half val = __float2half(0.0f);
                if (k < K && n < N) {
                    size_t const ic = k / 9;
                    size_t const fpos = k % 9;
                    val = __ldg(&filter[n * C_in * 9 + ic * 9 + fpos]);
                }
                smem_B[next_buf][tk][tn] = val;
            }
        }

        int const wm = warp_row * 32;
        int const wn = warp_col * 64;

        #pragma unroll
        for (int kk = 0; kk < BK15; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[buf][wm + mi * 16][kk], BK15 + PAD15);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[buf][kk][wn + ni * 16], TN15 + PAD15);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        __syncthreads();
        buf = next_buf;
    }

    // Store
    __shared__ int s_oh[TM15];
    __shared__ int s_ow[TM15];
    for (int idx = threadIdx.x; idx < TM15; idx += nthreads) {
        size_t const m = block_m + idx;
        if (m < M) { s_oh[idx] = (int)(m / out_W); s_ow[idx] = (int)(m % out_W); }
        else s_oh[idx] = -1;
    }
    __syncthreads();

    __shared__ half smem_store[16][16];
    int const wm = warp_row * 32;
    int const wn = warp_col * 64;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_half;

    for (int w = 0; w < NWARPS15; w++) {
        if (warp_id == w) {
            for (int mi = 0; mi < 2; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    for (int t = 0; t < c_frag[mi][ni].num_elements; t++)
                        c_half.x[t] = __float2half(c_frag[mi][ni].x[t]);
                    wmma::store_matrix_sync(&smem_store[0][0], c_half, 16, wmma::mem_row_major);
                    __syncwarp();
                    int const lane = threadIdx.x % 32;
                    for (int idx = lane; idx < 256; idx += 32) {
                        int const r = idx / 16;
                        int const c = idx % 16;
                        int const lm = wm + mi * 16 + r;
                        size_t const gn = block_n + wn + ni * 16 + c;
                        if (s_oh[lm] >= 0 && gn < N)
                            output[gn * out_H * out_W + s_oh[lm] * out_W + s_ow[lm]] = smem_store[r][c];
                    }
                }
            }
        }
        __syncthreads();
    }
}

void launch_wmma_half_padded_ldg_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM15 - 1) / TM15),
        (unsigned int)((C_out + TN15 - 1) / TN15));
    dim3 const block(NWARPS15 * 32);

    wmma_half_padded_ldg_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
