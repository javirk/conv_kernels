#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 42: BK=64 without double buffering
// BK=64 halves K-iterations (36 vs 72), reducing sync overhead.
// No double buffer keeps smem at ~40KB for 2 blocks/SM.
// Trade-off: lose compute-load overlap but gain from fewer iterations.
// With BK=64: 4 WMMA k-steps per iteration → more compute per sync.

static constexpr int TM42 = 128;
static constexpr int TN42 = 128;
static constexpr int BK42 = 64;
static constexpr int PAD_A42 = 8;
static constexpr int PAD_B42 = 8;
static constexpr int NWARPS42 = 8;

__global__ void __launch_bounds__(256, 2)
wmma_bk64_conv2d_3x3_kernel(
    half* __restrict__ output,
    half const* __restrict__ input,
    half const* __restrict__ filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const M = out_H * out_W;
    int const N = C_out;
    int const K = C_in * 9;
    int const HW = H * W;

    int const block_m = blockIdx.x * TM42;
    int const block_n = blockIdx.y * TN42;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS42 * 32;

    int const my_tm = threadIdx.x % TM42;
    int const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (my_m % out_W) : -1;
    int const my_base = (my_oh >= 0) ? my_oh * W + my_ow : 0;

    // Single buffer (no double buffer) - saves ~36KB smem
    __shared__ half smem_A[TM42][BK42 + PAD_A42];
    __shared__ half smem_B[BK42][TN42 + PAD_B42];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int const total_k_iters = (K + BK42 - 1) / BK42;

    for (int ki = 0; ki < total_k_iters; ki++) {
        int const k_base = ki * BK42;

        // Load A tile
        for (int idx = threadIdx.x; idx < TM42 * BK42; idx += nthreads) {
            int const tm = idx % TM42;
            int const tk = idx / TM42;
            int const k = k_base + tk;
            half val = __float2half(0.0f);
            if (my_oh >= 0 && k < K) {
                int ic = k / 9;
                int fpos = k - ic * 9;
                int fy = fpos / 3;
                int fx = fpos - fy * 3;
                val = __ldg(&input[ic * HW + fy * W + fx + my_base]);
            }
            smem_A[tm][tk] = val;
        }
        // Load B tile
        for (int idx = threadIdx.x; idx < BK42 * TN42; idx += nthreads) {
            int const tk = idx % BK42;
            int const tn = idx / BK42;
            int const k = k_base + tk;
            int const n = block_n + tn;
            half val = __float2half(0.0f);
            if (k < K && n < N) {
                val = __ldg(&filter[n * K + k]);
            }
            smem_B[tk][tn] = val;
        }
        __syncthreads();

        int const wm = warp_row * 32;
        int const wn = warp_col * 64;

        #pragma unroll
        for (int kk = 0; kk < BK42; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[wm + mi * 16][kk], BK42 + PAD_A42);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[kk][wn + ni * 16], TN42 + PAD_B42);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        __syncthreads();
    }

    // Coalesced store (from exp 41)
    __shared__ half smem_store[NWARPS42][16][16];

    int const wm = warp_row * 32;
    int const wn = warp_col * 64;
    int const lane = threadIdx.x % 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_half;
    int const out_HW = out_H * out_W;

    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 4; ni++) {
            #pragma unroll
            for (int t = 0; t < c_frag[mi][ni].num_elements; t++)
                c_half.x[t] = __float2half(c_frag[mi][ni].x[t]);
            wmma::store_matrix_sync(&smem_store[warp_id][0][0], c_half, 16, wmma::mem_row_major);
            __syncwarp();

            for (int idx = lane; idx < 256; idx += 32) {
                int const r = idx % 16;
                int const c = idx / 16;
                int const lm = wm + mi * 16 + r;
                int const gn = block_n + wn + ni * 16 + c;
                int const gm = block_m + lm;
                if (gm < M && gn < N) {
                    int const oh = gm / out_W;
                    int const ow = gm - oh * out_W;
                    output[gn * out_HW + oh * out_W + ow] = smem_store[warp_id][r][c];
                }
            }
        }
    }
}

void launch_wmma_bk64_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    int const out_H = (int)H - 2;
    int const out_W = (int)W - 2;
    int const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM42 - 1) / TM42),
        (unsigned int)((C_out + TN42 - 1) / TN42));
    dim3 const block(NWARPS42 * 32);

    wmma_bk64_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, (int)C_in, (int)C_out, (int)H, (int)W);
    CHECK_LAST_CUDA_ERROR();
}
