#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 57: Add #pragma unroll to load loops
// The A load loop iterates 32 times per thread. Partial unrolling may help
// by reducing loop overhead and enabling better instruction scheduling.
// Also try: use conditional moves instead of branching for boundary checks.

static constexpr int TM57 = 128;
static constexpr int TN57 = 128;
static constexpr int BK57 = 64;
static constexpr int PAD_A57 = 8;
static constexpr int PAD_B57 = 8;
static constexpr int NWARPS57 = 8;

__global__ void __launch_bounds__(256, 2)
wmma_unrollfull_conv2d_3x3_kernel(
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

    int const block_m = blockIdx.x * TM57;
    int const block_n = blockIdx.y * TN57;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS57 * 32;

    int const my_tm = threadIdx.x % TM57;
    int const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (my_m % out_W) : -1;
    int const my_base = (my_oh >= 0) ? my_oh * W + my_ow : 0;

    __shared__ half smem_A[TM57][BK57 + PAD_A57];
    __shared__ half smem_B[BK57][TN57 + PAD_B57];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int const total_k_iters = (K + BK57 - 1) / BK57;

    // Precompute: is this thread's spatial position valid?
    int const valid = (my_oh >= 0) ? 1 : 0;

    for (int ki = 0; ki < total_k_iters; ki++) {
        int const k_base = ki * BK57;

        // Load A with partial unroll
        #pragma unroll
        for (int idx = threadIdx.x; idx < TM57 * BK57; idx += nthreads) {
            int const tm = idx & (TM57 - 1);  // idx % 128 = idx & 127 (TM is power of 2)
            int const tk = idx >> 7;           // idx / 128 = idx >> 7
            int const k = k_base + tk;
            half val = __float2half(0.0f);
            if (valid && k < K) {
                int ic = k / 9;
                int fpos = k - ic * 9;
                int fy = fpos / 3;
                int fx = fpos - fy * 3;
                val = __ldg(&input[ic * HW + fy * W + fx + my_base]);
            }
            smem_A[tm][tk] = val;
        }
        // Load B with partial unroll
        #pragma unroll
        for (int idx = threadIdx.x; idx < BK57 * TN57; idx += nthreads) {
            int const tk = idx & (BK57 - 1);  // idx % 64 = idx & 63
            int const tn = idx >> 6;           // idx / 64 = idx >> 6
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
        for (int kk = 0; kk < BK57; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[wm + mi * 16][kk], BK57 + PAD_A57);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[kk][wn + ni * 16], TN57 + PAD_B57);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }
        __syncthreads();
    }

    __shared__ half smem_store[NWARPS57][16][16];
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

void launch_wmma_unrollfull_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    int const out_H = (int)H - 2;
    int const out_W = (int)W - 2;
    int const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM57 - 1) / TM57),
        (unsigned int)((C_out + TN57 - 1) / TN57));
    dim3 const block(NWARPS57 * 32);

    wmma_unrollfull_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, (int)C_in, (int)C_out, (int)H, (int)W);
    CHECK_LAST_CUDA_ERROR();
}
