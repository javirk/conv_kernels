#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 33: 32-bit int address math without aggressive launch bounds
// Exp 32 showed launch_bounds(256,3) forced spilling. Keep bounds at (256,2)
// but use int arithmetic to reduce instruction count per address calculation.
// 64-bit div/mod/mul each take 2x the instructions of 32-bit on SM 8.6.

static constexpr int TM33 = 128;
static constexpr int TN33 = 128;
static constexpr int BK33 = 32;
static constexpr int PAD33 = 8;
static constexpr int NWARPS33 = 8;

__global__ void __launch_bounds__(256, 2)
wmma_int_addr_ns_conv2d_3x3_kernel(
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

    int const block_m = blockIdx.x * TM33;
    int const block_n = blockIdx.y * TN33;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS33 * 32;

    int const my_tm = threadIdx.x % TM33;
    int const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (my_m % out_W) : -1;
    int const my_base = (my_oh >= 0) ? my_oh * W + my_ow : 0;

    __shared__ half smem_A[2][TM33][BK33 + PAD33];
    __shared__ half smem_B[2][BK33][TN33 + PAD33];
    __shared__ int k_offsets[2][BK33];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int const total_k_iters = (K + BK33 - 1) / BK33;

    // Precompute k_offsets for first tile
    if (threadIdx.x < BK33) {
        int k = threadIdx.x;
        if (k < K) {
            int ic = k / 9;
            int fpos = k - ic * 9;
            int fy = fpos / 3;
            int fx = fpos - fy * 3;
            k_offsets[0][threadIdx.x] = ic * HW + fy * W + fx;
        } else {
            k_offsets[0][threadIdx.x] = 0;
        }
    }
    __syncthreads();

    // Load first A tile
    for (int idx = threadIdx.x; idx < TM33 * BK33; idx += nthreads) {
        int const tm = idx % TM33;
        int const tk = idx / TM33;

        half val = __float2half(0.0f);
        if (my_oh >= 0 && tk < K) {
            val = __ldg(&input[k_offsets[0][tk] + my_base]);
        }
        smem_A[0][tm][tk] = val;
    }
    for (int idx = threadIdx.x; idx < BK33 * TN33; idx += nthreads) {
        int const tk = idx % BK33;
        int const tn = idx / BK33;
        int const n = block_n + tn;

        half val = __float2half(0.0f);
        if (tk < K && n < N) {
            val = __ldg(&filter[n * K + tk]);
        }
        smem_B[0][tk][tn] = val;
    }
    __syncthreads();

    int buf = 0;

    for (int ki = 0; ki < total_k_iters; ki++) {
        int const next_buf = 1 - buf;
        int const next_k_base = (ki + 1) * BK33;

        if (ki + 1 < total_k_iters) {
            if (threadIdx.x < BK33) {
                int k = next_k_base + threadIdx.x;
                if (k < K) {
                    int ic = k / 9;
                    int fpos = k - ic * 9;
                    int fy = fpos / 3;
                    int fx = fpos - fy * 3;
                    k_offsets[next_buf][threadIdx.x] = ic * HW + fy * W + fx;
                } else {
                    k_offsets[next_buf][threadIdx.x] = 0;
                }
            }
            for (int idx = threadIdx.x; idx < TM33 * BK33; idx += nthreads) {
                int const tm = idx % TM33;
                int const tk = idx / TM33;
                int const k = next_k_base + tk;

                half val = __float2half(0.0f);
                if (my_oh >= 0 && k < K) {
                    val = __ldg(&input[k_offsets[next_buf][tk] + my_base]);
                }
                smem_A[next_buf][tm][tk] = val;
            }
            for (int idx = threadIdx.x; idx < BK33 * TN33; idx += nthreads) {
                int const tk = idx % BK33;
                int const tn = idx / BK33;
                int const k = next_k_base + tk;
                int const n = block_n + tn;

                half val = __float2half(0.0f);
                if (k < K && n < N) {
                    val = __ldg(&filter[n * K + k]);
                }
                smem_B[next_buf][tk][tn] = val;
            }
        }

        int const wm = warp_row * 32;
        int const wn = warp_col * 64;

        #pragma unroll
        for (int kk = 0; kk < BK33; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[buf][wm + mi * 16][kk], BK33 + PAD33);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[buf][kk][wn + ni * 16], TN33 + PAD33);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        __syncthreads();
        buf = next_buf;
    }

    // Store output
    __shared__ half smem_store[NWARPS33][16][16];

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
                int const r = idx / 16;
                int const c = idx % 16;
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

void launch_wmma_int_addr_ns_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    int const out_H = (int)H - 2;
    int const out_W = (int)W - 2;
    int const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM33 - 1) / TM33),
        (unsigned int)((C_out + TN33 - 1) / TN33));
    dim3 const block(NWARPS33 * 32);

    wmma_int_addr_ns_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, (int)C_in, (int)C_out, (int)H, (int)W);
    CHECK_LAST_CUDA_ERROR();
}
