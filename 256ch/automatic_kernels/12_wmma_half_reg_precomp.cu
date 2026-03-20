#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 12: Register-precomputed oh/ow with transposed load
// Key insight: with transposed A load, tm = idx % TM is the SAME for each
// thread across all K iterations (since nthreads=256, TM=128, 256%128=0).
// So each thread precomputes oh,ow ONCE into registers.
// Result: ZERO div/mod in the inner loop!
//
// 128x128 tile, BK=32, 8 warps, double buffered

static constexpr int TM12 = 128;
static constexpr int TN12 = 128;
static constexpr int BK12 = 32;
static constexpr int NWARPS12 = 8;

__global__ void wmma_half_reg_precomp_conv2d_3x3_kernel(
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

    size_t const block_m = blockIdx.x * TM12;
    size_t const block_n = blockIdx.y * TN12;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS12 * 32;

    // Precompute oh, ow in REGISTERS (done once, reused for all K iterations)
    // With transposed load: tm = threadIdx.x % TM12 (same across all K steps)
    int const my_tm = threadIdx.x % TM12;
    size_t const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (int)(my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (int)(my_m % out_W) : -1;

    __shared__ half smem_A[2][TM12][BK12];   // 16KB
    __shared__ half smem_B[2][BK12][TN12];   // 16KB

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    size_t const total_k_iters = (K + BK12 - 1) / BK12;

    // Load first tile
    {
        // A load: each thread handles (my_tm, tk) for tk = threadIdx.x / TM12
        // Since nthreads=256, TM=128: each thread handles 2 columns
        // Actually: total elements = 128*32 = 4096, 256 threads → 16 per thread
        // idx = threadIdx.x, threadIdx.x+256, ..., +3840
        // tm = idx % 128, tk = idx / 128
        for (int idx = threadIdx.x; idx < TM12 * BK12; idx += nthreads) {
            int const tm = idx % TM12;
            int const tk = idx / TM12;
            size_t const k = tk;  // k_offset = 0

            half val = __float2half(0.0f);
            // For this thread's tm: if tm == my_tm, use precomputed oh/ow
            // But other threads handle tm != my_tm... Actually, since 256%128=0,
            // all idx for this thread have tm = threadIdx.x % 128 = my_tm
            if (my_oh >= 0 && k < K) {
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                val = input[ic * H * W + (size_t)(my_oh + fpos / 3) * W + (size_t)(my_ow + fpos % 3)];
            }
            smem_A[0][tm][tk] = val;
        }
        // B load
        for (int idx = threadIdx.x; idx < BK12 * TN12; idx += nthreads) {
            int const tn = idx % TN12;
            int const tk = idx / TN12;
            size_t const k = tk;
            size_t const n = block_n + tn;

            half val = __float2half(0.0f);
            if (k < K && n < N) {
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                val = filter[n * C_in * 9 + ic * 9 + fpos];
            }
            smem_B[0][tk][tn] = val;
        }
    }
    __syncthreads();

    int buf = 0;

    for (size_t ki = 0; ki < total_k_iters; ki++) {
        int const next_buf = 1 - buf;
        size_t const next_k_offset = (ki + 1) * BK12;

        if (ki + 1 < total_k_iters) {
            // A load using precomputed oh/ow (NO div/mod!)
            for (int idx = threadIdx.x; idx < TM12 * BK12; idx += nthreads) {
                int const tm = idx % TM12;
                int const tk = idx / TM12;
                size_t const k = next_k_offset + tk;

                half val = __float2half(0.0f);
                if (my_oh >= 0 && k < K) {
                    size_t const ic = k / 9;
                    size_t const fpos = k % 9;
                    val = input[ic * H * W + (size_t)(my_oh + fpos / 3) * W + (size_t)(my_ow + fpos % 3)];
                }
                smem_A[next_buf][tm][tk] = val;
            }
            // B load
            for (int idx = threadIdx.x; idx < BK12 * TN12; idx += nthreads) {
                int const tn = idx % TN12;
                int const tk = idx / TN12;
                size_t const k = next_k_offset + tk;
                size_t const n = block_n + tn;

                half val = __float2half(0.0f);
                if (k < K && n < N) {
                    size_t const ic = k / 9;
                    size_t const fpos = k % 9;
                    val = filter[n * C_in * 9 + ic * 9 + fpos];
                }
                smem_B[next_buf][tk][tn] = val;
            }
        }

        int const wm = warp_row * 32;
        int const wn = warp_col * 64;

        #pragma unroll
        for (int kk = 0; kk < BK12; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[buf][wm + mi * 16][kk], BK12);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[buf][kk][wn + ni * 16], TN12);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        __syncthreads();
        buf = next_buf;
    }

    // Store: precompute oh/ow for store positions in smem
    __shared__ int s_oh[TM12];
    __shared__ int s_ow[TM12];
    for (int idx = threadIdx.x; idx < TM12; idx += nthreads) {
        size_t const m = block_m + idx;
        if (m < M) {
            s_oh[idx] = (int)(m / out_W);
            s_ow[idx] = (int)(m % out_W);
        } else {
            s_oh[idx] = -1;
        }
    }
    __syncthreads();

    __shared__ half smem_store[16][16];
    int const wm = warp_row * 32;
    int const wn = warp_col * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_half;

    for (int w = 0; w < NWARPS12; w++) {
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

void launch_wmma_half_reg_precomp_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM12 - 1) / TM12),
        (unsigned int)((C_out + TN12 - 1) / TN12));
    dim3 const block(NWARPS12 * 32);

    wmma_half_reg_precomp_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
