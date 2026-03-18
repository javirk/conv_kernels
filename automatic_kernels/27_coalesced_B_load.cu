#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 27: Coalesced B (filter) loads
// Fix: swap B load indexing so adjacent threads access consecutive k values
// (contiguous in memory) rather than consecutive n values (strided by C_in*9)
// Also keeps parallel warp store from exp 26

static constexpr int TM27 = 128;
static constexpr int TN27 = 128;
static constexpr int BK27 = 32;
static constexpr int PAD27 = 8;
static constexpr int NWARPS27 = 8;

__global__ void wmma_coalesced_B_conv2d_3x3_kernel(
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

    size_t const block_m = blockIdx.x * TM27;
    size_t const block_n = blockIdx.y * TN27;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS27 * 32;

    int const my_tm = threadIdx.x % TM27;
    size_t const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (int)(my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (int)(my_m % out_W) : -1;

    __shared__ half smem_A[2][TM27][BK27 + PAD27];
    __shared__ half smem_B[2][BK27][TN27 + PAD27];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    size_t const total_k_iters = (K + BK27 - 1) / BK27;

    // Lambda-like macro for loading A tile (same as before - already coalesced in spatial dim)
    // Load first tile
    {
        for (int idx = threadIdx.x; idx < TM27 * BK27; idx += nthreads) {
            int const tm = idx % TM27;
            int const tk = idx / TM27;

            half val = __float2half(0.0f);
            if (my_oh >= 0 && (size_t)tk < K) {
                size_t const ic = tk / 9;
                size_t const fpos = tk % 9;
                val = __ldg(&input[ic * H * W + (size_t)(my_oh + fpos / 3) * W + (size_t)(my_ow + fpos % 3)]);
            }
            smem_A[0][tm][tk] = val;
        }
        // B load: swap indexing for coalesced access
        // filter layout: [C_out][C_in*9], so filter[n * K + k]
        // Adjacent threads should access adjacent k (contiguous in memory)
        for (int idx = threadIdx.x; idx < BK27 * TN27; idx += nthreads) {
            int const tk = idx % BK27;   // k-major: adjacent threads -> adjacent k -> contiguous memory
            int const tn = idx / BK27;
            size_t const n = block_n + tn;

            half val = __float2half(0.0f);
            if ((size_t)tk < K && n < N) {
                val = __ldg(&filter[n * K + tk]);
            }
            smem_B[0][tk][tn] = val;
        }
    }
    __syncthreads();

    int buf = 0;

    for (size_t ki = 0; ki < total_k_iters; ki++) {
        int const next_buf = 1 - buf;
        size_t const next_k_offset = (ki + 1) * BK27;

        if (ki + 1 < total_k_iters) {
            for (int idx = threadIdx.x; idx < TM27 * BK27; idx += nthreads) {
                int const tm = idx % TM27;
                int const tk = idx / TM27;
                size_t const k = next_k_offset + tk;

                half val = __float2half(0.0f);
                if (my_oh >= 0 && k < K) {
                    size_t const ic = k / 9;
                    size_t const fpos = k % 9;
                    val = __ldg(&input[ic * H * W + (size_t)(my_oh + fpos / 3) * W + (size_t)(my_ow + fpos % 3)]);
                }
                smem_A[next_buf][tm][tk] = val;
            }
            // Coalesced B load
            for (int idx = threadIdx.x; idx < BK27 * TN27; idx += nthreads) {
                int const tk = idx % BK27;
                int const tn = idx / BK27;
                size_t const k = next_k_offset + tk;
                size_t const n = block_n + tn;

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
        for (int kk = 0; kk < BK27; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[buf][wm + mi * 16][kk], BK27 + PAD27);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[buf][kk][wn + ni * 16], TN27 + PAD27);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        __syncthreads();
        buf = next_buf;
    }

    // Parallel store
    __shared__ half smem_store[NWARPS27][16][16];

    int const wm = warp_row * 32;
    int const wn = warp_col * 64;
    int const lane = threadIdx.x % 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_half;

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
                size_t const gn = block_n + wn + ni * 16 + c;
                size_t const gm = block_m + lm;
                if (gm < M && gn < N) {
                    size_t const oh = gm / out_W;
                    size_t const ow = gm % out_W;
                    output[gn * out_H * out_W + oh * out_W + ow] = smem_store[warp_id][r][c];
                }
            }
        }
    }
}

void launch_wmma_coalesced_B_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM27 - 1) / TM27),
        (unsigned int)((C_out + TN27 - 1) / TN27));
    dim3 const block(NWARPS27 * 32);

    wmma_coalesced_B_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
