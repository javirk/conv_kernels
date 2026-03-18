#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 29: Precompute k→spatial offset lookup in shared memory
// Profile shows 41% compute, 29% memory — bottleneck is loop overhead.
// Each A load does 4 integer divisions (ic=k/9, fpos=k%9, fy=fpos/3, fx=fpos%3).
// Precompute offset[tk] = ic*H*W + fy*W + fx in smem once per K-iteration.
// A load becomes: input[offset[tk] + my_oh*W + my_ow] (1 smem read, 0 divisions).
// Also keeps coalesced B load from exp 27.

static constexpr int TM29 = 128;
static constexpr int TN29 = 128;
static constexpr int BK29 = 32;
static constexpr int PAD29 = 8;
static constexpr int NWARPS29 = 8;

__global__ void __launch_bounds__(256, 2)
wmma_precomp_k_offsets_conv2d_3x3_kernel(
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

    size_t const block_m = blockIdx.x * TM29;
    size_t const block_n = blockIdx.y * TN29;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS29 * 32;

    int const my_tm = threadIdx.x % TM29;
    size_t const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (int)(my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (int)(my_m % out_W) : -1;
    // Precompute per-thread base address (oh*W + ow is constant per thread)
    size_t const my_base = (my_oh >= 0) ? (size_t)my_oh * W + (size_t)my_ow : 0;

    __shared__ half smem_A[2][TM29][BK29 + PAD29];
    __shared__ half smem_B[2][BK29][TN29 + PAD29];
    // Precomputed k→offset for current and next tile
    __shared__ size_t k_offsets[2][BK29];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    size_t const total_k_iters = (K + BK29 - 1) / BK29;

    // Precompute k_offsets for first tile
    if (threadIdx.x < BK29) {
        size_t k = threadIdx.x;
        if (k < K) {
            size_t ic = k / 9;
            size_t fpos = k % 9;
            k_offsets[0][threadIdx.x] = ic * H * W + (fpos / 3) * W + (fpos % 3);
        } else {
            k_offsets[0][threadIdx.x] = 0;
        }
    }
    __syncthreads();

    // Load first A tile using precomputed offsets
    for (int idx = threadIdx.x; idx < TM29 * BK29; idx += nthreads) {
        int const tm = idx % TM29;
        int const tk = idx / TM29;

        half val = __float2half(0.0f);
        if (my_oh >= 0 && (size_t)tk < K) {
            val = __ldg(&input[k_offsets[0][tk] + my_base]);
        }
        smem_A[0][tm][tk] = val;
    }
    // Load first B tile (coalesced: adjacent threads access consecutive k)
    for (int idx = threadIdx.x; idx < BK29 * TN29; idx += nthreads) {
        int const tk = idx % BK29;
        int const tn = idx / BK29;
        size_t const n = block_n + tn;

        half val = __float2half(0.0f);
        if ((size_t)tk < K && n < N) {
            val = __ldg(&filter[n * K + tk]);
        }
        smem_B[0][tk][tn] = val;
    }
    __syncthreads();

    int buf = 0;

    for (size_t ki = 0; ki < total_k_iters; ki++) {
        int const next_buf = 1 - buf;
        size_t const next_k_offset = (ki + 1) * BK29;

        if (ki + 1 < total_k_iters) {
            // Precompute k_offsets for next tile (first BK threads)
            if (threadIdx.x < BK29) {
                size_t k = next_k_offset + threadIdx.x;
                if (k < K) {
                    size_t ic = k / 9;
                    size_t fpos = k - ic * 9;  // avoid modulo
                    size_t fy = fpos / 3;
                    size_t fx = fpos - fy * 3;  // avoid modulo
                    k_offsets[next_buf][threadIdx.x] = ic * H * W + fy * W + fx;
                } else {
                    k_offsets[next_buf][threadIdx.x] = 0;
                }
            }
            // Load next A tile
            for (int idx = threadIdx.x; idx < TM29 * BK29; idx += nthreads) {
                int const tm = idx % TM29;
                int const tk = idx / TM29;
                size_t const k = next_k_offset + tk;

                half val = __float2half(0.0f);
                if (my_oh >= 0 && k < K) {
                    val = __ldg(&input[k_offsets[next_buf][tk] + my_base]);
                }
                smem_A[next_buf][tm][tk] = val;
            }
            // Coalesced B load
            for (int idx = threadIdx.x; idx < BK29 * TN29; idx += nthreads) {
                int const tk = idx % BK29;
                int const tn = idx / BK29;
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
        for (int kk = 0; kk < BK29; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[buf][wm + mi * 16][kk], BK29 + PAD29);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[buf][kk][wn + ni * 16], TN29 + PAD29);

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
    __shared__ half smem_store[NWARPS29][16][16];

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

void launch_wmma_precomp_k_offsets_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM29 - 1) / TM29),
        (unsigned int)((C_out + TN29 - 1) / TN29));
    dim3 const block(NWARPS29 * 32);

    wmma_precomp_k_offsets_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
