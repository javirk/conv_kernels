#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 11: Native half + transposed A load for perfect coalescing
// Key insight: NCHW layout has consecutive ow values contiguous.
// If consecutive threads load consecutive m values (same k), the global
// loads are perfectly coalesced since m variations = ow variations.
//
// For block_m covering 128 consecutive spatial positions:
// - All positions likely in same row (out_W=510 >> 128)
// - input[ic][oh+fy][ow+fx], input[ic][oh+fy][ow+fx+1], ... → contiguous!
//
// 128x128 tile, BK=32, 8 warps, double buffered

static constexpr int TM11 = 128;
static constexpr int TN11 = 128;
static constexpr int BK11 = 32;
static constexpr int NWARPS11 = 8;

__global__ void wmma_half_coalesced_conv2d_3x3_kernel(
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

    size_t const block_m = blockIdx.x * TM11;
    size_t const block_n = blockIdx.y * TN11;

    int const warp_id = threadIdx.x / 32;
    int const warp_row = warp_id / 2;
    int const warp_col = warp_id % 2;
    int const nthreads = NWARPS11 * 32;

    __shared__ half smem_A[2][TM11][BK11];   // 16KB
    __shared__ half smem_B[2][BK11][TN11];   // 16KB

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    size_t const total_k_iters = (K + BK11 - 1) / BK11;

    // Helper: load tile into smem buffer
    // A load: TRANSPOSED - consecutive threads → consecutive m (coalesced)
    auto load_a = [&](int buf_idx, size_t k_offset) {
        for (int idx = threadIdx.x; idx < TM11 * BK11; idx += nthreads) {
            int const tm = idx % TM11;   // TRANSPOSED: consecutive threads → consecutive m
            int const tk = idx / TM11;   // same k for warp
            size_t const m = block_m + tm;
            size_t const k = k_offset + tk;

            half val = __float2half(0.0f);
            if (m < M && k < K) {
                size_t const oh = m / out_W;
                size_t const ow = m % out_W;
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                val = input[ic * H * W + (oh + fpos / 3) * W + (ow + fpos % 3)];
            }
            smem_A[buf_idx][tm][tk] = val;
        }
    };

    auto load_b = [&](int buf_idx, size_t k_offset) {
        for (int idx = threadIdx.x; idx < BK11 * TN11; idx += nthreads) {
            int const tn = idx % TN11;   // consecutive threads → consecutive oc
            int const tk = idx / TN11;
            size_t const k = k_offset + tk;
            size_t const n = block_n + tn;

            half val = __float2half(0.0f);
            if (k < K && n < N) {
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                val = filter[n * C_in * 9 + ic * 9 + fpos];
            }
            smem_B[buf_idx][tk][tn] = val;
        }
    };

    // Preload first tile
    load_a(0, 0);
    load_b(0, 0);
    __syncthreads();

    int buf = 0;

    for (size_t ki = 0; ki < total_k_iters; ki++) {
        int const next_buf = 1 - buf;
        size_t const next_k_offset = (ki + 1) * BK11;

        if (ki + 1 < total_k_iters) {
            load_a(next_buf, next_k_offset);
            load_b(next_buf, next_k_offset);
        }

        int const wm = warp_row * 32;
        int const wn = warp_col * 64;

        #pragma unroll
        for (int kk = 0; kk < BK11; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[buf][wm + mi * 16][kk], BK11);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[buf][kk][wn + ni * 16], TN11);

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
    __shared__ half smem_store[16][16];
    int const wm = warp_row * 32;
    int const wn = warp_col * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_half;

    for (int w = 0; w < NWARPS11; w++) {
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
                        size_t const gm = block_m + wm + mi * 16 + r;
                        size_t const gn = block_n + wn + ni * 16 + c;
                        if (gm < M && gn < N) {
                            size_t const oh = gm / out_W;
                            size_t const ow = gm % out_W;
                            output[gn * out_H * out_W + oh * out_W + ow] = smem_store[r][c];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

void launch_wmma_half_coalesced_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM11 - 1) / TM11),
        (unsigned int)((C_out + TN11 - 1) / TN11));
    dim3 const block(NWARPS11 * 32);

    wmma_half_coalesced_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
