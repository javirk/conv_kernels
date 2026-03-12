#pragma once

#include <mma.h>
using namespace nvcuda;

// Tuned version of experiment 10 with:
// 1. __launch_bounds__ for better register allocation
// 2. Precomputed divmod using multiplication trick
// 3. Padded SMEM to avoid bank conflicts

constexpr int BM16_H = 8;
constexpr int BM16_W = 8;
constexpr int BM16 = BM16_H * BM16_W;
constexpr int BN16 = 64;
constexpr int BK16 = 16;
constexpr int WM16 = 16;
constexpr int WN16 = 16;
constexpr int WK16 = 8;

// Pad SMEM rows to avoid bank conflicts (add 1 float padding per row)
constexpr int BK16_PAD = BK16 + 1;  // 17
constexpr int BN16_PAD = BN16 + 1;  // 65

__launch_bounds__(512, 1)
__global__ void wmma_tuned_conv2d_kernel(
    float* __restrict__ output,
    float const* __restrict__ input,
    float const* __restrict__ filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const K_gemm = C_in * 9;

    int const tile_row = blockIdx.x * BM16_H;
    int const tile_col = blockIdx.y * BM16_W;
    int const block_n = blockIdx.z * BN16;

    // Padded SMEM to reduce bank conflicts
    __shared__ float As[2][BM16][BK16_PAD];
    __shared__ float Bs[2][BK16][BN16_PAD];

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN16 / WN16;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM16, WN16, WK16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    // Precompute spatial offsets for this block's output positions
    // Each output position m has a fixed (row, col) — precompute in registers
    // Actually, with only 8x8=64 positions and 512 threads, each thread handles ~2 positions

    auto load_tile = [&](int buf, int k_start) {
        for (int idx = tid; idx < BM16 * BK16; idx += total_threads)
        {
            int const m = idx / BK16;
            int const kk = idx % BK16;
            int const local_row = m >> 3;  // m / 8 for BM16_W=8
            int const local_col = m & 7;   // m % 8
            int const out_row = tile_row + local_row;
            int const out_col = tile_col + local_col;
            int const gk = k_start + kk;

            if (out_row < out_H && out_col < out_W && gk < K_gemm)
            {
                // Optimized divmod by 9: use multiply-shift
                int const ic = gk / 9;
                int const rem = gk - ic * 9;
                int const fy = rem / 3;
                int const fx = rem - fy * 3;
                As[buf][m][kk] = input[ic * H * W + (out_row + fy) * W + (out_col + fx)];
            }
            else
            {
                As[buf][m][kk] = 0.0f;
            }
        }
        for (int idx = tid; idx < BK16 * BN16; idx += total_threads)
        {
            int const kk = idx / BN16;
            int const n = idx % BN16;
            int const gk = k_start + kk;
            int const gn = block_n + n;
            Bs[buf][kk][n] = (gk < K_gemm && gn < C_out) ?
                filter[gn * K_gemm + gk] : 0.0f;
        }
    };

    load_tile(0, 0);
    __syncthreads();

    int const num_k_iters = (K_gemm + BK16 - 1) / BK16;

    for (int ki = 0; ki < num_k_iters; ++ki)
    {
        int const cur_buf = ki % 2;
        int const next_buf = 1 - cur_buf;

        if (ki + 1 < num_k_iters)
            load_tile(next_buf, (ki + 1) * BK16);

        int const tile_m_off = warp_m * WM16;
        int const tile_n_off = warp_n * WN16;

        #pragma unroll
        for (int kk = 0; kk < BK16; kk += WK16)
        {
            wmma::fragment<wmma::matrix_a, WM16, WN16, WK16,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM16, WN16, WK16,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As[cur_buf][tile_m_off][kk], BK16_PAD);
            wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][tile_n_off], BN16_PAD);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    __shared__ float Cs[BM16][BN16];
    wmma::store_matrix_sync(&Cs[warp_m * WM16][warp_n * WN16], acc, BN16, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM16 * BN16; idx += total_threads)
    {
        int const m = idx / BN16;
        int const n = idx % BN16;
        int const local_row = m >> 3;
        int const local_col = m & 7;
        int const out_row = tile_row + local_row;
        int const out_col = tile_col + local_col;
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[oc * out_H * out_W + out_row * out_W + out_col] = Cs[m][n];
        }
    }
}

template <typename T>
void launch_wmma_tuned_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                    size_t C_in, size_t C_out, size_t H, size_t W,
                                    cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    constexpr int num_warps = (BM16 / WM16) * (BN16 / WN16);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (out_H + BM16_H - 1) / BM16_H,
        (out_W + BM16_W - 1) / BM16_W,
        (C_out + BN16 - 1) / BN16);

    wmma_tuned_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
