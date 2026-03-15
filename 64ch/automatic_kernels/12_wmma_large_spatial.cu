#pragma once

#include <mma.h>
using namespace nvcuda;

// Larger spatial tile (16x16=256 output positions), BN=64 OC, BK=16.
// Each warp computes 16x16 output sub-tile. 16 warps in M, 4 warps in N = too many.
// Instead: BM=128 (8x16), BN=32, 8 warps M * 2 warps N = 16 warps = 512 threads.
// This keeps SMEM within 48KB while processing more spatial positions.

constexpr int BM12_H = 8;
constexpr int BM12_W = 16;
constexpr int BM12 = BM12_H * BM12_W;  // 128
constexpr int BN12 = 32;
constexpr int BK12 = 16;

constexpr int WM12 = 16;
constexpr int WN12 = 16;
constexpr int WK12 = 8;

__global__ void wmma_large_spatial_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const K_gemm = C_in * 9;

    int const tile_row = blockIdx.x * BM12_H;
    int const tile_col = blockIdx.y * BM12_W;
    int const block_n = blockIdx.z * BN12;

    // Double buffers
    __shared__ float As[2][BM12][BK12];   // 2*128*16 = 4096 floats = 16KB
    __shared__ float Bs[2][BK12][BN12];   // 2*16*32 = 1024 floats = 4KB
    // Total AB: 20KB. Cs: 128*32 = 4096 = 16KB. Total peak: 36KB < 48KB

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN12 / WN12;  // 2
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM12, WN12, WK12, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    auto load_a = [&](int buf, int k_start) {
        for (int idx = tid; idx < BM12 * BK12; idx += total_threads)
        {
            int const m = idx / BK12;
            int const kk = idx % BK12;
            int const local_row = m / BM12_W;
            int const local_col = m % BM12_W;
            int const out_row = tile_row + local_row;
            int const out_col = tile_col + local_col;
            int const gk = k_start + kk;

            if (out_row < out_H && out_col < out_W && gk < K_gemm)
            {
                int const ic = gk / 9;
                int const rem = gk % 9;
                int const fy = rem / 3;
                int const fx = rem % 3;
                As[buf][m][kk] = input[ic * H * W + (out_row + fy) * W + (out_col + fx)];
            }
            else
            {
                As[buf][m][kk] = 0.0f;
            }
        }
    };

    auto load_b = [&](int buf, int k_start) {
        for (int idx = tid; idx < BK12 * BN12; idx += total_threads)
        {
            int const kk = idx / BN12;
            int const n = idx % BN12;
            int const gk = k_start + kk;
            int const gn = block_n + n;
            Bs[buf][kk][n] = (gk < K_gemm && gn < C_out) ?
                filter[gn * K_gemm + gk] : 0.0f;
        }
    };

    load_a(0, 0);
    load_b(0, 0);
    __syncthreads();

    int const num_k_iters = (K_gemm + BK12 - 1) / BK12;

    for (int ki = 0; ki < num_k_iters; ++ki)
    {
        int const cur_buf = ki % 2;
        int const next_buf = 1 - cur_buf;

        if (ki + 1 < num_k_iters)
        {
            load_a(next_buf, (ki + 1) * BK12);
            load_b(next_buf, (ki + 1) * BK12);
        }

        int const tile_m_off = warp_m * WM12;
        int const tile_n_off = warp_n * WN12;

        #pragma unroll
        for (int kk = 0; kk < BK12; kk += WK12)
        {
            wmma::fragment<wmma::matrix_a, WM12, WN12, WK12,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM12, WN12, WK12,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As[cur_buf][tile_m_off][kk], BK12);
            wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][tile_n_off], BN12);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    __shared__ float Cs[BM12][BN12];
    wmma::store_matrix_sync(&Cs[warp_m * WM12][warp_n * WN12], acc, BN12, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM12 * BN12; idx += total_threads)
    {
        int const m = idx / BN12;
        int const n = idx % BN12;
        int const local_row = m / BM12_W;
        int const local_col = m % BM12_W;
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
void launch_wmma_large_spatial_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                            size_t C_in, size_t C_out, size_t H, size_t W,
                                            cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    constexpr int num_warps = (BM12 / WM12) * (BN12 / WN12);  // 8*2=16 warps
    dim3 const block(num_warps * 32);  // 512 threads
    dim3 const grid(
        (out_H + BM12_H - 1) / BM12_H,
        (out_W + BM12_W - 1) / BM12_W,
        (C_out + BN12 - 1) / BN12);

    wmma_large_spatial_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
