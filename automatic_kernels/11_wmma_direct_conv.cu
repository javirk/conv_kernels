#pragma once

#include <mma.h>
using namespace nvcuda;

// Direct convolution with WMMA: instead of implicit GEMM, tile spatially and
// iterate over (ic, fy, fx). For each input channel, load a spatial tile of
// input into SMEM, then for each of the 9 filter positions, form the A matrix
// by reading shifted positions from SMEM, multiply with filter weights in B.

// Block processes BM11_H x BM11_W output spatial positions and BN11 output channels
constexpr int BM11_H = 8;
constexpr int BM11_W = 8;
constexpr int BM11 = BM11_H * BM11_W;  // 64
constexpr int BN11 = 64;

// SMEM tile for input: need (BM11_H+2) x (BM11_W+2) per input channel
constexpr int SMEM_H11 = BM11_H + 2;
constexpr int SMEM_W11 = BM11_W + 2;

constexpr int WM11 = 16;
constexpr int WN11 = 16;
constexpr int WK11 = 8;

__global__ void wmma_direct_conv_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM11_H;
    int const tile_col = blockIdx.y * BM11_W;
    int const block_n = blockIdx.z * BN11;

    // SMEM for input tile and filter tile
    __shared__ float input_smem[SMEM_H11][SMEM_W11];
    // Filter: 9 weights per (oc, ic), we process BN11 output channels
    // For each filter position, we need a row of BN11 weights
    // But WMMA needs K=8, so we accumulate 8 input channels at a time
    // Actually let's think differently...

    // The GEMM is: for each (fy, fx), A[m, ic] * B[ic, n] where
    // A[m, ic] = input[ic, out_row_m + fy, out_col_m + fx]
    // B[ic, n] = filter[n, ic, fy, fx]
    // Sum over ic and (fy, fx)

    // So we can iterate over (fy, fx) and accumulate partial GEMMs of size [BM11, C_in] x [C_in, BN11]
    // For each (fy, fx), chunk C_in into groups of BK

    constexpr int BK11 = 8;  // must be WK11

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN11 / WN11;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM11, WN11, WK11, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    __shared__ float As[BM11][BK11];
    __shared__ float Bs[BK11][BN11];

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            for (int ic_start = 0; ic_start < C_in; ic_start += BK11)
            {
                // Load A[BM11][BK11]: A[m, k] = input[(ic_start+k), out_row_m + fy, out_col_m + fx]
                for (int idx = tid; idx < BM11 * BK11; idx += total_threads)
                {
                    int const m = idx / BK11;
                    int const k = idx % BK11;
                    int const local_row = m / BM11_W;
                    int const local_col = m % BM11_W;
                    int const row = tile_row + local_row + fy;
                    int const col = tile_col + local_col + fx;
                    int const ic = ic_start + k;
                    As[m][k] = (row < H && col < W && ic < C_in) ?
                        input[ic * H * W + row * W + col] : 0.0f;
                }

                // Load B[BK11][BN11]: B[k, n] = filter[(block_n+n), (ic_start+k), fy, fx]
                for (int idx = tid; idx < BK11 * BN11; idx += total_threads)
                {
                    int const k = idx / BN11;
                    int const n = idx % BN11;
                    int const ic = ic_start + k;
                    int const oc = block_n + n;
                    Bs[k][n] = (ic < C_in && oc < C_out) ?
                        filter[oc * C_in * 9 + ic * 9 + fy * 3 + fx] : 0.0f;
                }

                __syncthreads();

                // WMMA: single WK11=8 step (BK11 == WK11)
                wmma::fragment<wmma::matrix_a, WM11, WN11, WK11,
                               wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM11, WN11, WK11,
                               wmma::precision::tf32, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, &As[warp_m * WM11][0], BK11);
                wmma::load_matrix_sync(b_frag, &Bs[0][warp_n * WN11], BN11);
                wmma::mma_sync(acc, a_frag, b_frag, acc);

                __syncthreads();
            }
        }
    }

    // Store results
    __shared__ float Cs[BM11][BN11];
    wmma::store_matrix_sync(&Cs[warp_m * WM11][warp_n * WN11], acc, BN11, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM11 * BN11; idx += total_threads)
    {
        int const m = idx / BN11;
        int const n = idx % BN11;
        int const local_row = m / BM11_W;
        int const local_col = m % BM11_W;
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
void launch_wmma_direct_conv_3x3(T* d_output, T const* d_input, T const* d_filter,
                                   size_t C_in, size_t C_out, size_t H, size_t W,
                                   cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    constexpr int num_warps = (BM11 / WM11) * (BN11 / WN11);  // 4*4=16
    dim3 const block(num_warps * 32);  // 512 threads
    dim3 const grid(
        (out_H + BM11_H - 1) / BM11_H,
        (out_W + BM11_W - 1) / BM11_W,
        (C_out + BN11 - 1) / BN11);

    wmma_direct_conv_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
