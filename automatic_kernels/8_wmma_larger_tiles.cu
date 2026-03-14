#pragma once

#include <mma.h>
using namespace nvcuda;

// Larger tile WMMA with BK=32 and BM=128 to reduce main loop iterations
// and improve arithmetic intensity.

constexpr int BM8 = 128;
constexpr int BN8 = 64;
constexpr int BK8 = 32;
constexpr int WM8 = 16;
constexpr int WN8 = 16;
constexpr int WK8 = 8;

__device__ __forceinline__ float im2col_load8(
    float const* input, int n_idx, int k_idx,
    int out_W, int C_in, int H, int W)
{
    int const out_row = n_idx / out_W;
    int const out_col = n_idx % out_W;
    int const ic = k_idx / 9;
    int const rem = k_idx % 9;
    int const fy = rem / 3;
    int const fx = rem % 3;
    return input[ic * H * W + (out_row + fy) * W + (out_col + fx)];
}

__global__ void wmma_larger_tiles_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;
    int const K_gemm = C_in * 9;

    int const block_m = blockIdx.x * BM8;
    int const block_n = blockIdx.y * BN8;

    // Use a union to alias AB tiles with C tile (used at different times)
    // As + Bs = 128*32 + 32*64 = 4096 + 2048 = 6144 floats = 24576 bytes
    // Cs = 128*64 = 8192 floats = 32768 bytes
    extern __shared__ char smem8_raw[];
    float* smem8 = reinterpret_cast<float*>(smem8_raw);
    // During main loop: As at offset 0, Bs at offset BM8*BK8
    float* As_flat = smem8;
    float* Bs_flat = smem8 + BM8 * BK8;
    // After main loop: Cs reuses the same buffer from offset 0
    float* Cs_flat = smem8;

    // Warp arrangement: 8x4 warps (BM8/WM8=8, BN8/WN8=4)
    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN8 / WN8;  // 4
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM8, WN8, WK8, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    for (int k = 0; k < K_gemm; k += BK8)
    {
        // Load A[BM8][BK8]
        for (int idx = tid; idx < BM8 * BK8; idx += total_threads)
        {
            int const m = idx / BK8;
            int const kk = idx % BK8;
            int const gm = block_m + m;
            int const gk = k + kk;
            As_flat[m * BK8 + kk] = (gm < N && gk < K_gemm) ?
                im2col_load8(input, gm, gk, out_W, C_in, H, W) : 0.0f;
        }

        // Load B[BK8][BN8]
        for (int idx = tid; idx < BK8 * BN8; idx += total_threads)
        {
            int const kk = idx / BN8;
            int const n = idx % BN8;
            int const gk = k + kk;
            int const gn = block_n + n;
            Bs_flat[kk * BN8 + n] = (gk < K_gemm && gn < C_out) ?
                filter[gn * K_gemm + gk] : 0.0f;
        }

        __syncthreads();

        int const tile_m_off = warp_m * WM8;
        int const tile_n_off = warp_n * WN8;

        #pragma unroll
        for (int kk = 0; kk < BK8; kk += WK8)
        {
            wmma::fragment<wmma::matrix_a, WM8, WN8, WK8,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM8, WN8, WK8,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As_flat[tile_m_off * BK8 + kk], BK8);
            wmma::load_matrix_sync(b_frag, &Bs_flat[kk * BN8 + tile_n_off], BN8);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    // Store results — reuse smem as Cs
    wmma::store_matrix_sync(&Cs_flat[(warp_m * WM8) * BN8 + warp_n * WN8], acc, BN8, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM8 * BN8; idx += total_threads)
    {
        int const m = idx / BN8;
        int const n = idx % BN8;
        int const gm = block_m + m;
        int const gn = block_n + n;
        if (gm < N && gn < C_out)
        {
            output[gn * N + gm] = Cs_flat[m * BN8 + n];
        }
    }
}

template <typename T>
void launch_wmma_larger_tiles_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                           size_t C_in, size_t C_out, size_t H, size_t W,
                                           cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const N = out_H * out_W;

    constexpr int num_warps = (BM8 / WM8) * (BN8 / WN8);  // 8*4 = 32 warps = 1024 threads
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (N + BM8 - 1) / BM8,
        (C_out + BN8 - 1) / BN8);

    // Max of (As+Bs) and Cs: max(128*32+32*64, 128*64) = max(6144, 8192) = 8192 floats
    size_t const smem_size = BM8 * BN8 * sizeof(float);  // 32768 bytes
    wmma_larger_tiles_conv2d_kernel<<<grid, block, smem_size, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
