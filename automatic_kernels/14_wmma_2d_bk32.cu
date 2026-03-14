#pragma once

#include <mma.h>
using namespace nvcuda;

// Best of experiment 10 (2D spatial grid + double buffer) with BK=32.
// More K-dim work per iteration = fewer main loop iterations = less sync overhead.
// Uses aliased SMEM for AB buffers and C buffer.

constexpr int BM14_H = 8;
constexpr int BM14_W = 8;
constexpr int BM14 = BM14_H * BM14_W;  // 64
constexpr int BN14 = 64;
constexpr int BK14 = 32;

constexpr int WM14 = 16;
constexpr int WN14 = 16;
constexpr int WK14 = 8;

__global__ void wmma_2d_bk32_conv2d_kernel(
    float* output, float const* input, float const* filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const K_gemm = C_in * 9;

    int const tile_row = blockIdx.x * BM14_H;
    int const tile_col = blockIdx.y * BM14_W;
    int const block_n = blockIdx.z * BN14;

    // Double buffers: As[2][64][32] + Bs[2][32][64]
    // = 2*2048 + 2*2048 = 8192 floats = 32KB
    // Cs[64][64] = 4096 floats = 16KB — reuses same buffer after loop
    extern __shared__ char smem14_raw[];
    float* smem14 = reinterpret_cast<float*>(smem14_raw);

    // As at offsets [buf * BM14 * BK14]
    // Bs at offset [2 * BM14 * BK14 + buf * BK14 * BN14]
    auto As = [&](int buf, int m, int k) -> float& {
        return smem14[buf * BM14 * BK14 + m * BK14 + k];
    };
    auto Bs = [&](int buf, int k, int n) -> float& {
        return smem14[2 * BM14 * BK14 + buf * BK14 * BN14 + k * BN14 + n];
    };
    // Cs reuses from offset 0
    auto Cs = [&](int m, int n) -> float& {
        return smem14[m * BN14 + n];
    };

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN14 / WN14;  // 4
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM14, WN14, WK14, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;  // 512

    auto load_tile = [&](int buf, int k_start) {
        for (int idx = tid; idx < BM14 * BK14; idx += total_threads)
        {
            int const m = idx / BK14;
            int const kk = idx % BK14;
            int const local_row = m / BM14_W;
            int const local_col = m % BM14_W;
            int const out_row = tile_row + local_row;
            int const out_col = tile_col + local_col;
            int const gk = k_start + kk;

            if (out_row < out_H && out_col < out_W && gk < K_gemm)
            {
                int const ic = gk / 9;
                int const rem = gk % 9;
                int const fy = rem / 3;
                int const fx = rem % 3;
                As(buf, m, kk) = __ldg(&input[ic * H * W + (out_row + fy) * W + (out_col + fx)]);
            }
            else
            {
                As(buf, m, kk) = 0.0f;
            }
        }
        for (int idx = tid; idx < BK14 * BN14; idx += total_threads)
        {
            int const kk = idx / BN14;
            int const n = idx % BN14;
            int const gk = k_start + kk;
            int const gn = block_n + n;
            Bs(buf, kk, n) = (gk < K_gemm && gn < C_out) ?
                __ldg(&filter[gn * K_gemm + gk]) : 0.0f;
        }
    };

    load_tile(0, 0);
    __syncthreads();

    int const num_k_iters = (K_gemm + BK14 - 1) / BK14;

    for (int ki = 0; ki < num_k_iters; ++ki)
    {
        int const cur_buf = ki % 2;
        int const next_buf = 1 - cur_buf;

        if (ki + 1 < num_k_iters)
            load_tile(next_buf, (ki + 1) * BK14);

        int const tile_m_off = warp_m * WM14;
        int const tile_n_off = warp_n * WN14;

        #pragma unroll
        for (int kk = 0; kk < BK14; kk += WK14)
        {
            wmma::fragment<wmma::matrix_a, WM14, WN14, WK14,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM14, WN14, WK14,
                           wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As(cur_buf, tile_m_off, kk), BK14);
            wmma::load_matrix_sync(b_frag, &Bs(cur_buf, kk, tile_n_off), BN14);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    // Store to Cs (aliased with As/Bs, safe since loop is done)
    int const tile_m_off = warp_m * WM14;
    int const tile_n_off = warp_n * WN14;
    wmma::store_matrix_sync(&Cs(tile_m_off, tile_n_off), acc, BN14, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM14 * BN14; idx += total_threads)
    {
        int const m = idx / BN14;
        int const n = idx % BN14;
        int const local_row = m / BM14_W;
        int const local_col = m % BM14_W;
        int const out_row = tile_row + local_row;
        int const out_col = tile_col + local_col;
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[oc * out_H * out_W + out_row * out_W + out_col] = Cs(m, n);
        }
    }
}

template <typename T>
void launch_wmma_2d_bk32_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                      size_t C_in, size_t C_out, size_t H, size_t W,
                                      cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    constexpr int num_warps = (BM14 / WM14) * (BN14 / WN14);  // 4*4=16
    dim3 const block(num_warps * 32);  // 512 threads
    dim3 const grid(
        (out_H + BM14_H - 1) / BM14_H,
        (out_W + BM14_W - 1) / BM14_W,
        (C_out + BN14 - 1) / BN14);

    // SMEM: max of AB buffers and C buffer
    // AB = 2*64*32 + 2*32*64 = 8192 floats = 32768 bytes
    // C = 64*64 = 4096 floats = 16384 bytes
    size_t const smem_size = (2 * BM14 * BK14 + 2 * BK14 * BN14) * sizeof(float);

    wmma_2d_bk32_conv2d_kernel<<<grid, block, smem_size, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
