#pragma once

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Kernel 40: FP16 WMMA BK=64, single ic tile. Best for C_in >= 64.

constexpr int BM40_H = 16;
constexpr int BM40_W = 8;
constexpr int BM40 = BM40_H * BM40_W;
constexpr int BN40 = 64;
constexpr int BK40 = 64;
constexpr int WM40 = 16;
constexpr int WN40 = 16;
constexpr int WK40 = 16;

constexpr int BK40_PAD = BK40 + 8;
constexpr int BN40_PAD = BN40 + 8;

__device__ __forceinline__ void load_As_40(
    __half As[][BK40_PAD], __half const* __restrict__ input,
    int tile_row, int tile_col, int fy, int fx,
    int C_in, int H, int W, int tid, int total_threads)
{
    constexpr int vec_count = (BM40 * BK40) / 8;
    for (int vi = tid; vi < vec_count; vi += total_threads)
    {
        int const elem = vi * 8;
        int const m = elem / BK40;
        int const k = elem % BK40;
        int const row = tile_row + (m / BM40_W) + fy;
        int const col = tile_col + (m % BM40_W) + fx;
        float4 val;
        if (row < H && col < W && k + 7 < C_in) {
            val = *reinterpret_cast<float4 const*>(
                &input[row * W * C_in + col * C_in + k]);
        } else {
            __half tmp[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                tmp[i] = (row < H && col < W && k + i < C_in) ?
                    input[row * W * C_in + col * C_in + k + i] : __float2half(0.0f);
            }
            val = *reinterpret_cast<float4*>(tmp);
        }
        *reinterpret_cast<float4*>(&As[m][k]) = val;
    }
}

__device__ __forceinline__ void load_Bs_40(
    __half Bs[][BN40_PAD], __half const* __restrict__ filter_t,
    int block_n, int fy, int fx,
    int C_in, int C_out, int tid, int total_threads)
{
    constexpr int vec_count = (BK40 * BN40) / 8;
    for (int vi = tid; vi < vec_count; vi += total_threads)
    {
        int const elem = vi * 8;
        int const k = elem / BN40;
        int const n = elem % BN40;
        int const oc = block_n + n;
        float4 val;
        if (k < C_in && oc + 7 < C_out) {
            val = *reinterpret_cast<float4 const*>(
                &filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + k * C_out + oc]);
        } else {
            __half tmp[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                tmp[i] = (k < C_in && oc + i < C_out) ?
                    filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + k * C_out + oc + i] : __float2half(0.0f);
            }
            val = *reinterpret_cast<float4*>(tmp);
        }
        *reinterpret_cast<float4*>(&Bs[k][n]) = val;
    }
}

__launch_bounds__(256, 2)
__global__ void wmma_fp16_bk64_conv2d_kernel(
    float* __restrict__ output,
    __half const* __restrict__ input,
    __half const* __restrict__ filter_t,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM40_H;
    int const tile_col = blockIdx.y * BM40_W;
    int const block_n = blockIdx.z * BN40;

    __shared__ __half As[BM40][BK40_PAD];
    __shared__ __half Bs[BK40][BN40_PAD];

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = 2;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM40, WN40, WK40, float> acc00, acc10, acc01, acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            load_As_40(As, input, tile_row, tile_col, fy, fx,
                       C_in, H, W, tid, total_threads);
            load_Bs_40(Bs, filter_t, block_n, fy, fx,
                       C_in, C_out, tid, total_threads);
            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK40; kk += WK40)
            {
                wmma::fragment<wmma::matrix_a, WM40, WN40, WK40,
                               __half, wmma::row_major> a_frag0, a_frag1;
                wmma::fragment<wmma::matrix_b, WM40, WN40, WK40,
                               __half, wmma::row_major> b_frag0, b_frag1;

                int const m_base = warp_m * 32;
                int const n_base = warp_n * 32;
                wmma::load_matrix_sync(a_frag0, &As[m_base][kk], BK40_PAD);
                wmma::load_matrix_sync(a_frag1, &As[m_base + 16][kk], BK40_PAD);
                wmma::load_matrix_sync(b_frag0, &Bs[kk][n_base], BN40_PAD);
                wmma::load_matrix_sync(b_frag1, &Bs[kk][n_base + 16], BN40_PAD);

                wmma::mma_sync(acc00, a_frag0, b_frag0, acc00);
                wmma::mma_sync(acc10, a_frag1, b_frag0, acc10);
                wmma::mma_sync(acc01, a_frag0, b_frag1, acc01);
                wmma::mma_sync(acc11, a_frag1, b_frag1, acc11);
            }

            __syncthreads();
        }
    }

    float* Cs = reinterpret_cast<float*>(&As[0][0]);

    if (warp_m < 2)
    {
        int const m_off = warp_m * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN40 + n_off], acc00, BN40, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN40 + n_off], acc10, BN40, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN40 + n_off + 16], acc01, BN40, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN40 + n_off + 16], acc11, BN40, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN40; idx += total_threads)
    {
        int const m = idx / BN40;
        int const n = idx & (BN40 - 1);
        int const out_row = tile_row + (m / BM40_W);
        int const out_col = tile_col + (m % BM40_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN40 + n];
    }
    __syncthreads();

    if (warp_m >= 2)
    {
        int const m_off = (warp_m - 2) * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN40 + n_off], acc00, BN40, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN40 + n_off], acc10, BN40, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN40 + n_off + 16], acc01, BN40, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN40 + n_off + 16], acc11, BN40, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN40; idx += total_threads)
    {
        int const m = idx / BN40;
        int const n = idx & (BN40 - 1);
        int const m_global = m + 64;
        int const out_row = tile_row + (m_global / BM40_W);
        int const out_col = tile_col + (m_global % BM40_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN40 + n];
    }
}

inline void launch_conv2d_40(float* d_output, float const* d_input, float const* d_filter,
                              size_t C_in, size_t C_out, size_t H, size_t W,
                              cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    float *d_input_hwc, *d_filter_nhwc, *d_filter_t;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_t, C_out * C_in * 9 * sizeof(float)));

    __half *d_input_h, *d_filter_t_h;
    float *d_output_hwc;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_h, C_in * H * W * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_t_h, C_out * C_in * 9 * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(float)));

    int const n1 = C_in * H * W;
    int const n2 = C_out * C_in * 9;
    chw_to_hwc_kernel<<<(n1+255)/256, 256, 0, stream>>>(d_input_hwc, d_input, C_in, H, W);
    reorder_filter_nhwc_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_nhwc, d_filter, C_out, C_in);
    transpose_filter_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_t, d_filter_nhwc, C_in, C_out);
    float_to_half_kernel<<<(n1+255)/256, 256, 0, stream>>>(d_input_h, d_input_hwc, n1);
    float_to_half_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_t_h, d_filter_t, n2);

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM40_H - 1) / BM40_H,
        (out_W + BM40_W - 1) / BM40_W,
        (C_out + BN40 - 1) / BN40);
    wmma_fp16_bk64_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_h, d_filter_t_h, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_filter_t);
    cudaFree(d_input_h);
    cudaFree(d_filter_t_h);
    cudaFree(d_output_hwc);
}
