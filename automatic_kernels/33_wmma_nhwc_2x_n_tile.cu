#pragma once

#include <mma.h>
using namespace nvcuda;

// Experiment 33: 2x register tiling along N — each warp computes 2 B fragments.
//
// Base: exp 32 (transposed filter, BM=128, BN=64).
// Change: Each warp now handles 2 WN=16 tiles along N (32 oc per warp).
// Warp layout: 4 warps along M (each handles 32 rows = 2 WMMA tiles),
//              2 warps along N (each handles 32 oc = 2 WMMA tiles).
// This gives 4*2 = 8 warps. Each warp has 2*2 = 4 accumulators.
// Benefits: reuse A fragments across 2 B fragments.
// BM=128 (16x8), BN=64 stays the same.
// warps_m=4, warps_n=2.

constexpr int BM33_H = 16;
constexpr int BM33_W = 8;
constexpr int BM33 = BM33_H * BM33_W;  // 128
constexpr int BN33 = 64;
constexpr int BK33 = 16;
constexpr int WM33 = 16;
constexpr int WN33 = 16;
constexpr int WK33 = 8;

constexpr int BK33_PAD = BK33 + 1;
constexpr int BN33_PAD = BN33 + 1;

__device__ __forceinline__ void load_As_tile_33(
    float As[][BK33_PAD], float const* __restrict__ input,
    int tile_row, int tile_col, int fy, int fx, int ic_base,
    int C_in, int H, int W, int tid, int total_threads)
{
    constexpr int vec4_count = (BM33 * BK33) / 4;
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        int const elem = vi * 4;
        int const m = elem / BK33;
        int const k = elem % BK33;
        int const row = tile_row + (m / BM33_W) + fy;
        int const col = tile_col + (m % BM33_W) + fx;
        int const ic = ic_base + k;
        float4 val;
        if (row < H && col < W && ic + 3 < C_in) {
            val = *reinterpret_cast<float4 const*>(
                &input[row * W * C_in + col * C_in + ic]);
        } else {
            val.x = (row < H && col < W && ic < C_in) ?
                input[row * W * C_in + col * C_in + ic] : 0.0f;
            val.y = (row < H && col < W && ic + 1 < C_in) ?
                input[row * W * C_in + col * C_in + ic + 1] : 0.0f;
            val.z = (row < H && col < W && ic + 2 < C_in) ?
                input[row * W * C_in + col * C_in + ic + 2] : 0.0f;
            val.w = (row < H && col < W && ic + 3 < C_in) ?
                input[row * W * C_in + col * C_in + ic + 3] : 0.0f;
        }
        As[m][k]     = val.x;
        As[m][k + 1] = val.y;
        As[m][k + 2] = val.z;
        As[m][k + 3] = val.w;
    }
}

__device__ __forceinline__ void load_Bs_tile_33(
    float Bs[][BN33_PAD], float const* __restrict__ filter_t,
    int block_n, int fy, int fx, int ic_base,
    int C_in, int C_out, int tid, int total_threads)
{
    constexpr int vec4_count = (BK33 * BN33) / 4;
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        int const elem = vi * 4;
        int const k = elem / BN33;
        int const n = elem % BN33;
        int const ic = ic_base + k;
        int const oc = block_n + n;
        float4 val;
        if (ic < C_in && oc + 3 < C_out) {
            val = *reinterpret_cast<float4 const*>(
                &filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc]);
        } else {
            val.x = (ic < C_in && oc < C_out) ?
                filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc] : 0.0f;
            val.y = (ic < C_in && oc + 1 < C_out) ?
                filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc + 1] : 0.0f;
            val.z = (ic < C_in && oc + 2 < C_out) ?
                filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc + 2] : 0.0f;
            val.w = (ic < C_in && oc + 3 < C_out) ?
                filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc + 3] : 0.0f;
        }
        Bs[k][n]     = val.x;
        Bs[k][n + 1] = val.y;
        Bs[k][n + 2] = val.z;
        Bs[k][n + 3] = val.w;
    }
}

__launch_bounds__(256, 2)
__global__ void wmma_nhwc_2x_n_tile_conv2d_kernel(
    float* __restrict__ output,
    float const* __restrict__ input,
    float const* __restrict__ filter_t,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM33_H;
    int const tile_col = blockIdx.y * BM33_W;
    int const block_n = blockIdx.z * BN33;

    __shared__ float As[2][BM33][BK33_PAD];
    __shared__ float Bs[2][BK33][BN33_PAD];

    int const warp_id = threadIdx.x / 32;
    // 4 warps along M, 2 along N
    constexpr int warps_n = 2;
    int const warp_m = warp_id / warps_n;   // 0..3
    int const warp_n = warp_id % warps_n;   // 0..1

    // Each warp: 2 tiles along M (32 rows), 2 tiles along N (32 oc)
    // 4 accumulators: (m0,n0), (m1,n0), (m0,n1), (m1,n1)
    wmma::fragment<wmma::accumulator, WM33, WN33, WK33, float> acc00, acc10, acc01, acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;
    int const num_ic_iters = (C_in + BK33 - 1) / BK33;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            load_As_tile_33(As[0], input, tile_row, tile_col, fy, fx, 0,
                           C_in, H, W, tid, total_threads);
            load_Bs_tile_33(Bs[0], filter_t, block_n, fy, fx, 0,
                           C_in, C_out, tid, total_threads);
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK33;
                    load_As_tile_33(As[next_buf], input, tile_row, tile_col,
                                   fy, fx, next_ic, C_in, H, W, tid, total_threads);
                    load_Bs_tile_33(Bs[next_buf], filter_t, block_n, fy, fx,
                                   next_ic, C_in, C_out, tid, total_threads);
                }

                #pragma unroll
                for (int kk = 0; kk < BK33; kk += WK33)
                {
                    wmma::fragment<wmma::matrix_a, WM33, WN33, WK33,
                                   wmma::precision::tf32, wmma::row_major> a_frag0, a_frag1;
                    wmma::fragment<wmma::matrix_b, WM33, WN33, WK33,
                                   wmma::precision::tf32, wmma::row_major> b_frag0, b_frag1;

                    int const m_base = warp_m * 32;
                    int const n_base = warp_n * 32;
                    wmma::load_matrix_sync(a_frag0, &As[cur_buf][m_base][kk], BK33_PAD);
                    wmma::load_matrix_sync(a_frag1, &As[cur_buf][m_base + 16][kk], BK33_PAD);
                    wmma::load_matrix_sync(b_frag0, &Bs[cur_buf][kk][n_base], BN33_PAD);
                    wmma::load_matrix_sync(b_frag1, &Bs[cur_buf][kk][n_base + 16], BN33_PAD);

                    wmma::mma_sync(acc00, a_frag0, b_frag0, acc00);
                    wmma::mma_sync(acc10, a_frag1, b_frag0, acc10);
                    wmma::mma_sync(acc01, a_frag0, b_frag1, acc01);
                    wmma::mma_sync(acc11, a_frag1, b_frag1, acc11);
                }

                __syncthreads();
            }
        }
    }

    // Store — each warp writes 32x32 = 1024 elements
    // Total: 128*64 = 8192 elements, need 2 passes of 64*64 = 4096 each
    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);

    // Pass 1: warps with warp_m < 2 store (first 64 rows)
    if (warp_m < 2)
    {
        int const m_off = warp_m * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN33 + n_off], acc00, BN33, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN33 + n_off], acc10, BN33, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN33 + n_off + 16], acc01, BN33, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN33 + n_off + 16], acc11, BN33, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN33; idx += total_threads)
    {
        int const m = idx / BN33;
        int const n = idx & (BN33 - 1);
        int const out_row = tile_row + (m / BM33_W);
        int const out_col = tile_col + (m % BM33_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN33 + n];
    }
    __syncthreads();

    // Pass 2: warps with warp_m >= 2 store (second 64 rows)
    if (warp_m >= 2)
    {
        int const m_off = (warp_m - 2) * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN33 + n_off], acc00, BN33, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN33 + n_off], acc10, BN33, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN33 + n_off + 16], acc01, BN33, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN33 + n_off + 16], acc11, BN33, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN33; idx += total_threads)
    {
        int const m = idx / BN33;
        int const n = idx & (BN33 - 1);
        int const m_global = m + 64;
        int const out_row = tile_row + (m_global / BM33_W);
        int const out_col = tile_col + (m_global % BM33_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN33 + n];
    }
}

float profile_nhwc_2x_n_tile_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats = 100;
    constexpr int num_warmups = 10;
    int const out_H = H - 2;
    int const out_W = W - 2;

    float *d_input_hwc, *d_filter_nhwc, *d_filter_t, *d_output_hwc;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_t, C_out * C_in * 9 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    int const filter_elems = C_out * C_in * 9;
    transpose_filter_kernel<<<(filter_elems + 255) / 256, 256, 0, stream>>>(
        d_filter_t, d_filter_nhwc, C_in, C_out);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM33_H - 1) / BM33_H,
        (out_W + BM33_W - 1) / BM33_W,
        (C_out + BN33 - 1) / BN33);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_2x_n_tile_conv2d_kernel<<<grid, block, 0, s>>>(
            d_output_hwc, d_input_hwc, d_filter_t, C_in, C_out, H, W);
    };

    std::function<void(cudaStream_t)> fn = kernel_fn;
    float latency = measure_performance(fn, stream, num_repeats, num_warmups);

    CHECK_CUDA_ERROR(cudaFree(d_input_hwc));
    CHECK_CUDA_ERROR(cudaFree(d_filter_nhwc));
    CHECK_CUDA_ERROR(cudaFree(d_filter_t));
    CHECK_CUDA_ERROR(cudaFree(d_output_hwc));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

template <typename T>
void launch_wmma_nhwc_2x_n_tile_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                              size_t C_in, size_t C_out, size_t H, size_t W,
                                              cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    T *d_input_hwc, *d_filter_nhwc, *d_filter_t, *d_output_hwc;
    cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(T));
    cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(T));
    cudaMalloc(&d_filter_t, C_out * C_in * 9 * sizeof(T));
    cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(T));

    int const n1 = C_in * H * W;
    chw_to_hwc_kernel<<<(n1+255)/256, 256, 0, stream>>>(d_input_hwc, d_input, C_in, H, W);
    int const n2 = C_out * C_in * 9;
    reorder_filter_nhwc_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_nhwc, d_filter, C_out, C_in);
    transpose_filter_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_t, d_filter_nhwc, C_in, C_out);

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM33_H - 1) / BM33_H,
        (out_W + BM33_W - 1) / BM33_W,
        (C_out + BN33 - 1) / BN33);
    wmma_nhwc_2x_n_tile_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_hwc, d_filter_t, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_filter_t);
    cudaFree(d_output_hwc);
}
