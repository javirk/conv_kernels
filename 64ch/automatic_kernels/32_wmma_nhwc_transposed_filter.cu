#pragma once

#include <mma.h>
using namespace nvcuda;

// Experiment 32: Pre-transpose filter to [fy, fx, ic, oc] for coalesced Bs loads.
//
// Base: exp 29 (BM=128, vec4 loads for As, vec4 along K for Bs).
// Problem: Bs loads access filter[oc, fy, fx, ic] where consecutive threads
// load different oc values with stride 9*C_in → poor coalescing.
// Fix: Pre-transpose filter to [fy, fx, ic, oc] layout on device.
// Now Bs loading: filter_t[fy*3*C_in*C_out + fx*C_in*C_out + ic*C_out + oc]
// For fixed (fy,fx,ic), consecutive oc values are contiguous → float4 loads!

// Device kernel to transpose filter from [oc, fy, fx, ic] to [fy, fx, ic, oc]
__global__ void transpose_filter_kernel(
    float* __restrict__ filter_t,     // [3, 3, C_in, C_out]
    float const* __restrict__ filter, // [C_out, 3, 3, C_in]
    int C_in, int C_out)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int const total = 9 * C_in * C_out;
    if (idx >= total) return;

    // Decode idx as [fy, fx, ic, oc]
    int const oc = idx % C_out;
    int const ic = (idx / C_out) % C_in;
    int const fx = (idx / (C_out * C_in)) % 3;
    int const fy = idx / (C_out * C_in * 3);

    filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc] =
        filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic];
}

constexpr int BM32_H = 16;
constexpr int BM32_W = 8;
constexpr int BM32 = BM32_H * BM32_W;  // 128
constexpr int BN32 = 64;
constexpr int BK32 = 16;
constexpr int WM32 = 16;
constexpr int WN32 = 16;
constexpr int WK32 = 8;

constexpr int BK32_PAD = BK32 + 1;  // 17
constexpr int BN32_PAD = BN32 + 1;  // 65

__device__ __forceinline__ void load_As_tile_32(
    float As[][BK32_PAD], float const* __restrict__ input,
    int tile_row, int tile_col, int fy, int fx, int ic_base,
    int C_in, int H, int W, int tid, int total_threads)
{
    constexpr int vec4_count = (BM32 * BK32) / 4;
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        int const elem = vi * 4;
        int const m = elem / BK32;
        int const k = elem % BK32;
        int const row = tile_row + (m / BM32_W) + fy;
        int const col = tile_col + (m % BM32_W) + fx;
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

// Load Bs from transposed filter [fy, fx, ic, oc] — float4 along N (oc) dimension
__device__ __forceinline__ void load_Bs_tile_32(
    float Bs[][BN32_PAD], float const* __restrict__ filter_t,
    int block_n, int fy, int fx, int ic_base,
    int C_in, int C_out, int tid, int total_threads)
{
    // Bs: [BK32][BN32] = [16][64]. Load with float4 along N (oc).
    // filter_t[fy, fx, ic, oc] = filter_t[fy*3*C_in*C_out + fx*C_in*C_out + ic*C_out + oc]
    // Consecutive oc values are contiguous → perfect for float4!
    constexpr int vec4_count = (BK32 * BN32) / 4;  // 256
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        int const elem = vi * 4;
        int const k = elem / BN32;
        int const n = elem % BN32;
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
__global__ void wmma_nhwc_transposed_filter_conv2d_kernel(
    float* __restrict__ output,
    float const* __restrict__ input,
    float const* __restrict__ filter_t, // [3, 3, C_in, C_out] transposed
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM32_H;
    int const tile_col = blockIdx.y * BM32_W;
    int const block_n = blockIdx.z * BN32;

    __shared__ float As[2][BM32][BK32_PAD];
    __shared__ float Bs[2][BK32][BN32_PAD];

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = BN32 / WN32;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM32, WN32, WK32, float> acc0, acc1, acc2, acc3;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);
    wmma::fill_fragment(acc3, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;
    int const num_ic_iters = (C_in + BK32 - 1) / BK32;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            load_As_tile_32(As[0], input, tile_row, tile_col, fy, fx, 0,
                           C_in, H, W, tid, total_threads);
            load_Bs_tile_32(Bs[0], filter_t, block_n, fy, fx, 0,
                           C_in, C_out, tid, total_threads);
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK32;
                    load_As_tile_32(As[next_buf], input, tile_row, tile_col,
                                   fy, fx, next_ic, C_in, H, W, tid, total_threads);
                    load_Bs_tile_32(Bs[next_buf], filter_t, block_n, fy, fx,
                                   next_ic, C_in, C_out, tid, total_threads);
                }

                #pragma unroll
                for (int kk = 0; kk < BK32; kk += WK32)
                {
                    wmma::fragment<wmma::matrix_a, WM32, WN32, WK32,
                                   wmma::precision::tf32, wmma::row_major> a_frag0, a_frag1, a_frag2, a_frag3;
                    wmma::fragment<wmma::matrix_b, WM32, WN32, WK32,
                                   wmma::precision::tf32, wmma::row_major> b_frag;

                    int const m_base = warp_m * 64;
                    wmma::load_matrix_sync(a_frag0, &As[cur_buf][m_base][kk], BK32_PAD);
                    wmma::load_matrix_sync(a_frag1, &As[cur_buf][m_base + 16][kk], BK32_PAD);
                    wmma::load_matrix_sync(a_frag2, &As[cur_buf][m_base + 32][kk], BK32_PAD);
                    wmma::load_matrix_sync(a_frag3, &As[cur_buf][m_base + 48][kk], BK32_PAD);
                    wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][warp_n * WN32], BN32_PAD);
                    wmma::mma_sync(acc0, a_frag0, b_frag, acc0);
                    wmma::mma_sync(acc1, a_frag1, b_frag, acc1);
                    wmma::mma_sync(acc2, a_frag2, b_frag, acc2);
                    wmma::mma_sync(acc3, a_frag3, b_frag, acc3);
                }

                __syncthreads();
            }
        }
    }

    // Two-pass store
    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);

    if (warp_m == 0)
    {
        wmma::store_matrix_sync(&Cs[0 * BN32 + warp_n * WN32], acc0, BN32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[16 * BN32 + warp_n * WN32], acc1, BN32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[32 * BN32 + warp_n * WN32], acc2, BN32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[48 * BN32 + warp_n * WN32], acc3, BN32, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN32; idx += total_threads)
    {
        int const m = idx / BN32;
        int const n = idx & (BN32 - 1);
        int const out_row = tile_row + (m / BM32_W);
        int const out_col = tile_col + (m % BM32_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN32 + n];
    }
    __syncthreads();

    if (warp_m == 1)
    {
        wmma::store_matrix_sync(&Cs[0 * BN32 + warp_n * WN32], acc0, BN32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[16 * BN32 + warp_n * WN32], acc1, BN32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[32 * BN32 + warp_n * WN32], acc2, BN32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[48 * BN32 + warp_n * WN32], acc3, BN32, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN32; idx += total_threads)
    {
        int const m = idx / BN32;
        int const n = idx & (BN32 - 1);
        int const m_global = m + 64;
        int const out_row = tile_row + (m_global / BM32_W);
        int const out_col = tile_col + (m_global % BM32_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN32 + n];
    }
}

float profile_nhwc_transposed_filter_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
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

    // Pre-transpose filter (one-time cost, not measured)
    int const filter_elems = C_out * C_in * 9;
    transpose_filter_kernel<<<(filter_elems + 255) / 256, 256, 0, stream>>>(
        d_filter_t, d_filter_nhwc, C_in, C_out);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM32_H - 1) / BM32_H,
        (out_W + BM32_W - 1) / BM32_W,
        (C_out + BN32 - 1) / BN32);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_transposed_filter_conv2d_kernel<<<grid, block, 0, s>>>(
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
void launch_wmma_nhwc_transposed_filter_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
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
        (out_H + BM32_H - 1) / BM32_H,
        (out_W + BM32_W - 1) / BM32_W,
        (C_out + BN32 - 1) / BN32);
    wmma_nhwc_transposed_filter_conv2d_kernel<<<grid, block, 0, stream>>>(
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
