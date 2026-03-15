#pragma once

#include <mma.h>
using namespace nvcuda;

// Experiment 28: Vec4 for filter loads along K (ic) dimension.
//
// Base: exp 26 (vec4 loads for As, scalar for Bs).
// Problem: Bs loading does 4 scalar loads per float4 with stride 9*C_in between
// consecutive oc values — terrible for memory coalescing.
// Fix: Restructure Bs loading to use float4 along the K (ic) dimension where
// filter[oc, fy, fx, ic..ic+3] IS contiguous. Each thread loads one (k,n) group
// where k advances by 4 (float4 along ic).

constexpr int BM28_H = 8;
constexpr int BM28_W = 8;
constexpr int BM28 = BM28_H * BM28_W;  // 64
constexpr int BN28 = 64;
constexpr int BK28 = 16;
constexpr int WM28 = 16;
constexpr int WN28 = 16;
constexpr int WK28 = 8;

constexpr int BK28_PAD = BK28 + 1;  // 17
constexpr int BN28_PAD = BN28 + 1;  // 65

// Helper to load As tile with float4 along K (ic) dimension
__device__ __forceinline__ void load_As_tile_28(
    float As[][BK28_PAD], float const* __restrict__ input,
    int tile_row, int tile_col, int fy, int fx, int ic_base,
    int C_in, int H, int W, int tid, int total_threads)
{
    // As: 64 rows x 16 cols = 1024 elements = 256 float4s
    constexpr int vec4_count = (BM28 * BK28) / 4;  // 256
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        int const elem = vi * 4;
        int const m = elem / BK28;
        int const k = elem % BK28;
        int const row = tile_row + (m >> 3) + fy;
        int const col = tile_col + (m & 7) + fx;
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

// Helper to load Bs tile with float4 along K (ic) dimension
__device__ __forceinline__ void load_Bs_tile_28(
    float Bs[][BN28_PAD], float const* __restrict__ filter,
    int block_n, int fy, int fx, int ic_base,
    int C_in, int C_out, int tid, int total_threads)
{
    // Bs: 16 rows x 64 cols = 1024 elements
    // Layout: Bs[k][n] where k=ic_local, n=oc_local
    // Filter: [oc, fy, fx, ic] → filter[oc*9*C_in + fy*3*C_in + fx*C_in + ic]
    // For fixed oc, consecutive ic values are contiguous → float4 along k!
    // We iterate over (n, k/4) pairs: each thread loads float4 of 4 consecutive ic values
    constexpr int vec4_count = (BK28 * BN28) / 4;  // 256
    for (int vi = tid; vi < vec4_count; vi += total_threads)
    {
        // Map vi to (n, k_base) where we load 4 consecutive k values for a given n
        int const n = vi / (BK28 / 4);   // which oc (0..63)
        int const k4 = vi % (BK28 / 4);  // which k group (0..3)
        int const k = k4 * 4;
        int const oc = block_n + n;
        int const ic = ic_base + k;
        float4 val;
        if (ic + 3 < C_in && oc < C_out) {
            val = *reinterpret_cast<float4 const*>(
                &filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic]);
        } else {
            val.x = (ic < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
            val.y = (ic + 1 < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic + 1] : 0.0f;
            val.z = (ic + 2 < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic + 2] : 0.0f;
            val.w = (ic + 3 < C_in && oc < C_out) ?
                filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic + 3] : 0.0f;
        }
        Bs[k][n]     = val.x;
        Bs[k + 1][n] = val.y;
        Bs[k + 2][n] = val.z;
        Bs[k + 3][n] = val.w;
    }
}

__launch_bounds__(256, 2)
__global__ void wmma_nhwc_vec4_filter_conv2d_kernel(
    float* __restrict__ output,       // [out_H, out_W, C_out]
    float const* __restrict__ input,  // [H, W, C_in]
    float const* __restrict__ filter, // [C_out, 3, 3, C_in]
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM28_H;
    int const tile_col = blockIdx.y * BM28_W;
    int const block_n = blockIdx.z * BN28;

    __shared__ float As[2][BM28][BK28_PAD];
    __shared__ float Bs[2][BK28][BN28_PAD];

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = BN28 / WN28;  // 4
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM28, WN28, WK28, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;
    int const num_ic_iters = (C_in + BK28 - 1) / BK28;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            load_As_tile_28(As[0], input, tile_row, tile_col, fy, fx, 0,
                           C_in, H, W, tid, total_threads);
            load_Bs_tile_28(Bs[0], filter, block_n, fy, fx, 0,
                           C_in, C_out, tid, total_threads);
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK28;
                    load_As_tile_28(As[next_buf], input, tile_row, tile_col,
                                   fy, fx, next_ic, C_in, H, W, tid, total_threads);
                    load_Bs_tile_28(Bs[next_buf], filter, block_n, fy, fx,
                                   next_ic, C_in, C_out, tid, total_threads);
                }

                #pragma unroll
                for (int kk = 0; kk < BK28; kk += WK28)
                {
                    wmma::fragment<wmma::matrix_a, WM28, WN28, WK28,
                                   wmma::precision::tf32, wmma::row_major> a_frag0, a_frag1;
                    wmma::fragment<wmma::matrix_b, WM28, WN28, WK28,
                                   wmma::precision::tf32, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag0, &As[cur_buf][warp_m * 32][kk], BK28_PAD);
                    wmma::load_matrix_sync(a_frag1, &As[cur_buf][warp_m * 32 + 16][kk], BK28_PAD);
                    wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][warp_n * WN28], BN28_PAD);
                    wmma::mma_sync(acc0, a_frag0, b_frag, acc0);
                    wmma::mma_sync(acc1, a_frag1, b_frag, acc1);
                }

                __syncthreads();
            }
        }
    }

    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);
    wmma::store_matrix_sync(&Cs[(warp_m * 32) * BN28 + warp_n * WN28], acc0, BN28, wmma::mem_row_major);
    wmma::store_matrix_sync(&Cs[(warp_m * 32 + 16) * BN28 + warp_n * WN28], acc1, BN28, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM28 * BN28; idx += total_threads)
    {
        int const m = idx / BN28;
        int const n = idx & (BN28 - 1);
        int const out_row = tile_row + (m >> 3);
        int const out_col = tile_col + (m & 7);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN28 + n];
        }
    }
}

float profile_nhwc_vec4_filter_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats = 100;
    constexpr int num_warmups = 10;
    int const out_H = H - 2;
    int const out_W = W - 2;

    float *d_input_hwc, *d_filter_nhwc, *d_output_hwc;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM28_H - 1) / BM28_H,
        (out_W + BM28_W - 1) / BM28_W,
        (C_out + BN28 - 1) / BN28);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_vec4_filter_conv2d_kernel<<<grid, block, 0, s>>>(
            d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    };

    std::function<void(cudaStream_t)> fn = kernel_fn;
    float latency = measure_performance(fn, stream, num_repeats, num_warmups);

    CHECK_CUDA_ERROR(cudaFree(d_input_hwc));
    CHECK_CUDA_ERROR(cudaFree(d_filter_nhwc));
    CHECK_CUDA_ERROR(cudaFree(d_output_hwc));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

template <typename T>
void launch_wmma_nhwc_vec4_filter_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                               size_t C_in, size_t C_out, size_t H, size_t W,
                                               cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    T *d_input_hwc, *d_filter_nhwc, *d_output_hwc;
    cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(T));
    cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(T));
    cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(T));

    int const n1 = C_in * H * W;
    chw_to_hwc_kernel<<<(n1+255)/256, 256, 0, stream>>>(d_input_hwc, d_input, C_in, H, W);
    int const n2 = C_out * C_in * 9;
    reorder_filter_nhwc_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_nhwc, d_filter, C_out, C_in);

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM28_H - 1) / BM28_H,
        (out_W + BM28_W - 1) / BM28_W,
        (C_out + BN28 - 1) / BN28);
    wmma_nhwc_vec4_filter_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_output_hwc);
}
