#pragma once

#include <mma.h>
using namespace nvcuda;

// Experiment 26: Vectorized float4 loads for SMEM filling.
//
// Base: exp 25 (padded SMEM, 8 warps, 2x reg tiling).
// Change: Use float4 loads when filling As and Bs from global memory.
// NHWC layout means channels are contiguous → perfect for float4.
// BK=16 → 4 float4 loads per row of As, 16 float4 loads per row of Bs.
// This should significantly reduce the number of load instructions.

constexpr int BM26_H = 8;
constexpr int BM26_W = 8;
constexpr int BM26 = BM26_H * BM26_W;  // 64
constexpr int BN26 = 64;
constexpr int BK26 = 16;
constexpr int WM26 = 16;
constexpr int WN26 = 16;
constexpr int WK26 = 8;

constexpr int BK26_PAD = BK26 + 1;  // 17
constexpr int BN26_PAD = BN26 + 1;  // 65

__launch_bounds__(256, 2)
__global__ void wmma_nhwc_vec4_conv2d_kernel(
    float* __restrict__ output,       // [out_H, out_W, C_out]
    float const* __restrict__ input,  // [H, W, C_in]
    float const* __restrict__ filter, // [C_out, 3, 3, C_in]
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM26_H;
    int const tile_col = blockIdx.y * BM26_W;
    int const block_n = blockIdx.z * BN26;

    __shared__ float As[2][BM26][BK26_PAD];   // padded for bank conflicts
    __shared__ float Bs[2][BK26][BN26_PAD];   // padded for bank conflicts

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = BN26 / WN26;  // 4
    int const warp_m = warp_id / warps_n;  // 0 or 1
    int const warp_n = warp_id % warps_n;  // 0..3

    wmma::fragment<wmma::accumulator, WM26, WN26, WK26, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;  // 256
    int const num_ic_iters = (C_in + BK26 - 1) / BK26;

    // As: 64 rows x 16 cols = 1024 elements = 256 float4s → 1 float4 per thread
    // Bs: 16 rows x 64 cols = 1024 elements = 256 float4s → 1 float4 per thread
    constexpr int As_vec4_count = (BM26 * BK26) / 4;  // 256
    constexpr int Bs_vec4_count = (BK26 * BN26) / 4;  // 256

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            // Load first ic chunk with float4
            {
                int const ic_base = 0;
                for (int vi = tid; vi < As_vec4_count; vi += total_threads)
                {
                    int const elem = vi * 4;
                    int const m = elem / BK26;
                    int const k = elem % BK26;
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
                    As[0][m][k]     = val.x;
                    As[0][m][k + 1] = val.y;
                    As[0][m][k + 2] = val.z;
                    As[0][m][k + 3] = val.w;
                }
                for (int vi = tid; vi < Bs_vec4_count; vi += total_threads)
                {
                    int const elem = vi * 4;
                    int const k = elem / BN26;
                    int const n = elem % BN26;
                    int const oc = block_n + n;
                    int const ic = ic_base + k;
                    float4 val;
                    // Filter: [C_out, 3, 3, C_in] → row k maps to ic, col n maps to oc
                    // But Bs is [BK][BN] = [ic_local][oc_local], and filter is [oc][fy][fx][ic]
                    // So we're loading 4 consecutive oc values (n, n+1, n+2, n+3)
                    // Filter layout: filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic]
                    // For consecutive oc: stride is 9*C_in, NOT contiguous!
                    // So we can't use float4 for Bs along the n dimension.
                    // Instead, load scalar.
                    val.x = (ic < C_in && oc < C_out) ?
                        filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                    val.y = (ic < C_in && oc + 1 < C_out) ?
                        filter[(oc + 1) * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                    val.z = (ic < C_in && oc + 2 < C_out) ?
                        filter[(oc + 2) * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                    val.w = (ic < C_in && oc + 3 < C_out) ?
                        filter[(oc + 3) * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                    Bs[0][k][n]     = val.x;
                    Bs[0][k][n + 1] = val.y;
                    Bs[0][k][n + 2] = val.z;
                    Bs[0][k][n + 3] = val.w;
                }
            }
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK26;
                    for (int vi = tid; vi < As_vec4_count; vi += total_threads)
                    {
                        int const elem = vi * 4;
                        int const m = elem / BK26;
                        int const k = elem % BK26;
                        int const row = tile_row + (m >> 3) + fy;
                        int const col = tile_col + (m & 7) + fx;
                        int const ic = next_ic + k;
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
                        As[next_buf][m][k]     = val.x;
                        As[next_buf][m][k + 1] = val.y;
                        As[next_buf][m][k + 2] = val.z;
                        As[next_buf][m][k + 3] = val.w;
                    }
                    for (int vi = tid; vi < Bs_vec4_count; vi += total_threads)
                    {
                        int const elem = vi * 4;
                        int const k = elem / BN26;
                        int const n = elem % BN26;
                        int const oc = block_n + n;
                        int const ic = next_ic + k;
                        float4 val;
                        val.x = (ic < C_in && oc < C_out) ?
                            filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                        val.y = (ic < C_in && oc + 1 < C_out) ?
                            filter[(oc + 1) * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                        val.z = (ic < C_in && oc + 2 < C_out) ?
                            filter[(oc + 2) * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                        val.w = (ic < C_in && oc + 3 < C_out) ?
                            filter[(oc + 3) * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                        Bs[next_buf][k][n]     = val.x;
                        Bs[next_buf][k][n + 1] = val.y;
                        Bs[next_buf][k][n + 2] = val.z;
                        Bs[next_buf][k][n + 3] = val.w;
                    }
                }

                #pragma unroll
                for (int kk = 0; kk < BK26; kk += WK26)
                {
                    wmma::fragment<wmma::matrix_a, WM26, WN26, WK26,
                                   wmma::precision::tf32, wmma::row_major> a_frag0, a_frag1;
                    wmma::fragment<wmma::matrix_b, WM26, WN26, WK26,
                                   wmma::precision::tf32, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag0, &As[cur_buf][warp_m * 32][kk], BK26_PAD);
                    wmma::load_matrix_sync(a_frag1, &As[cur_buf][warp_m * 32 + 16][kk], BK26_PAD);
                    wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][warp_n * WN26], BN26_PAD);
                    wmma::mma_sync(acc0, a_frag0, b_frag, acc0);
                    wmma::mma_sync(acc1, a_frag1, b_frag, acc1);
                }

                __syncthreads();
            }
        }
    }

    // Store using aliased SMEM
    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);
    wmma::store_matrix_sync(&Cs[(warp_m * 32) * BN26 + warp_n * WN26], acc0, BN26, wmma::mem_row_major);
    wmma::store_matrix_sync(&Cs[(warp_m * 32 + 16) * BN26 + warp_n * WN26], acc1, BN26, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM26 * BN26; idx += total_threads)
    {
        int const m = idx / BN26;
        int const n = idx & (BN26 - 1);
        int const out_row = tile_row + (m >> 3);
        int const out_col = tile_col + (m & 7);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN26 + n];
        }
    }
}

// Profiling function
float profile_nhwc_vec4_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
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
        (out_H + BM26_H - 1) / BM26_H,
        (out_W + BM26_W - 1) / BM26_W,
        (C_out + BN26 - 1) / BN26);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_vec4_conv2d_kernel<<<grid, block, 0, s>>>(
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

// Launch wrapper for unit tests
template <typename T>
void launch_wmma_nhwc_vec4_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
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
        (out_H + BM26_H - 1) / BM26_H,
        (out_W + BM26_W - 1) / BM26_W,
        (C_out + BN26 - 1) / BN26);
    wmma_nhwc_vec4_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_output_hwc);
}
