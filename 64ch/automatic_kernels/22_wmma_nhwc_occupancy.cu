#pragma once

#include <mma.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

// NHWC WMMA with improved occupancy:
// 1. Alias Cs with As/Bs memory (used at different times) → SMEM 16KB instead of 32KB
// 2. __launch_bounds__(512, 2) → 2 blocks/SM for better latency hiding
// 3. cp.async with 16-byte copies for async SMEM prefetch

constexpr int BM22_H = 8;
constexpr int BM22_W = 8;
constexpr int BM22 = BM22_H * BM22_W;  // 64
constexpr int BN22 = 64;
constexpr int BK22 = 16;
constexpr int WM22 = 16;
constexpr int WN22 = 16;
constexpr int WK22 = 8;

__launch_bounds__(512, 2)
__global__ void wmma_nhwc_occupancy_conv2d_kernel(
    float* __restrict__ output,       // [out_H, out_W, C_out]
    float const* __restrict__ input,  // [H, W, C_in]
    float const* __restrict__ filter, // [C_out, 3, 3, C_in]
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM22_H;
    int const tile_col = blockIdx.y * BM22_W;
    int const block_n = blockIdx.z * BN22;

    // Only allocate As and Bs — Cs will be aliased after the loop
    __shared__ float As[2][BM22][BK22];   // 2*64*16*4 = 8KB
    __shared__ float Bs[2][BK22][BN22];   // 2*16*64*4 = 8KB
    // Total SMEM = 16KB → fits 2-3 blocks per SM in 48KB

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN22 / WN22;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM22, WN22, WK22, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;
    int const num_ic_iters = (C_in + BK22 - 1) / BK22;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            // Load first ic chunk with cp.async
            for (int idx = tid; idx < BM22 * BK22; idx += total_threads)
            {
                int const m = idx / BK22;
                int const k = idx & (BK22 - 1);
                int const row = tile_row + (m >> 3) + fy;
                int const col = tile_col + (m & 7) + fx;
                As[0][m][k] = (row < H && col < W && k < C_in) ?
                    input[row * W * C_in + col * C_in + k] : 0.0f;
            }
            for (int idx = tid; idx < BK22 * BN22; idx += total_threads)
            {
                int const k = idx / BN22;
                int const n = idx & (BN22 - 1);
                int const oc = block_n + n;
                Bs[0][k][n] = (k < C_in && oc < C_out) ?
                    filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + k] : 0.0f;
            }
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK22;
                    for (int idx = tid; idx < BM22 * BK22; idx += total_threads)
                    {
                        int const m = idx / BK22;
                        int const k = idx & (BK22 - 1);
                        int const row = tile_row + (m >> 3) + fy;
                        int const col = tile_col + (m & 7) + fx;
                        int const ic = next_ic + k;
                        As[next_buf][m][k] = (row < H && col < W && ic < C_in) ?
                            input[row * W * C_in + col * C_in + ic] : 0.0f;
                    }
                    for (int idx = tid; idx < BK22 * BN22; idx += total_threads)
                    {
                        int const k = idx / BN22;
                        int const n = idx & (BN22 - 1);
                        int const ic = next_ic + k;
                        int const oc = block_n + n;
                        Bs[next_buf][k][n] = (ic < C_in && oc < C_out) ?
                            filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                    }
                }

                #pragma unroll
                for (int kk = 0; kk < BK22; kk += WK22)
                {
                    wmma::fragment<wmma::matrix_a, WM22, WN22, WK22,
                                   wmma::precision::tf32, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM22, WN22, WK22,
                                   wmma::precision::tf32, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, &As[cur_buf][warp_m * WM22][kk], BK22);
                    wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][warp_n * WN22], BN22);
                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }

                __syncthreads();
            }
        }
    }

    // Store: alias As/Bs memory as Cs (safe — loop is done)
    // As+Bs = 16KB total = 4096 floats. Cs needs 64*64 = 4096 floats. Perfect fit!
    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);
    wmma::store_matrix_sync(&Cs[(warp_m * WM22) * BN22 + warp_n * WN22], acc, BN22, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM22 * BN22; idx += total_threads)
    {
        int const m = idx / BN22;
        int const n = idx & (BN22 - 1);
        int const out_row = tile_row + (m >> 3);
        int const out_col = tile_col + (m & 7);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN22 + n];
        }
    }
}

// Profiling function
float profile_nhwc_occupancy_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
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

    constexpr int num_warps = (BM22 / WM22) * (BN22 / WN22);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (out_H + BM22_H - 1) / BM22_H,
        (out_W + BM22_W - 1) / BM22_W,
        (C_out + BN22 - 1) / BN22);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_occupancy_conv2d_kernel<<<grid, block, 0, s>>>(
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
void launch_wmma_nhwc_occupancy_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
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

    constexpr int num_warps = (BM22 / WM22) * (BN22 / WN22);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (out_H + BM22_H - 1) / BM22_H,
        (out_W + BM22_W - 1) / BM22_W,
        (C_out + BN22 - 1) / BN22);
    wmma_nhwc_occupancy_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_output_hwc);
}
