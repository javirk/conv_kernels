#pragma once

#include <mma.h>
using namespace nvcuda;

// NHWC WMMA with BK=C_in (full channel dimension per filter position).
// For C_in=64, each filter position loads 64 channels in one SMEM tile.
// No ic-loop double buffering needed — just 9 filter position iterations
// with single-buffer SMEM.

constexpr int BM19_H = 8;
constexpr int BM19_W = 8;
constexpr int BM19 = BM19_H * BM19_W;  // 64
constexpr int BN19 = 64;
constexpr int BK19 = 64;               // full C_in in one shot
constexpr int WM19 = 16;
constexpr int WN19 = 16;
constexpr int WK19 = 8;

// SMEM: As[64][64] = 16KB, Bs[64][64] = 16KB, Cs[64][64] = 16KB
// Total = 48KB — right at the limit. Use Cs aliased with As.

__launch_bounds__(512, 1)
__global__ void wmma_nhwc_fullk_conv2d_kernel(
    float* __restrict__ output,       // [out_H, out_W, C_out]
    float const* __restrict__ input,  // [H, W, C_in]
    float const* __restrict__ filter, // [C_out, 3, 3, C_in]
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM19_H;
    int const tile_col = blockIdx.y * BM19_W;
    int const block_n = blockIdx.z * BN19;

    // Single buffer since no ic loop
    __shared__ float As[BM19][BK19];   // 64*64 = 16KB
    __shared__ float Bs[BK19][BN19];   // 64*64 = 16KB

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN19 / WN19;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM19, WN19, WK19, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            // Load A[BM19][C_in]: input at shifted positions, all channels
            for (int idx = tid; idx < BM19 * C_in; idx += total_threads)
            {
                int const m = idx / C_in;
                int const ic = idx % C_in;
                int const row = tile_row + (m >> 3) + fy;
                int const col = tile_col + (m & 7) + fx;
                As[m][ic] = (row < H && col < W) ?
                    input[row * W * C_in + col * C_in + ic] : 0.0f;
            }
            // Pad remaining columns if C_in < BK19
            if (C_in < BK19)
            {
                for (int idx = tid; idx < BM19 * (BK19 - C_in); idx += total_threads)
                {
                    int const m = idx / (BK19 - C_in);
                    int const k = C_in + idx % (BK19 - C_in);
                    As[m][k] = 0.0f;
                }
            }

            // Load B[C_in][BN19]: filter weights for this (fy,fx)
            for (int idx = tid; idx < C_in * BN19; idx += total_threads)
            {
                int const ic = idx / BN19;
                int const n = idx & (BN19 - 1);
                int const oc = block_n + n;
                Bs[ic][n] = (oc < C_out) ?
                    filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
            }
            // Pad remaining rows if C_in < BK19
            if (C_in < BK19)
            {
                for (int idx = tid; idx < (BK19 - C_in) * BN19; idx += total_threads)
                {
                    int const k = C_in + idx / BN19;
                    int const n = idx % BN19;
                    Bs[k][n] = 0.0f;
                }
            }

            __syncthreads();

            // WMMA compute: iterate over K=BK19 in chunks of WK19=8
            #pragma unroll
            for (int kk = 0; kk < BK19; kk += WK19)
            {
                wmma::fragment<wmma::matrix_a, WM19, WN19, WK19,
                               wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM19, WN19, WK19,
                               wmma::precision::tf32, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, &As[warp_m * WM19][kk], BK19);
                wmma::load_matrix_sync(b_frag, &Bs[kk][warp_n * WN19], BN19);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }

            __syncthreads();
        }
    }

    // Store using As as scratch (aliased — safe since loop is done)
    // Actually Cs needs BM19*BN19 = 64*64 = same as As. Reuse As memory.
    float* Cs = &As[0][0];  // reinterpret As as flat Cs[BM19][BN19]
    // But BN19=64 and BK19=64, so stride is the same!
    wmma::store_matrix_sync(&Cs[(warp_m * WM19) * BN19 + warp_n * WN19], acc, BN19, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM19 * BN19; idx += total_threads)
    {
        int const m = idx / BN19;
        int const n = idx & (BN19 - 1);
        int const out_row = tile_row + (m >> 3);
        int const out_col = tile_col + (m & 7);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN19 + n];
        }
    }
}

// Profiling function
float profile_nhwc_fullk_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
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

    constexpr int num_warps = (BM19 / WM19) * (BN19 / WN19);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (out_H + BM19_H - 1) / BM19_H,
        (out_W + BM19_W - 1) / BM19_W,
        (C_out + BN19 - 1) / BN19);

    std::function<void(cudaStream_t)> fn = [&](cudaStream_t s) {
        wmma_nhwc_fullk_conv2d_kernel<<<grid, block, 0, s>>>(
            d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    };

    float latency = measure_performance(fn, stream, num_repeats, num_warmups);

    CHECK_CUDA_ERROR(cudaFree(d_input_hwc));
    CHECK_CUDA_ERROR(cudaFree(d_filter_nhwc));
    CHECK_CUDA_ERROR(cudaFree(d_output_hwc));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return latency;
}

// Launch wrapper for unit tests
template <typename T>
void launch_wmma_nhwc_fullk_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
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

    constexpr int num_warps = (BM19 / WM19) * (BN19 / WN19);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (out_H + BM19_H - 1) / BM19_H,
        (out_W + BM19_W - 1) / BM19_W,
        (C_out + BN19 - 1) / BN19);
    wmma_nhwc_fullk_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_output_hwc);
}
