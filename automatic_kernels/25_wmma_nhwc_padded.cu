#pragma once

#include <mma.h>
using namespace nvcuda;

// NHWC WMMA with 8 warps, 2x register tiling (exp 23 base) + SMEM padding.
//
// Bank conflict analysis:
//   As[64][16]: stride=16 words, gcd(16,32)=16 → 16-way conflicts!
//   Bs[16][64]: stride=64 words, gcd(64,32)=32 → all same bank per column!
//
// Fix: pad to coprime-with-32 strides:
//   As[64][17]: stride=17, gcd(17,32)=1 → zero bank conflicts
//   Bs[16][65]: stride=65, gcd(65,32)=1 → zero bank conflicts
//
// SMEM: As[2][64][17]=8.5KB, Bs[2][16][65]=8.1KB ≈ 17KB total.
// With aliased Cs: still fits 2 blocks/SM.

constexpr int BM25_H = 8;
constexpr int BM25_W = 8;
constexpr int BM25 = BM25_H * BM25_W;  // 64
constexpr int BN25 = 64;
constexpr int BK25 = 16;
constexpr int WM25 = 16;
constexpr int WN25 = 16;
constexpr int WK25 = 8;

// Padded dimensions for bank-conflict-free access
constexpr int BK25_PAD = BK25 + 1;  // 17
constexpr int BN25_PAD = BN25 + 1;  // 65

__launch_bounds__(256, 2)
__global__ void wmma_nhwc_padded_conv2d_kernel(
    float* __restrict__ output,       // [out_H, out_W, C_out]
    float const* __restrict__ input,  // [H, W, C_in]
    float const* __restrict__ filter, // [C_out, 3, 3, C_in]
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM25_H;
    int const tile_col = blockIdx.y * BM25_W;
    int const block_n = blockIdx.z * BN25;

    __shared__ float As[2][BM25][BK25_PAD];   // 2*64*17 = 2176 floats
    __shared__ float Bs[2][BK25][BN25_PAD];   // 2*16*65 = 2080 floats

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = BN25 / WN25;  // 4
    int const warp_m = warp_id / warps_n;  // 0 or 1
    int const warp_n = warp_id % warps_n;  // 0..3

    wmma::fragment<wmma::accumulator, WM25, WN25, WK25, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;
    int const num_ic_iters = (C_in + BK25 - 1) / BK25;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            // Load first ic chunk
            for (int idx = tid; idx < BM25 * BK25; idx += total_threads)
            {
                int const m = idx / BK25;
                int const k = idx % BK25;
                int const row = tile_row + (m >> 3) + fy;
                int const col = tile_col + (m & 7) + fx;
                As[0][m][k] = (row < H && col < W && k < C_in) ?
                    input[row * W * C_in + col * C_in + k] : 0.0f;
            }
            for (int idx = tid; idx < BK25 * BN25; idx += total_threads)
            {
                int const k = idx / BN25;
                int const n = idx % BN25;
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
                    int const next_ic = (ici + 1) * BK25;
                    for (int idx = tid; idx < BM25 * BK25; idx += total_threads)
                    {
                        int const m = idx / BK25;
                        int const k = idx % BK25;
                        int const row = tile_row + (m >> 3) + fy;
                        int const col = tile_col + (m & 7) + fx;
                        int const ic = next_ic + k;
                        As[next_buf][m][k] = (row < H && col < W && ic < C_in) ?
                            input[row * W * C_in + col * C_in + ic] : 0.0f;
                    }
                    for (int idx = tid; idx < BK25 * BN25; idx += total_threads)
                    {
                        int const k = idx / BN25;
                        int const n = idx % BN25;
                        int const ic = next_ic + k;
                        int const oc = block_n + n;
                        Bs[next_buf][k][n] = (ic < C_in && oc < C_out) ?
                            filter[oc * 9 * C_in + fy * 3 * C_in + fx * C_in + ic] : 0.0f;
                    }
                }

                #pragma unroll
                for (int kk = 0; kk < BK25; kk += WK25)
                {
                    wmma::fragment<wmma::matrix_a, WM25, WN25, WK25,
                                   wmma::precision::tf32, wmma::row_major> a_frag0, a_frag1;
                    wmma::fragment<wmma::matrix_b, WM25, WN25, WK25,
                                   wmma::precision::tf32, wmma::row_major> b_frag;

                    // Use padded stride for WMMA loads
                    wmma::load_matrix_sync(a_frag0, &As[cur_buf][warp_m * 32][kk], BK25_PAD);
                    wmma::load_matrix_sync(a_frag1, &As[cur_buf][warp_m * 32 + 16][kk], BK25_PAD);
                    wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][warp_n * WN25], BN25_PAD);
                    wmma::mma_sync(acc0, a_frag0, b_frag, acc0);
                    wmma::mma_sync(acc1, a_frag1, b_frag, acc1);
                }

                __syncthreads();
            }
        }
    }

    // Store — use Bs memory as Cs (Bs is 2*16*65=2080 floats, need 64*64=4096)
    // Actually As+Bs total = 2176+2080 = 4256 floats > 4096. Can alias both.
    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);
    wmma::store_matrix_sync(&Cs[(warp_m * 32) * BN25 + warp_n * WN25], acc0, BN25, wmma::mem_row_major);
    wmma::store_matrix_sync(&Cs[(warp_m * 32 + 16) * BN25 + warp_n * WN25], acc1, BN25, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM25 * BN25; idx += total_threads)
    {
        int const m = idx / BN25;
        int const n = idx & (BN25 - 1);
        int const out_row = tile_row + (m >> 3);
        int const out_col = tile_col + (m & 7);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN25 + n];
        }
    }
}

// Profiling function
float profile_nhwc_padded_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
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
        (out_H + BM25_H - 1) / BM25_H,
        (out_W + BM25_W - 1) / BM25_W,
        (C_out + BN25 - 1) / BN25);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_padded_conv2d_kernel<<<grid, block, 0, s>>>(
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
void launch_wmma_nhwc_padded_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
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
        (out_H + BM25_H - 1) / BM25_H,
        (out_W + BM25_W - 1) / BM25_W,
        (C_out + BN25 - 1) / BN25);
    wmma_nhwc_padded_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output_hwc, d_input_hwc, d_filter_nhwc, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();

    int const n3 = C_out * out_H * out_W;
    hwc_to_chw_kernel<<<(n3+255)/256, 256, 0, stream>>>(d_output, d_output_hwc, C_out, out_H, out_W);

    cudaStreamSynchronize(stream);
    cudaFree(d_input_hwc);
    cudaFree(d_filter_nhwc);
    cudaFree(d_output_hwc);
}
