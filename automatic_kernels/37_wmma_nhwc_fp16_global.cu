#pragma once

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Experiment 37: FP16 in global memory — load halves directly, no conversion.
//
// Base: exp 36 (FP16 WMMA with float→half conversion during SMEM load).
// Change: Pre-convert input and filter to FP16 in global memory.
// Load half2/half4 directly → halves bandwidth requirement.
// Use __half2 loads for 2x bandwidth.

// Kernel to convert float to half in-place
__global__ void float_to_half_kernel(__half* __restrict__ out,
                                      float const* __restrict__ in, int n)
{
    int const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

constexpr int BM37_H = 16;
constexpr int BM37_W = 8;
constexpr int BM37 = BM37_H * BM37_W;
constexpr int BN37 = 64;
constexpr int BK37 = 16;
constexpr int WM37 = 16;
constexpr int WN37 = 16;
constexpr int WK37 = 16;

constexpr int BK37_PAD = BK37 + 8;  // 24
constexpr int BN37_PAD = BN37 + 8;  // 72

__device__ __forceinline__ void load_As_half_tile_37(
    __half As[][BK37_PAD], __half const* __restrict__ input,
    int tile_row, int tile_col, int fy, int fx, int ic_base,
    int C_in, int H, int W, int tid, int total_threads)
{
    // Load __half2 (2 halves at a time) — BM37*BK37 = 2048 halves = 1024 half2
    // With 256 threads: 4 half2 per thread
    constexpr int half2_count = (BM37 * BK37) / 2;  // 1024
    for (int vi = tid; vi < half2_count; vi += total_threads)
    {
        int const elem = vi * 2;
        int const m = elem / BK37;
        int const k = elem % BK37;
        int const row = tile_row + (m / BM37_W) + fy;
        int const col = tile_col + (m % BM37_W) + fx;
        int const ic = ic_base + k;
        __half2 val;
        if (row < H && col < W && ic + 1 < C_in) {
            val = *reinterpret_cast<__half2 const*>(
                &input[row * W * C_in + col * C_in + ic]);
        } else {
            __half h0 = (row < H && col < W && ic < C_in) ?
                input[row * W * C_in + col * C_in + ic] : __float2half(0.0f);
            __half h1 = (row < H && col < W && ic + 1 < C_in) ?
                input[row * W * C_in + col * C_in + ic + 1] : __float2half(0.0f);
            val = __halves2half2(h0, h1);
        }
        As[m][k]     = __low2half(val);
        As[m][k + 1] = __high2half(val);
    }
}

__device__ __forceinline__ void load_Bs_half_tile_37(
    __half Bs[][BN37_PAD], __half const* __restrict__ filter_t,
    int block_n, int fy, int fx, int ic_base,
    int C_in, int C_out, int tid, int total_threads)
{
    // filter_t: [fy, fx, ic, oc] in __half. Load half2 along oc dimension.
    constexpr int half2_count = (BK37 * BN37) / 2;  // 512
    for (int vi = tid; vi < half2_count; vi += total_threads)
    {
        int const elem = vi * 2;
        int const k = elem / BN37;
        int const n = elem % BN37;
        int const ic = ic_base + k;
        int const oc = block_n + n;
        __half2 val;
        if (ic < C_in && oc + 1 < C_out) {
            val = *reinterpret_cast<__half2 const*>(
                &filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc]);
        } else {
            __half h0 = (ic < C_in && oc < C_out) ?
                filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc] : __float2half(0.0f);
            __half h1 = (ic < C_in && oc + 1 < C_out) ?
                filter_t[fy * 3 * C_in * C_out + fx * C_in * C_out + ic * C_out + oc + 1] : __float2half(0.0f);
            val = __halves2half2(h0, h1);
        }
        Bs[k][n]     = __low2half(val);
        Bs[k][n + 1] = __high2half(val);
    }
}

__launch_bounds__(256, 2)
__global__ void wmma_nhwc_fp16_global_conv2d_kernel(
    float* __restrict__ output,
    __half const* __restrict__ input,
    __half const* __restrict__ filter_t,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM37_H;
    int const tile_col = blockIdx.y * BM37_W;
    int const block_n = blockIdx.z * BN37;

    __shared__ __half As[2][BM37][BK37_PAD];
    __shared__ __half Bs[2][BK37][BN37_PAD];

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = 2;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM37, WN37, WK37, float> acc00, acc10, acc01, acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;
    int const num_ic_iters = (C_in + BK37 - 1) / BK37;

    for (int fy = 0; fy < 3; ++fy)
    {
        for (int fx = 0; fx < 3; ++fx)
        {
            load_As_half_tile_37(As[0], input, tile_row, tile_col, fy, fx, 0,
                                C_in, H, W, tid, total_threads);
            load_Bs_half_tile_37(Bs[0], filter_t, block_n, fy, fx, 0,
                                C_in, C_out, tid, total_threads);
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK37;
                    load_As_half_tile_37(As[next_buf], input, tile_row, tile_col,
                                        fy, fx, next_ic, C_in, H, W, tid, total_threads);
                    load_Bs_half_tile_37(Bs[next_buf], filter_t, block_n, fy, fx,
                                        next_ic, C_in, C_out, tid, total_threads);
                }

                {
                    wmma::fragment<wmma::matrix_a, WM37, WN37, WK37,
                                   __half, wmma::row_major> a_frag0, a_frag1;
                    wmma::fragment<wmma::matrix_b, WM37, WN37, WK37,
                                   __half, wmma::row_major> b_frag0, b_frag1;

                    int const m_base = warp_m * 32;
                    int const n_base = warp_n * 32;
                    wmma::load_matrix_sync(a_frag0, &As[cur_buf][m_base][0], BK37_PAD);
                    wmma::load_matrix_sync(a_frag1, &As[cur_buf][m_base + 16][0], BK37_PAD);
                    wmma::load_matrix_sync(b_frag0, &Bs[cur_buf][0][n_base], BN37_PAD);
                    wmma::load_matrix_sync(b_frag1, &Bs[cur_buf][0][n_base + 16], BN37_PAD);

                    wmma::mma_sync(acc00, a_frag0, b_frag0, acc00);
                    wmma::mma_sync(acc10, a_frag1, b_frag0, acc10);
                    wmma::mma_sync(acc01, a_frag0, b_frag1, acc01);
                    wmma::mma_sync(acc11, a_frag1, b_frag1, acc11);
                }

                __syncthreads();
            }
        }
    }

    float* Cs = reinterpret_cast<float*>(&As[0][0][0]);

    if (warp_m < 2)
    {
        int const m_off = warp_m * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN37 + n_off], acc00, BN37, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN37 + n_off], acc10, BN37, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN37 + n_off + 16], acc01, BN37, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN37 + n_off + 16], acc11, BN37, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN37; idx += total_threads)
    {
        int const m = idx / BN37;
        int const n = idx & (BN37 - 1);
        int const out_row = tile_row + (m / BM37_W);
        int const out_col = tile_col + (m % BM37_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN37 + n];
    }
    __syncthreads();

    if (warp_m >= 2)
    {
        int const m_off = (warp_m - 2) * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN37 + n_off], acc00, BN37, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN37 + n_off], acc10, BN37, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN37 + n_off + 16], acc01, BN37, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN37 + n_off + 16], acc11, BN37, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN37; idx += total_threads)
    {
        int const m = idx / BN37;
        int const n = idx & (BN37 - 1);
        int const m_global = m + 64;
        int const out_row = tile_row + (m_global / BM37_W);
        int const out_col = tile_col + (m_global % BM37_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN37 + n];
    }
}

float profile_nhwc_fp16_global_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats = 100;
    constexpr int num_warmups = 10;
    int const out_H = H - 2;
    int const out_W = W - 2;

    // Allocate float buffers for layout conversion, then half buffers for kernel
    float *d_input_hwc_f, *d_filter_nhwc_f, *d_filter_t_f;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_hwc_f, C_in * H * W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_nhwc_f, C_out * C_in * 9 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_t_f, C_out * C_in * 9 * sizeof(float)));

    __half *d_input_h, *d_filter_t_h;
    float *d_output_hwc;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_h, C_in * H * W * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter_t_h, C_out * C_in * 9 * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Pre-convert (not measured)
    int const filter_elems = C_out * C_in * 9;
    transpose_filter_kernel<<<(filter_elems + 255) / 256, 256, 0, stream>>>(
        d_filter_t_f, d_filter_nhwc_f, C_in, C_out);
    int const input_elems = C_in * H * W;
    float_to_half_kernel<<<(input_elems + 255) / 256, 256, 0, stream>>>(
        d_input_h, d_input_hwc_f, input_elems);
    float_to_half_kernel<<<(filter_elems + 255) / 256, 256, 0, stream>>>(
        d_filter_t_h, d_filter_t_f, filter_elems);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM37_H - 1) / BM37_H,
        (out_W + BM37_W - 1) / BM37_W,
        (C_out + BN37 - 1) / BN37);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_fp16_global_conv2d_kernel<<<grid, block, 0, s>>>(
            d_output_hwc, d_input_h, d_filter_t_h, C_in, C_out, H, W);
    };

    std::function<void(cudaStream_t)> fn = kernel_fn;
    float latency = measure_performance(fn, stream, num_repeats, num_warmups);

    CHECK_CUDA_ERROR(cudaFree(d_input_hwc_f));
    CHECK_CUDA_ERROR(cudaFree(d_filter_nhwc_f));
    CHECK_CUDA_ERROR(cudaFree(d_filter_t_f));
    CHECK_CUDA_ERROR(cudaFree(d_input_h));
    CHECK_CUDA_ERROR(cudaFree(d_filter_t_h));
    CHECK_CUDA_ERROR(cudaFree(d_output_hwc));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

template <typename T>
void launch_wmma_nhwc_fp16_global_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                               size_t C_in, size_t C_out, size_t H, size_t W,
                                               cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    T *d_input_hwc, *d_filter_nhwc, *d_filter_t;
    cudaMalloc(&d_input_hwc, C_in * H * W * sizeof(T));
    cudaMalloc(&d_filter_nhwc, C_out * C_in * 9 * sizeof(T));
    cudaMalloc(&d_filter_t, C_out * C_in * 9 * sizeof(T));

    int const n1 = C_in * H * W;
    chw_to_hwc_kernel<<<(n1+255)/256, 256, 0, stream>>>(d_input_hwc, d_input, C_in, H, W);
    int const n2 = C_out * C_in * 9;
    reorder_filter_nhwc_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_nhwc, d_filter, C_out, C_in);
    transpose_filter_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_t, d_filter_nhwc, C_in, C_out);

    __half *d_input_h, *d_filter_t_h;
    T *d_output_hwc;
    cudaMalloc(&d_input_h, C_in * H * W * sizeof(__half));
    cudaMalloc(&d_filter_t_h, C_out * C_in * 9 * sizeof(__half));
    cudaMalloc(&d_output_hwc, C_out * out_H * out_W * sizeof(T));

    float_to_half_kernel<<<(n1+255)/256, 256, 0, stream>>>(d_input_h, d_input_hwc, n1);
    float_to_half_kernel<<<(n2+255)/256, 256, 0, stream>>>(d_filter_t_h, d_filter_t, n2);

    dim3 const block(256);
    dim3 const grid(
        (out_H + BM37_H - 1) / BM37_H,
        (out_W + BM37_W - 1) / BM37_W,
        (C_out + BN37 - 1) / BN37);
    wmma_nhwc_fp16_global_conv2d_kernel<<<grid, block, 0, stream>>>(
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
