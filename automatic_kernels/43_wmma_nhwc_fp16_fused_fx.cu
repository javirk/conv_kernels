#pragma once

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Experiment 43: Fuse fx loop — load wider column strip, compute 3 fx per fy.
//
// Base: exp 40 (FP16 BK=64, BM=128=16x8, 1.26ms).
// Insight: For a given fy, the 3 fx values load nearly overlapping input patches.
// Instead of 3 separate As loads per fy (24KB each = 72KB total),
// load a single wider strip covering cols [tile_col..tile_col+9] = 10 cols.
// Then for each fx, extract the right 8-col window via SMEM indexing.
//
// Layout: As_wide[BM_H * 10][BK_PAD] = [160][72].
// For fx, the 128 spatial positions map to As_wide[row*10 + col_local + fx][k].
// But WMMA needs contiguous rows — so we re-pack into As[128][72] per fx.
// Net: 1 wide load (20KB) + 3 small SMEM-to-SMEM copies (16KB each) vs 3 loads (48KB).
// Actually the copies add syncs too... let's just load Bs 3x and reuse As_wide.
//
// Alternative approach: keep BM_W=8 layout but process 3 fx per SMEM load.
// For each fy: load As_wide (10 cols * 16 rows * 64 ch), load Bs for each fx.
// Compute happens between syncs for each fx using different SMEM row offsets.
//
// Actually simplest approach: just pack the 3 fx windows into As for each fy
// using a wider BM. Map m = fx*128 + spatial_pos. BM becomes 384 (3*128).
// But that's too much SMEM.
//
// OK different strategy entirely: Register-level B fragment reuse.
// For a given fy, preload all 3 Bs (for fx=0,1,2) into registers.
// Then load As for each fx and compute immediately.
// Bs is loaded once and the 3 b_frag sets are kept in registers.
//
// Bs per fx: 64*64 halves in SMEM → WMMA fragments.
// Per warp, per fx: 4 k-steps × 2 b_frags = 8 b_frags.
// 3 fx × 8 = 24 b_frags in registers. Each is 8 halves = 4 regs → 96 regs for B.
// Plus 4 acc frags × 8 floats = 32 regs. Plus a_frags. Total ~140+ regs.
// With launch_bounds(256,2): 128 regs/thread. Too tight.
//
// Simpler: just reduce syncs by only syncing once for combined Bs load.
// Actually the real win is: load As once for all 3 fx, not 3 separate times.
//
// NEW IDEA: For each fy, expand BM_W from 8 to 10 (covering the 10 unique columns).
// As[160][72]. Then for each fx=0,1,2, the WMMA tiles read from different
// row offsets within As. No re-packing needed!
//
// The trick: each warp computes from a subset of the 160 rows.
// For fx=0: rows with (m%10) in [0..7] → m = r*10+c, c in [0..7]
// For fx=1: rows with (m%10) in [1..8]
// For fx=2: rows with (m%10) in [2..9]
//
// BUT WMMA load_matrix_sync needs contiguous 16×K with uniform stride.
// The rows are non-contiguous... can't do this directly.
//
// FALLBACK: simpler approach. Just reduce the number of syncs from 18 to 12
// by loading Bs for next fx while computing current fx (software pipeline Bs only).
// Bs is small (9.2KB), so double-buffered Bs adds only 9.2KB.
// Total SMEM: As[128][72] + Bs[2][64][72] = 18.4KB + 18.4KB = 36.8KB.
// 2 blocks at 100KB: 73.6KB < 100KB. ✓
// Saves 1 sync per fx transition (no need to wait for Bs load when As unchanged).

constexpr int BM43_H = 16;
constexpr int BM43_W = 8;
constexpr int BM43 = BM43_H * BM43_W;
constexpr int BN43 = 64;
constexpr int BK43 = 64;
constexpr int WM43 = 16;
constexpr int WN43 = 16;
constexpr int WK43 = 16;

constexpr int BK43_PAD = BK43 + 8;  // 72
constexpr int BN43_PAD = BN43 + 8;  // 72

__device__ __forceinline__ void load_As43(
    __half As[][BK43_PAD], __half const* __restrict__ input,
    int tile_row, int tile_col, int fy, int fx,
    int C_in, int H, int W, int tid, int total_threads)
{
    constexpr int vec_count = (BM43 * BK43) / 8;
    for (int vi = tid; vi < vec_count; vi += total_threads)
    {
        int const elem = vi * 8;
        int const m = elem / BK43;
        int const k = elem % BK43;
        int const row = tile_row + (m / BM43_W) + fy;
        int const col = tile_col + (m % BM43_W) + fx;
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

__device__ __forceinline__ void load_Bs43(
    __half Bs[][BN43_PAD], __half const* __restrict__ filter_t,
    int block_n, int fy, int fx,
    int C_in, int C_out, int tid, int total_threads)
{
    constexpr int vec_count = (BK43 * BN43) / 8;
    for (int vi = tid; vi < vec_count; vi += total_threads)
    {
        int const elem = vi * 8;
        int const k = elem / BN43;
        int const n = elem % BN43;
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
__global__ void wmma_nhwc_fp16_fused_fx_conv2d_kernel(
    float* __restrict__ output,
    __half const* __restrict__ input,
    __half const* __restrict__ filter_t,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM43_H;
    int const tile_col = blockIdx.y * BM43_W;
    int const block_n = blockIdx.z * BN43;

    // Double-buffered Bs, single-buffered As
    __shared__ __half As[BM43][BK43_PAD];       // 128*72*2 = 18432 bytes
    __shared__ __half Bs[2][BK43][BN43_PAD];    // 2*64*72*2 = 18432 bytes
    // Total: 36864 bytes ≈ 36KB. 2 blocks × 36KB = 72KB < 100KB ✓

    int const warp_id = threadIdx.x / 32;
    constexpr int warps_n = 2;
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM43, WN43, WK43, float> acc00, acc10, acc01, acc11;
    wmma::fill_fragment(acc00, 0.0f);
    wmma::fill_fragment(acc10, 0.0f);
    wmma::fill_fragment(acc01, 0.0f);
    wmma::fill_fragment(acc11, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    // Linearize the 9 (fy,fx) iterations for pipelining
    // iter 0: (0,0), iter 1: (0,1), ..., iter 8: (2,2)
    // Pipeline: load Bs[next] while computing with Bs[cur]

    // Load first iteration: As for (0,0), Bs[0] for (0,0)
    load_As43(As, input, tile_row, tile_col, 0, 0, C_in, H, W, tid, total_threads);
    load_Bs43(Bs[0], filter_t, block_n, 0, 0, C_in, C_out, tid, total_threads);
    __syncthreads();

    for (int iter = 0; iter < 9; ++iter)
    {
        int const fy = iter / 3;
        int const fx = iter % 3;
        int const cur_buf = iter & 1;

        // Prefetch next iteration's Bs (and As if fx changes row)
        if (iter + 1 < 9)
        {
            int const next_fy = (iter + 1) / 3;
            int const next_fx = (iter + 1) % 3;
            int const next_buf = 1 - cur_buf;

            // Load next Bs into alternate buffer
            load_Bs43(Bs[next_buf], filter_t, block_n, next_fy, next_fx,
                     C_in, C_out, tid, total_threads);
        }

        // Compute with current As and Bs[cur_buf]
        #pragma unroll
        for (int kk = 0; kk < BK43; kk += WK43)
        {
            wmma::fragment<wmma::matrix_a, WM43, WN43, WK43,
                           __half, wmma::row_major> a_frag0, a_frag1;
            wmma::fragment<wmma::matrix_b, WM43, WN43, WK43,
                           __half, wmma::row_major> b_frag0, b_frag1;

            int const m_base = warp_m * 32;
            int const n_base = warp_n * 32;
            wmma::load_matrix_sync(a_frag0, &As[m_base][kk], BK43_PAD);
            wmma::load_matrix_sync(a_frag1, &As[m_base + 16][kk], BK43_PAD);
            wmma::load_matrix_sync(b_frag0, &Bs[cur_buf][kk][n_base], BN43_PAD);
            wmma::load_matrix_sync(b_frag1, &Bs[cur_buf][kk][n_base + 16], BN43_PAD);

            wmma::mma_sync(acc00, a_frag0, b_frag0, acc00);
            wmma::mma_sync(acc10, a_frag1, b_frag0, acc10);
            wmma::mma_sync(acc01, a_frag0, b_frag1, acc01);
            wmma::mma_sync(acc11, a_frag1, b_frag1, acc11);
        }

        __syncthreads();

        // Load next As (different for each fx)
        if (iter + 1 < 9)
        {
            int const next_fy = (iter + 1) / 3;
            int const next_fx = (iter + 1) % 3;
            load_As43(As, input, tile_row, tile_col, next_fy, next_fx,
                     C_in, H, W, tid, total_threads);
            __syncthreads();
        }
    }

    // Store results via SMEM (reuse As)
    float* Cs = reinterpret_cast<float*>(&As[0][0]);

    if (warp_m < 2)
    {
        int const m_off = warp_m * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN43 + n_off], acc00, BN43, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN43 + n_off], acc10, BN43, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN43 + n_off + 16], acc01, BN43, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN43 + n_off + 16], acc11, BN43, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN43; idx += total_threads)
    {
        int const m = idx / BN43;
        int const n = idx & (BN43 - 1);
        int const out_row = tile_row + (m / BM43_W);
        int const out_col = tile_col + (m % BM43_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN43 + n];
    }
    __syncthreads();

    if (warp_m >= 2)
    {
        int const m_off = (warp_m - 2) * 32;
        int const n_off = warp_n * 32;
        wmma::store_matrix_sync(&Cs[m_off * BN43 + n_off], acc00, BN43, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN43 + n_off], acc10, BN43, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[m_off * BN43 + n_off + 16], acc01, BN43, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cs[(m_off + 16) * BN43 + n_off + 16], acc11, BN43, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < 64 * BN43; idx += total_threads)
    {
        int const m = idx / BN43;
        int const n = idx & (BN43 - 1);
        int const m_global = m + 64;
        int const out_row = tile_row + (m_global / BM43_W);
        int const out_col = tile_col + (m_global % BM43_W);
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
            output[out_row * out_W * C_out + out_col * C_out + oc] = Cs[m * BN43 + n];
    }
}

float profile_nhwc_fp16_fused_fx_conv2d(size_t C_in, size_t C_out, size_t H, size_t W)
{
    constexpr int num_repeats = 100;
    constexpr int num_warmups = 10;
    int const out_H = H - 2;
    int const out_W = W - 2;

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
        (out_H + BM43_H - 1) / BM43_H,
        (out_W + BM43_W - 1) / BM43_W,
        (C_out + BN43 - 1) / BN43);

    auto kernel_fn = [&](cudaStream_t s) {
        wmma_nhwc_fp16_fused_fx_conv2d_kernel<<<grid, block, 0, s>>>(
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
void launch_wmma_nhwc_fp16_fused_fx_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
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
        (out_H + BM43_H - 1) / BM43_H,
        (out_W + BM43_W - 1) / BM43_W,
        (C_out + BN43 - 1) / BN43);
    wmma_nhwc_fp16_fused_fx_conv2d_kernel<<<grid, block, 0, stream>>>(
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
