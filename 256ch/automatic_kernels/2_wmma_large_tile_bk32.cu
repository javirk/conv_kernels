#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Exp 2: Larger tiles + BK=32 to halve K-loop iterations
// Tile: 128x128, BK=32, 8 warps (256 threads)
// Each warp: 2x2 WMMA = 32x32
// Warp layout: 4x2 (4 rows, 2 cols)
// smem: A[128][32]=8KB + B[32][128]=8KB = 16KB total
// Output store: done in warp-sequential chunks via 32x32 smem buffer (4KB)

static constexpr int TM2 = 128;
static constexpr int TN2 = 128;
static constexpr int BK2 = 32;
static constexpr int NWARPS2 = 8;

template <typename T>
__global__ void wmma_large_tile_bk32_conv2d_3x3_kernel(
    T* __restrict__ output,
    T const* __restrict__ input,
    T const* __restrict__ filter,
    size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;
    size_t const N = C_out;
    size_t const K = C_in * 9;

    size_t const block_m = blockIdx.x * TM2;
    size_t const block_n = blockIdx.y * TN2;

    int const warp_id = threadIdx.x / 32;
    // 4 rows x 2 cols of warps
    int const warp_row = warp_id / 2;  // 0..3
    int const warp_col = warp_id % 2;  // 0..1

    __shared__ half smem_A[TM2][BK2];    // 8KB
    __shared__ half smem_B[BK2][TN2];    // 8KB

    // Each warp: 32x64 output (2x4 WMMA tiles)
    // Actually: 4 warp rows x 2 warp cols = covers 128x128
    // Each warp covers 32x64 ... no, let's do 32x32 per warp for simplicity
    // 8 warps: need to cover 128x128 = 16384 elements
    // 4 rows x 2 cols, each 32x64? That's 128x128. Each warp: 2x4 WMMA tiles
    // Let's do: warp_row covers 32 of M, warp_col covers 64 of N
    // So each warp does 2 WMMA in M, 4 WMMA in N = 8 tiles

    // Actually simpler: 4x2 layout, each warp 32x64
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int const nthreads = NWARPS2 * 32;

    for (size_t k_offset = 0; k_offset < K; k_offset += BK2) {
        // Load A: TM2 x BK2 = 128*32 = 4096 elements, 256 threads => 16 each
        for (int idx = threadIdx.x; idx < TM2 * BK2; idx += nthreads) {
            int const tm = idx / BK2;
            int const tk = idx % BK2;
            size_t const m = block_m + tm;
            size_t const k = k_offset + tk;

            half val = __float2half(0.0f);
            if (m < M && k < K) {
                size_t const oh = m / out_W;
                size_t const ow = m % out_W;
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                size_t const fy = fpos / 3;
                size_t const fx = fpos % 3;
                val = __float2half(input[ic * H * W + (oh + fy) * W + (ow + fx)]);
            }
            smem_A[tm][tk] = val;
        }

        // Load B: BK2 x TN2 = 32*128 = 4096 elements
        for (int idx = threadIdx.x; idx < BK2 * TN2; idx += nthreads) {
            int const tk = idx / TN2;
            int const tn = idx % TN2;
            size_t const k = k_offset + tk;
            size_t const n = block_n + tn;

            half val = __float2half(0.0f);
            if (k < K && n < N) {
                size_t const ic = k / 9;
                size_t const fpos = k % 9;
                size_t const fy = fpos / 3;
                size_t const fx = fpos % 3;
                val = __float2half(filter[n * C_in * 9 + ic * 9 + fy * 3 + fx]);
            }
            smem_B[tk][tn] = val;
        }

        __syncthreads();

        int const wm = warp_row * 32;
        int const wn = warp_col * 64;

        // Process BK2=32 as two BK=16 steps
        for (int kk = 0; kk < BK2; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi], &smem_A[wm + mi * 16][kk], BK2);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &smem_B[kk][wn + ni * 16], TN2);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        __syncthreads();
    }

    // Store results - each warp writes its 32x64 chunk directly to global
    // Use smem as staging: 32x64 floats = 8KB, but we only have limited smem
    // Instead, store per-warp through a 16x16 smem tile (1KB)
    // Each warp stores its 2x4 = 8 WMMA tiles one at a time
    __shared__ float smem_store[16][16];  // tiny buffer, reused

    int const wm = warp_row * 32;
    int const wn = warp_col * 64;

    for (int w = 0; w < NWARPS2; w++) {
        if (warp_id == w) {
            for (int mi = 0; mi < 2; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    wmma::store_matrix_sync(&smem_store[0][0], c_frag[mi][ni], 16, wmma::mem_row_major);
                    __syncwarp();
                    int const lane = threadIdx.x % 32;
                    // 32 threads write 16x16=256 values = 8 per thread
                    for (int idx = lane; idx < 256; idx += 32) {
                        int const r = idx / 16;
                        int const c = idx % 16;
                        size_t const gm = block_m + wm + mi * 16 + r;
                        size_t const gn = block_n + wn + ni * 16 + c;
                        if (gm < M && gn < N) {
                            size_t const oh = gm / out_W;
                            size_t const ow = gm % out_W;
                            output[gn * out_H * out_W + oh * out_W + ow] = static_cast<T>(smem_store[r][c]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
void launch_wmma_large_tile_bk32_conv2d_3x3(
    T* d_output, T const* d_input, T const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;
    size_t const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM2 - 1) / TM2),
        (unsigned int)((C_out + TN2 - 1) / TN2));
    dim3 const block(NWARPS2 * 32);

    wmma_large_tile_bk32_conv2d_3x3_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
