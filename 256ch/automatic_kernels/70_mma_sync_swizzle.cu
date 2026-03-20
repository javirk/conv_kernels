#pragma once

#include <cuda_fp16.h>

// Exp 70: mma.sync m16n8k16 + XOR swizzle + ldmatrix
// Drop WMMA entirely for native Ampere PTX instructions:
// 1. Zero bank conflicts via XOR swizzle + ldmatrix (vs 2-way with WMMA PAD=8)
// 2. Fewer fragment registers (~100 vs ~126)
// 3. No padding → less smem → room for future double buffer

static constexpr int TM70 = 128;
static constexpr int TN70 = 128;
static constexpr int BK70 = 64;
static constexpr int NWARPS70 = 8;

// XOR swizzle: permute 8-half groups by row index to eliminate bank conflicts.
// bank = col_swizzled/2 % 32. Different rows access different groups → different banks.
__device__ __forceinline__ int swizzle_off(int row, int col, int stride) {
    return row * stride + (((col >> 3) ^ (row & 7)) << 3 | (col & 7));
}

// Convert generic pointer to shared memory address for PTX instructions
__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ void __launch_bounds__(256, 2)
mma_sync_swizzle_conv2d_3x3_kernel(
    half* __restrict__ output,
    half const* __restrict__ input,
    half const* __restrict__ filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    int const M = out_H * out_W;
    int const N = C_out;
    int const K = C_in * 9;
    int const HW = H * W;

    int const block_m = blockIdx.x * TM70;
    int const block_n = blockIdx.y * TN70;

    int const lane = threadIdx.x & 31;
    int const warp_id = threadIdx.x >> 5;
    int const warp_row = warp_id >> 1;  // 0-3
    int const warp_col = warp_id & 1;   // 0-1

    // Spatial position for A loads (fixed per thread)
    int const my_tm = threadIdx.x & (TM70 - 1);
    int const my_m = block_m + my_tm;
    int const my_oh = (my_m < M) ? (my_m / out_W) : -1;
    int const my_ow = (my_m < M) ? (my_m % out_W) : -1;
    int const my_base = (my_oh >= 0) ? my_oh * W + my_ow : 0;
    int const valid = (my_oh >= 0) ? 1 : 0;

    // Shared memory (no padding — XOR swizzle handles bank conflicts)
    extern __shared__ char smem_raw[];
    half* smem_A = reinterpret_cast<half*>(smem_raw);                              // [128][64]
    half* smem_B = reinterpret_cast<half*>(smem_raw + TM70 * BK70 * sizeof(half)); // [64][128]

    // C accumulators: 2 M-tiles × 8 N-tiles × 4 floats
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;
    float c40 = 0, c41 = 0, c42 = 0, c43 = 0;
    float c50 = 0, c51 = 0, c52 = 0, c53 = 0;
    float c60 = 0, c61 = 0, c62 = 0, c63 = 0;
    float c70 = 0, c71 = 0, c72 = 0, c73 = 0;
    float c80 = 0, c81 = 0, c82 = 0, c83 = 0;
    float c90 = 0, c91 = 0, c92 = 0, c93 = 0;
    float ca0 = 0, ca1 = 0, ca2 = 0, ca3 = 0;
    float cb0 = 0, cb1 = 0, cb2 = 0, cb3 = 0;
    float cc0 = 0, cc1 = 0, cc2 = 0, cc3 = 0;
    float cd0 = 0, cd1 = 0, cd2 = 0, cd3 = 0;
    float ce0 = 0, ce1 = 0, ce2 = 0, ce3 = 0;
    float cf0 = 0, cf1 = 0, cf2 = 0, cf3 = 0;

    // Use array for cleaner indexing in store phase
    float* c_arr = &c00;  // all 64 floats are contiguous? No, they're separate variables.
    // We'll handle the store with a helper below.

    int const total_k_iters = (K + BK70 - 1) / BK70;
    int const wm = warp_row * 32;
    int const wn = warp_col * 64;
    int const nthreads = NWARPS70 * 32;

    for (int ki = 0; ki < total_k_iters; ki++) {
        int const k_base = ki * BK70;

        // ===== Load A: im2col input → smem_A with XOR swizzle =====
        #pragma unroll 16
        for (int idx = threadIdx.x; idx < TM70 * BK70; idx += nthreads) {
            int const tm = idx & (TM70 - 1);
            int const tk = idx >> 7;
            int const k = k_base + tk;
            half val = __float2half(0.0f);
            if (valid && k < K) {
                int ic = k / 9;
                int fpos = k - ic * 9;
                int fy = fpos / 3;
                int fx = fpos - fy * 3;
                val = __ldg(&input[ic * HW + fy * W + fx + my_base]);
            }
            smem_A[swizzle_off(tm, tk, BK70)] = val;
        }

        // ===== Load B: filter → smem_B with XOR swizzle =====
        #pragma unroll 16
        for (int idx = threadIdx.x; idx < BK70 * TN70; idx += nthreads) {
            int const tk = idx & (BK70 - 1);
            int const tn = idx >> 6;
            int const k = k_base + tk;
            int const n = block_n + tn;
            half val = __float2half(0.0f);
            if (k < K && n < N) {
                val = __ldg(&filter[n * K + k]);
            }
            smem_B[swizzle_off(tk, tn, TN70)] = val;
        }
        __syncthreads();

        // ===== Compute: mma.sync m16n8k16 =====
        #pragma unroll
        for (int kk = 0; kk < BK70; kk += 16) {
            // Load A fragments for both M-tiles via ldmatrix.x4
            // Thread lane loads row (lane & 15), col group (lane >= 16 ? kk+8 : kk)
            uint32_t a0_0, a1_0, a2_0, a3_0;  // A fragment for mi=0
            uint32_t a0_1, a1_1, a2_1, a3_1;  // A fragment for mi=1
            {
                // mi=0: rows wm..wm+15
                int a_row = wm + (lane & 15);
                int a_col = kk + ((lane >> 4) << 3);
                uint32_t addr = smem_addr(&smem_A[swizzle_off(a_row, a_col, BK70)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a0_0), "=r"(a1_0), "=r"(a2_0), "=r"(a3_0) : "r"(addr));
            }
            {
                // mi=1: rows wm+16..wm+31
                int a_row = wm + 16 + (lane & 15);
                int a_col = kk + ((lane >> 4) << 3);
                uint32_t addr = smem_addr(&smem_A[swizzle_off(a_row, a_col, BK70)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a0_1), "=r"(a1_1), "=r"(a2_1), "=r"(a3_1) : "r"(addr));
            }

            // For each of 8 N-tiles: load B, compute for both M-tiles
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                // Load B fragment via ldmatrix.x2.trans
                // B is [BK][TN] in smem, load 16×8 block at (kk, wn+ni*8)
                uint32_t b0, b1;
                {
                    int b_row = kk + (lane & 15);
                    int b_col = wn + ni * 8;
                    uint32_t addr = smem_addr(&smem_B[swizzle_off(b_row, b_col, TN70)]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                        : "=r"(b0), "=r"(b1) : "r"(addr));
                }

                // mma.sync for mi=0
                // Map (mi=0, ni) to accumulator variables
                #define MMA_SYNC(d0, d1, d2, d3, a0, a1, a2, a3) \
                    asm volatile( \
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) \
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                          "r"(b0), "r"(b1), \
                          "f"(d0), "f"(d1), "f"(d2), "f"(d3))

                // mi=0, ni=0..7
                if (ni == 0) { MMA_SYNC(c00,c01,c02,c03, a0_0,a1_0,a2_0,a3_0); }
                if (ni == 1) { MMA_SYNC(c10,c11,c12,c13, a0_0,a1_0,a2_0,a3_0); }
                if (ni == 2) { MMA_SYNC(c20,c21,c22,c23, a0_0,a1_0,a2_0,a3_0); }
                if (ni == 3) { MMA_SYNC(c30,c31,c32,c33, a0_0,a1_0,a2_0,a3_0); }
                if (ni == 4) { MMA_SYNC(c40,c41,c42,c43, a0_0,a1_0,a2_0,a3_0); }
                if (ni == 5) { MMA_SYNC(c50,c51,c52,c53, a0_0,a1_0,a2_0,a3_0); }
                if (ni == 6) { MMA_SYNC(c60,c61,c62,c63, a0_0,a1_0,a2_0,a3_0); }
                if (ni == 7) { MMA_SYNC(c70,c71,c72,c73, a0_0,a1_0,a2_0,a3_0); }

                // mi=1, ni=0..7
                if (ni == 0) { MMA_SYNC(c80,c81,c82,c83, a0_1,a1_1,a2_1,a3_1); }
                if (ni == 1) { MMA_SYNC(c90,c91,c92,c93, a0_1,a1_1,a2_1,a3_1); }
                if (ni == 2) { MMA_SYNC(ca0,ca1,ca2,ca3, a0_1,a1_1,a2_1,a3_1); }
                if (ni == 3) { MMA_SYNC(cb0,cb1,cb2,cb3, a0_1,a1_1,a2_1,a3_1); }
                if (ni == 4) { MMA_SYNC(cc0,cc1,cc2,cc3, a0_1,a1_1,a2_1,a3_1); }
                if (ni == 5) { MMA_SYNC(cd0,cd1,cd2,cd3, a0_1,a1_1,a2_1,a3_1); }
                if (ni == 6) { MMA_SYNC(ce0,ce1,ce2,ce3, a0_1,a1_1,a2_1,a3_1); }
                if (ni == 7) { MMA_SYNC(cf0,cf1,cf2,cf3, a0_1,a1_1,a2_1,a3_1); }
                #undef MMA_SYNC
            }
        }
        __syncthreads();
    }

    // ===== Store output: accumulators → smem → coalesced global writes =====
    // Reuse smem_A for store buffer: each warp gets 16*8 = 128 halfs
    half* store_buf = smem_A + warp_id * 128;
    int const out_HW = out_H * out_W;

    // mma m16n8k16 C layout per thread:
    //   c[0] → row (lane/4),     col (lane%4)*2
    //   c[1] → row (lane/4),     col (lane%4)*2 + 1
    //   c[2] → row (lane/4 + 8), col (lane%4)*2
    //   c[3] → row (lane/4 + 8), col (lane%4)*2 + 1
    int const sr0 = lane >> 2;         // 0-7
    int const sr1 = sr0 + 8;          // 8-15
    int const sc0 = (lane & 3) << 1;  // 0,2,4,6
    int const sc1 = sc0 + 1;          // 1,3,5,7

    // Helper to store one 16x8 tile
    #define STORE_TILE(mi, ni, v0, v1, v2, v3) do { \
        store_buf[sr0 * 8 + sc0] = __float2half(v0); \
        store_buf[sr0 * 8 + sc1] = __float2half(v1); \
        store_buf[sr1 * 8 + sc0] = __float2half(v2); \
        store_buf[sr1 * 8 + sc1] = __float2half(v3); \
        __syncwarp(); \
        for (int _idx = lane; _idx < 128; _idx += 32) { \
            int _r = _idx & 15; \
            int _c = _idx >> 4; \
            int _gm = block_m + wm + (mi)*16 + _r; \
            int _gn = block_n + wn + (ni)*8 + _c; \
            if (_gm < M && _gn < N) { \
                int _oh = _gm / out_W; \
                int _ow = _gm - _oh * out_W; \
                output[_gn * out_HW + _oh * out_W + _ow] = store_buf[_r * 8 + _c]; \
            } \
        } \
        __syncwarp(); \
    } while(0)

    STORE_TILE(0, 0, c00, c01, c02, c03);
    STORE_TILE(0, 1, c10, c11, c12, c13);
    STORE_TILE(0, 2, c20, c21, c22, c23);
    STORE_TILE(0, 3, c30, c31, c32, c33);
    STORE_TILE(0, 4, c40, c41, c42, c43);
    STORE_TILE(0, 5, c50, c51, c52, c53);
    STORE_TILE(0, 6, c60, c61, c62, c63);
    STORE_TILE(0, 7, c70, c71, c72, c73);
    STORE_TILE(1, 0, c80, c81, c82, c83);
    STORE_TILE(1, 1, c90, c91, c92, c93);
    STORE_TILE(1, 2, ca0, ca1, ca2, ca3);
    STORE_TILE(1, 3, cb0, cb1, cb2, cb3);
    STORE_TILE(1, 4, cc0, cc1, cc2, cc3);
    STORE_TILE(1, 5, cd0, cd1, cd2, cd3);
    STORE_TILE(1, 6, ce0, ce1, ce2, ce3);
    STORE_TILE(1, 7, cf0, cf1, cf2, cf3);
    #undef STORE_TILE
}

void launch_mma_sync_swizzle_conv2d_3x3(
    half* d_output, half const* d_input, half const* d_filter,
    size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t stream)
{
    int const out_H = (int)H - 2;
    int const out_W = (int)W - 2;
    int const M = out_H * out_W;

    dim3 const grid(
        (unsigned int)((M + TM70 - 1) / TM70),
        (unsigned int)((C_out + TN70 - 1) / TN70));
    dim3 const block(NWARPS70 * 32);

    size_t smem_size = (TM70 * BK70 + BK70 * TN70) * sizeof(half);
    mma_sync_swizzle_conv2d_3x3_kernel<<<grid, block, smem_size, stream>>>(
        d_output, d_input, d_filter, (int)C_in, (int)C_out, (int)H, (int)W);
    CHECK_LAST_CUDA_ERROR();
}
