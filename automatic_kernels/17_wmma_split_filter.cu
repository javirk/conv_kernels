#pragma once

#include <mma.h>
using namespace nvcuda;

// Split-filter WMMA: iterate over (fy, fx) explicitly, WMMA over ic dimension.
// No divmod needed — A[m, ic] = input[ic, row_m+fy, col_m+fx] is simple addressing.
// B[ic, oc] = filter[oc, ic, fy, fx] is also straightforward.

constexpr int BM17_H = 8;
constexpr int BM17_W = 8;
constexpr int BM17 = BM17_H * BM17_W;  // 64
constexpr int BN17 = 64;
constexpr int BK17 = 16;   // chunk of C_in
constexpr int WM17 = 16;
constexpr int WN17 = 16;
constexpr int WK17 = 8;

__launch_bounds__(512, 1)
__global__ void wmma_split_filter_conv2d_kernel(
    float* __restrict__ output,
    float const* __restrict__ input,
    float const* __restrict__ filter,
    int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    int const tile_row = blockIdx.x * BM17_H;
    int const tile_col = blockIdx.y * BM17_W;
    int const block_n = blockIdx.z * BN17;

    __shared__ float As[2][BM17][BK17];
    __shared__ float Bs[2][BK17][BN17];

    int const warp_id = threadIdx.x / 32;
    int const warps_n = BN17 / WN17;  // 4
    int const warp_m = warp_id / warps_n;
    int const warp_n = warp_id % warps_n;

    wmma::fragment<wmma::accumulator, WM17, WN17, WK17, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int const tid = threadIdx.x;
    int const total_threads = blockDim.x;

    int const num_ic_iters = (C_in + BK17 - 1) / BK17;

    // Iterate over 3x3 filter positions
    #pragma unroll
    for (int fy = 0; fy < 3; ++fy)
    {
        #pragma unroll
        for (int fx = 0; fx < 3; ++fx)
        {
            // Load first ic chunk
            int ic_start = 0;
            {
                int const buf = 0;
                // Load A[BM17][BK17]: A[m, k] = input[(ic_start+k), row_m+fy, col_m+fx]
                for (int idx = tid; idx < BM17 * BK17; idx += total_threads)
                {
                    int const m = idx / BK17;
                    int const k = idx & (BK17 - 1);
                    int const local_row = m >> 3;
                    int const local_col = m & 7;
                    int const row = tile_row + local_row + fy;
                    int const col = tile_col + local_col + fx;
                    int const ic = ic_start + k;
                    As[buf][m][k] = (row < H && col < W && ic < C_in) ?
                        input[ic * H * W + row * W + col] : 0.0f;
                }
                // Load B[BK17][BN17]: B[k, n] = filter[(block_n+n), (ic_start+k), fy, fx]
                for (int idx = tid; idx < BK17 * BN17; idx += total_threads)
                {
                    int const k = idx / BN17;
                    int const n = idx & (BN17 - 1);
                    int const ic = ic_start + k;
                    int const oc = block_n + n;
                    Bs[buf][k][n] = (ic < C_in && oc < C_out) ?
                        filter[oc * C_in * 9 + ic * 9 + fy * 3 + fx] : 0.0f;
                }
            }
            __syncthreads();

            for (int ici = 0; ici < num_ic_iters; ++ici)
            {
                int const cur_buf = ici & 1;
                int const next_buf = 1 - cur_buf;

                // Prefetch next ic chunk
                if (ici + 1 < num_ic_iters)
                {
                    int const next_ic = (ici + 1) * BK17;
                    for (int idx = tid; idx < BM17 * BK17; idx += total_threads)
                    {
                        int const m = idx / BK17;
                        int const k = idx & (BK17 - 1);
                        int const local_row = m >> 3;
                        int const local_col = m & 7;
                        int const row = tile_row + local_row + fy;
                        int const col = tile_col + local_col + fx;
                        int const ic = next_ic + k;
                        As[next_buf][m][k] = (row < H && col < W && ic < C_in) ?
                            input[ic * H * W + row * W + col] : 0.0f;
                    }
                    for (int idx = tid; idx < BK17 * BN17; idx += total_threads)
                    {
                        int const k = idx / BN17;
                        int const n = idx & (BN17 - 1);
                        int const ic = next_ic + k;
                        int const oc = block_n + n;
                        Bs[next_buf][k][n] = (ic < C_in && oc < C_out) ?
                            filter[oc * C_in * 9 + ic * 9 + fy * 3 + fx] : 0.0f;
                    }
                }

                // Compute
                int const tile_m_off = warp_m * WM17;
                int const tile_n_off = warp_n * WN17;

                #pragma unroll
                for (int kk = 0; kk < BK17; kk += WK17)
                {
                    wmma::fragment<wmma::matrix_a, WM17, WN17, WK17,
                                   wmma::precision::tf32, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM17, WN17, WK17,
                                   wmma::precision::tf32, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, &As[cur_buf][tile_m_off][kk], BK17);
                    wmma::load_matrix_sync(b_frag, &Bs[cur_buf][kk][tile_n_off], BN17);
                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }

                __syncthreads();
            }
        }
    }

    // Store
    __shared__ float Cs[BM17][BN17];
    wmma::store_matrix_sync(&Cs[warp_m * WM17][warp_n * WN17], acc, BN17, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < BM17 * BN17; idx += total_threads)
    {
        int const m = idx / BN17;
        int const n = idx & (BN17 - 1);
        int const local_row = m >> 3;
        int const local_col = m & 7;
        int const out_row = tile_row + local_row;
        int const out_col = tile_col + local_col;
        int const oc = block_n + n;
        if (out_row < out_H && out_col < out_W && oc < C_out)
        {
            output[oc * out_H * out_W + out_row * out_W + out_col] = Cs[m][n];
        }
    }
}

template <typename T>
void launch_wmma_split_filter_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                           size_t C_in, size_t C_out, size_t H, size_t W,
                                           cudaStream_t stream)
{
    int const out_H = H - 2;
    int const out_W = W - 2;

    constexpr int num_warps = (BM17 / WM17) * (BN17 / WN17);
    dim3 const block(num_warps * 32);
    dim3 const grid(
        (out_H + BM17_H - 1) / BM17_H,
        (out_W + BM17_W - 1) / BM17_W,
        (C_out + BN17 - 1) / BN17);

    wmma_split_filter_conv2d_kernel<<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
