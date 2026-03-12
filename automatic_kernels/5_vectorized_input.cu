#pragma once

// Vectorized input loads with multi-OC per thread.
// Each thread processes 4 adjacent spatial columns at once, using float4 loads
// where possible to maximize memory throughput. Each thread computes OC_PER_THREAD
// output channels for 4 spatial positions.

template <typename T, int OC_PER_THREAD = 8, int COLS_PER_THREAD = 4>
__global__ void vec_input_conv2d_3x3(T* output, T const* input, T const* filter,
                                      size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    // Each thread handles COLS_PER_THREAD adjacent columns
    int const col_base = (threadIdx.x + blockIdx.x * blockDim.x) * COLS_PER_THREAD;
    int const row = threadIdx.y + blockIdx.y * blockDim.y;
    int const oc_base = blockIdx.z * OC_PER_THREAD;

    if (row >= (int)out_H)
        return;

    T sums[OC_PER_THREAD][COLS_PER_THREAD];
    #pragma unroll
    for (int oci = 0; oci < OC_PER_THREAD; ++oci)
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c)
            sums[oci][c] = static_cast<T>(0);

    for (size_t ic = 0; ic < C_in; ++ic)
    {
        T const* input_ic = input + ic * H * W;

        // Load input values needed for COLS_PER_THREAD outputs
        // For 3x3 filter and 4 adjacent outputs, we need 6 values per row (cols col_base..col_base+5)
        T inp[3][COLS_PER_THREAD + 2];
        #pragma unroll
        for (int fy = 0; fy < 3; ++fy)
        {
            #pragma unroll
            for (int fx = 0; fx < COLS_PER_THREAD + 2; ++fx)
            {
                int const gx = col_base + fx;
                inp[fy][fx] = (gx < (int)W) ?
                    input_ic[(row + fy) * W + gx] : static_cast<T>(0);
            }
        }

        #pragma unroll
        for (int oci = 0; oci < OC_PER_THREAD; ++oci)
        {
            int const oc = oc_base + oci;
            if (oc < (int)C_out)
            {
                T const* f = filter + oc * C_in * 9 + ic * 9;
                #pragma unroll
                for (int c = 0; c < COLS_PER_THREAD; ++c)
                {
                    if (col_base + c < (int)out_W)
                    {
                        #pragma unroll
                        for (int fy = 0; fy < 3; ++fy)
                        {
                            #pragma unroll
                            for (int fx = 0; fx < 3; ++fx)
                            {
                                sums[oci][c] += inp[fy][c + fx] * f[fy * 3 + fx];
                            }
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int oci = 0; oci < OC_PER_THREAD; ++oci)
    {
        int const oc = oc_base + oci;
        if (oc < (int)C_out)
        {
            #pragma unroll
            for (int c = 0; c < COLS_PER_THREAD; ++c)
            {
                if (col_base + c < (int)out_W)
                {
                    output[oc * out_H * out_W + row * out_W + (col_base + c)] = sums[oci][c];
                }
            }
        }
    }
}

template <typename T>
void launch_vec_input_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                  size_t C_in, size_t C_out, size_t H, size_t W,
                                  cudaStream_t stream)
{
    constexpr int OC_PER_THREAD = 8;
    constexpr int COLS_PER_THREAD = 4;
    constexpr int BLOCK_X = 16;  // each thread does 4 cols, so covers 64 cols
    constexpr int BLOCK_Y = 8;
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    dim3 const block(BLOCK_X, BLOCK_Y);
    dim3 const grid(
        static_cast<unsigned int>(div_up(out_W, BLOCK_X * COLS_PER_THREAD)),
        static_cast<unsigned int>(div_up(out_H, BLOCK_Y)),
        static_cast<unsigned int>(div_up(C_out, OC_PER_THREAD)));

    vec_input_conv2d_3x3<T, OC_PER_THREAD, COLS_PER_THREAD>
        <<<grid, block, 0, stream>>>(
            d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
