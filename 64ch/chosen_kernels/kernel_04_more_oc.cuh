#pragma once

// Kernel 04: More OC per thread. Best for C_in < 16.
// NCHW layout, float32. Simple scalar kernel.

template <typename T, int OC_PER_THREAD = 8>
__global__ void more_oc_conv2d_3x3(T* output, T const* input, T const* filter,
                                    size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    int const col = threadIdx.x + blockIdx.x * blockDim.x;
    int const row = threadIdx.y + blockIdx.y * blockDim.y;
    int const oc_base = blockIdx.z * OC_PER_THREAD;

    if (row >= (int)out_H || col >= (int)out_W)
        return;

    T sums[OC_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OC_PER_THREAD; ++i)
        sums[i] = static_cast<T>(0);

    for (size_t ic = 0; ic < C_in; ++ic)
    {
        T inp[9];
        T const* input_ic = input + ic * H * W;
        #pragma unroll
        for (int fy = 0; fy < 3; ++fy)
        {
            #pragma unroll
            for (int fx = 0; fx < 3; ++fx)
            {
                inp[fy * 3 + fx] = input_ic[(row + fy) * W + (col + fx)];
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
                for (int k = 0; k < 9; ++k)
                {
                    sums[oci] += inp[k] * f[k];
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
            output[oc * out_H * out_W + row * out_W + col] = sums[oci];
        }
    }
}

inline void launch_conv2d_04(float* d_output, float const* d_input, float const* d_filter,
                              size_t C_in, size_t C_out, size_t H, size_t W,
                              cudaStream_t stream)
{
    constexpr int OC_PER_THREAD = 8;
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 8;
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    dim3 const block(BLOCK_X, BLOCK_Y);
    dim3 const grid(
        static_cast<unsigned int>(div_up(out_W, BLOCK_X)),
        static_cast<unsigned int>(div_up(out_H, BLOCK_Y)),
        static_cast<unsigned int>(div_up(C_out, OC_PER_THREAD)));

    more_oc_conv2d_3x3<float, OC_PER_THREAD><<<grid, block, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
