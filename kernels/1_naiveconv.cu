#pragma once

template <typename T>
__global__ void naive_conv2d_3x3(T* output, T const* input, T const* filter,
                                 size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H{H - 2};
    size_t const out_W{W - 2};

    size_t const col{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const row{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const oc{blockIdx.z};

    if (row < out_H && col < out_W && oc < C_out)
    {
        T sum{static_cast<T>(0)};
        for (size_t ic{0}; ic < C_in; ++ic)
        {
            #pragma unroll
            for (int fy{0}; fy < 3; ++fy)
            {
                #pragma unroll
                for (int fx{0}; fx < 3; ++fx)
                {
                    sum += input[ic * H * W + (row + fy) * W + (col + fx)] *
                           filter[oc * C_in * 9 + ic * 9 + fy * 3 + fx];
                }
            }
        }
        output[oc * out_H * out_W + row * out_W + col] = sum;
    }
}

template <typename T>
void launch_naive_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                             size_t C_in, size_t C_out, size_t H, size_t W,
                             cudaStream_t stream)
{
    constexpr size_t BLOCK_SIZE_X{16};
    constexpr size_t BLOCK_SIZE_Y{16};
    size_t const out_H{H - 2};
    size_t const out_W{W - 2};
    dim3 const block_size{BLOCK_SIZE_X, BLOCK_SIZE_Y};
    dim3 const grid_size{
        static_cast<unsigned int>(div_up(out_W, BLOCK_SIZE_X)),
        static_cast<unsigned int>(div_up(out_H, BLOCK_SIZE_Y)),
        static_cast<unsigned int>(C_out)};
    naive_conv2d_3x3<<<grid_size, block_size, 0, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
