#pragma once

// Shared memory tiling for conv2d 3x3
// Each thread block loads a tile of input into shared memory (with halo for 3x3 filter),
// then computes output from shared memory instead of global memory.
// This reduces redundant global memory reads by up to 9x.

template <typename T, int TILE_W = 32, int TILE_H = 32>
__global__ void shared_mem_conv2d_3x3(T* output, T const* input, T const* filter,
                                       size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    // Output coordinates
    int const tx = threadIdx.x;
    int const ty = threadIdx.y;
    int const out_col = blockIdx.x * TILE_W + tx;
    int const out_row = blockIdx.y * TILE_H + ty;
    int const oc = blockIdx.z;

    // Shared memory tile includes halo: TILE + 2 in each spatial dimension
    constexpr int SMEM_W = TILE_W + 2;
    constexpr int SMEM_H = TILE_H + 2;
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    T sum = static_cast<T>(0);

    for (size_t ic = 0; ic < C_in; ++ic)
    {
        T const* input_ic = input + ic * H * W;

        // Cooperatively load the input tile (with halo) into shared memory
        // Each thread loads one element, but we need SMEM_H * SMEM_W elements
        int const num_elements = SMEM_H * SMEM_W;
        int const threads_per_block = TILE_W * TILE_H;
        int const tid = ty * TILE_W + tx;

        for (int idx = tid; idx < num_elements; idx += threads_per_block)
        {
            int const sy = idx / SMEM_W;
            int const sx = idx % SMEM_W;
            int const gy = blockIdx.y * TILE_H + sy;
            int const gx = blockIdx.x * TILE_W + sx;

            if (gy < (int)H && gx < (int)W)
                smem[sy * SMEM_W + sx] = input_ic[gy * W + gx];
            else
                smem[sy * SMEM_W + sx] = static_cast<T>(0);
        }

        __syncthreads();

        // Compute convolution from shared memory
        if (out_row < (int)out_H && out_col < (int)out_W)
        {
            T const* filter_oc_ic = filter + oc * C_in * 9 + ic * 9;
            #pragma unroll
            for (int fy = 0; fy < 3; ++fy)
            {
                #pragma unroll
                for (int fx = 0; fx < 3; ++fx)
                {
                    sum += smem[(ty + fy) * SMEM_W + (tx + fx)] * filter_oc_ic[fy * 3 + fx];
                }
            }
        }

        __syncthreads();
    }

    if (out_row < (int)out_H && out_col < (int)out_W)
    {
        output[oc * out_H * out_W + out_row * out_W + out_col] = sum;
    }
}

template <typename T>
void launch_shared_mem_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                   size_t C_in, size_t C_out, size_t H, size_t W,
                                   cudaStream_t stream)
{
    constexpr int TILE_W = 32;
    constexpr int TILE_H = 32;
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    dim3 const block(TILE_W, TILE_H);
    dim3 const grid(
        static_cast<unsigned int>(div_up(out_W, TILE_W)),
        static_cast<unsigned int>(div_up(out_H, TILE_H)),
        static_cast<unsigned int>(C_out));

    size_t const smem_size = (TILE_W + 2) * (TILE_H + 2) * sizeof(T);

    shared_mem_conv2d_3x3<T, TILE_W, TILE_H><<<grid, block, smem_size, stream>>>(
        d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
