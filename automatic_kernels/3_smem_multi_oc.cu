#pragma once

// Combine shared memory input tiling with multi-OC-per-thread.
// Each thread block loads an input tile into SMEM once per input channel,
// and each thread computes OC_PER_THREAD output channels from SMEM.
// This amortizes both global memory reads AND sync overhead.

template <typename T, int TILE_W = 32, int TILE_H = 16, int OC_PER_THREAD = 4>
__global__ void smem_multi_oc_conv2d_3x3(T* output, T const* input, T const* filter,
                                          size_t C_in, size_t C_out, size_t H, size_t W)
{
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    int const tx = threadIdx.x;
    int const ty = threadIdx.y;
    int const out_col = blockIdx.x * TILE_W + tx;
    int const out_row = blockIdx.y * TILE_H + ty;
    int const oc_base = blockIdx.z * OC_PER_THREAD;

    constexpr int SMEM_W = TILE_W + 2;
    constexpr int SMEM_H = TILE_H + 2;
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    T sums[OC_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OC_PER_THREAD; ++i)
        sums[i] = static_cast<T>(0);

    int const tid = ty * TILE_W + tx;
    int const threads_per_block = TILE_W * TILE_H;
    int const num_elements = SMEM_H * SMEM_W;

    for (size_t ic = 0; ic < C_in; ++ic)
    {
        T const* input_ic = input + ic * H * W;

        // Cooperatively load input tile into SMEM
        for (int idx = tid; idx < num_elements; idx += threads_per_block)
        {
            int const sy = idx / SMEM_W;
            int const sx = idx % SMEM_W;
            int const gy = blockIdx.y * TILE_H + sy;
            int const gx = blockIdx.x * TILE_W + sx;
            smem[sy * SMEM_W + sx] = (gy < (int)H && gx < (int)W) ?
                input_ic[gy * W + gx] : static_cast<T>(0);
        }

        __syncthreads();

        if (out_row < (int)out_H && out_col < (int)out_W)
        {
            // Read input patch from SMEM once
            T inp[9];
            #pragma unroll
            for (int fy = 0; fy < 3; ++fy)
            {
                #pragma unroll
                for (int fx = 0; fx < 3; ++fx)
                {
                    inp[fy * 3 + fx] = smem[(ty + fy) * SMEM_W + (tx + fx)];
                }
            }

            // Accumulate for all output channels
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

        __syncthreads();
    }

    if (out_row < (int)out_H && out_col < (int)out_W)
    {
        #pragma unroll
        for (int oci = 0; oci < OC_PER_THREAD; ++oci)
        {
            int const oc = oc_base + oci;
            if (oc < (int)C_out)
            {
                output[oc * out_H * out_W + out_row * out_W + out_col] = sums[oci];
            }
        }
    }
}

template <typename T>
void launch_smem_multi_oc_conv2d_3x3(T* d_output, T const* d_input, T const* d_filter,
                                      size_t C_in, size_t C_out, size_t H, size_t W,
                                      cudaStream_t stream)
{
    constexpr int TILE_W = 32;
    constexpr int TILE_H = 16;
    constexpr int OC_PER_THREAD = 4;
    size_t const out_H = H - 2;
    size_t const out_W = W - 2;

    dim3 const block(TILE_W, TILE_H);
    dim3 const grid(
        static_cast<unsigned int>(div_up(out_W, TILE_W)),
        static_cast<unsigned int>(div_up(out_H, TILE_H)),
        static_cast<unsigned int>(div_up(C_out, OC_PER_THREAD)));

    size_t const smem_size = (TILE_W + 2) * (TILE_H + 2) * sizeof(T);

    smem_multi_oc_conv2d_3x3<T, TILE_W, TILE_H, OC_PER_THREAD>
        <<<grid, block, smem_size, stream>>>(
            d_output, d_input, d_filter, C_in, C_out, H, W);
    CHECK_LAST_CUDA_ERROR();
}
