
template <typename T>
bool verify_conv2d_implementation(
    std::function<void(T*, T const*, T const*, size_t, size_t, size_t, size_t,
                       cudaStream_t)>
        conv_function,
    size_t C_in, size_t C_out, size_t H, size_t W)
{
    std::mt19937 gen{0};
    cudaStream_t stream;
    size_t const out_H{H - 2};
    size_t const out_W{W - 2};
    size_t const input_size{C_in * H * W};
    size_t const filter_size{C_out * C_in * 9};
    size_t const output_size{C_out * out_H * out_W};

    std::vector<T> input(input_size);
    std::vector<T> filter(filter_size);
    std::vector<T> output(output_size, static_cast<T>(0));
    std::vector<T> output_ref(output_size, static_cast<T>(0));

    // Initialize input and filter with random values
    std::uniform_real_distribution<T> dist(-1.0f, 1.0f);
    for (auto& v : input)
        v = dist(gen);
    for (auto& v : filter)
        v = dist(gen);

    // Compute reference output
    for (size_t oc{0}; oc < C_out; ++oc)
    {
        for (size_t r{0}; r < out_H; ++r)
        {
            for (size_t c{0}; c < out_W; ++c)
            {
                T sum{static_cast<T>(0)};
                for (size_t ic{0}; ic < C_in; ++ic)
                {
                    for (int fy{0}; fy < 3; ++fy)
                    {
                        for (int fx{0}; fx < 3; ++fx)
                        {
                            sum +=
                                input[ic * H * W + (r + fy) * W + (c + fx)] *
                                filter[oc * C_in * 9 + ic * 9 + fy * 3 + fx];
                        }
                    }
                }
                output_ref[oc * out_H * out_W + r * out_W + c] = sum;
            }
        }
    }

    // Run the convolution kernel
    // Allocate device memory
    T* d_input;
    T* d_output;
    T* d_filter;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filter_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), input_size * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filter, filter.data(),
                                filter_size * sizeof(T),
                                cudaMemcpyHostToDevice));

    conv_function(d_output, d_input, d_filter, C_in, C_out, H, W, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output,
                                output_size * sizeof(T),
                                cudaMemcpyDeviceToHost));

    bool correct{true};
    for (size_t i{0}; i < output_size; ++i)
    {
        if (std::abs(output[i] - output_ref[i]) > static_cast<T>(1e-3))
        {
            correct = false;
            break;
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_filter));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return correct;
}


