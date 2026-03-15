// Minimal test: verifies that chosen_kernels/conv_kernel.cu compiles and runs correctly.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "chosen_kernels/conv_kernel.cu"

void cpu_conv2d_3x3(float* output, float const* input, float const* filter,
                     int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    for (int oc = 0; oc < C_out; ++oc)
        for (int oh = 0; oh < out_H; ++oh)
            for (int ow = 0; ow < out_W; ++ow)
            {
                float sum = 0.0f;
                for (int ic = 0; ic < C_in; ++ic)
                    for (int fy = 0; fy < 3; ++fy)
                        for (int fx = 0; fx < 3; ++fx)
                            sum += input[ic * H * W + (oh + fy) * W + (ow + fx)]
                                 * filter[oc * C_in * 9 + ic * 9 + fy * 3 + fx];
                output[oc * out_H * out_W + oh * out_W + ow] = sum;
            }
}

bool test_config(int C_in, int C_out, int H, int W)
{
    int const out_H = H - 2;
    int const out_W = W - 2;
    size_t const in_size = C_in * H * W;
    size_t const filt_size = C_out * C_in * 9;
    size_t const out_size = C_out * out_H * out_W;

    std::vector<float> h_input(in_size), h_filter(filt_size);
    std::vector<float> h_output_cpu(out_size), h_output_gpu(out_size);

    srand(42);
    for (auto& v : h_input) v = (float)rand() / RAND_MAX - 0.5f;
    for (auto& v : h_filter) v = (float)rand() / RAND_MAX - 0.5f;

    cpu_conv2d_3x3(h_output_cpu.data(), h_input.data(), h_filter.data(), C_in, C_out, H, W);

    float *d_in, *d_filt, *d_out;
    cudaMalloc(&d_in, in_size * sizeof(float));
    cudaMalloc(&d_filt, filt_size * sizeof(float));
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMemcpy(d_in, h_input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filt, h_filter.data(), filt_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_conv2d_3x3(d_out, d_in, d_filt, C_in, C_out, H, W, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_output_gpu.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (size_t i = 0; i < out_size; ++i)
        max_err = fmaxf(max_err, fabsf(h_output_cpu[i] - h_output_gpu[i]));

    cudaFree(d_in);
    cudaFree(d_filt);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    // FP16 kernels have ~1e-2 tolerance, FP32 scalar ~1e-5
    float tol = (C_in >= 16 && C_out >= 16) ? 0.05f : 1e-3f;
    bool pass = max_err < tol;
    printf("  C_in=%3d C_out=%3d %dx%d -> max_err=%.5f %s\n",
           C_in, C_out, H, W, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int main()
{
    printf("Testing conv_kernel dispatcher...\n");
    bool all_pass = true;

    // Test each dispatch path
    all_pass &= test_config(8, 8, 32, 32);       // -> kernel 04
    all_pass &= test_config(4, 16, 16, 16);      // -> kernel 04 (C_in < 16)
    all_pass &= test_config(16, 16, 32, 32);     // -> kernel 38
    all_pass &= test_config(16, 32, 64, 64);     // -> kernel 38
    all_pass &= test_config(32, 32, 32, 32);     // -> kernel 39
    all_pass &= test_config(32, 64, 64, 64);     // -> kernel 39
    all_pass &= test_config(64, 64, 32, 32);     // -> kernel 40
    all_pass &= test_config(64, 64, 128, 128);   // -> kernel 40
    all_pass &= test_config(128, 128, 32, 32);   // -> kernel 40
    all_pass &= test_config(256, 256, 16, 16);   // -> kernel 40

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
