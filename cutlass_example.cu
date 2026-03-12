#include <iostream>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;

    // Define CUTLASS GEMM
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor
    >;

    // Host data (all ones)
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N, 0.0f);

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // GEMM: C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;

    Gemm gemm_op;
    Gemm::Arguments args(
        {M, N, K},
        {d_A, M},   // A with leading dimension
        {d_B, K},   // B with leading dimension
        {d_C, M},   // C source
        {d_C, M},   // C destination
        {alpha, beta}
    );

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed: " << cutlassGetStatusString(status) << std::endl;
        return 1;
    }
    cudaDeviceSynchronize();

    // Copy back and verify
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        if (h_C[i] != static_cast<float>(K)) {
            correct = false;
            break;
        }
    }

    std::cout << "C[0] = " << h_C[0] << " (expected " << K << ")" << std::endl;
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return correct ? 0 : 1;
}
