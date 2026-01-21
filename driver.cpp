#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

struct MemRefDescriptor2 {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

struct MemRefDescriptor0 {
    float *allocated;
    float *aligned;
    int64_t offset;
};

extern "C" void _mlir_ciface_main1(MemRefDescriptor0* result, MemRefDescriptor2* input);

int main() {
    const int iterations = 10;
    const int64_t N = 64; 
    const double total_flops = (2.0 * N * N * N) + (N * N) + (N * N); 
    
    std::cout << "Benchmarking main1 (64x64 GPU Chain)..." << std::endl;
    
    MemRefDescriptor0 resultDesc; 
    MemRefDescriptor2 argDesc;
    // Allocate device memory for scalar result
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    // Initialize result memory (optional)
    cudaMemset(d_result, 0, sizeof(float));
    resultDesc.allocated = d_result;
    resultDesc.aligned = d_result;
    resultDesc.offset = 0; 
    
    float* h_data = (float*)malloc(N * N * sizeof(float));
    for(int i=0; i<N*N; ++i) h_data[i] = 1.0f;
    
    float *d_input;
    cudaMalloc(&d_input, N * N * sizeof(float));
    
    cudaMemcpy(d_input, h_data, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    argDesc.allocated = d_input;
    argDesc.aligned = d_input;
    argDesc.offset = 0;
    argDesc.sizes[0] = N; argDesc.sizes[1] = N;
    argDesc.strides[0] = N; argDesc.strides[1] = 1;
    
    // Warmup
    _mlir_ciface_main1(&resultDesc, &argDesc);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        _mlir_ciface_main1(&resultDesc, &argDesc);
    }
    cudaDeviceSynchronize(); 
    auto end = std::chrono::high_resolution_clock::now();
    
    float h_result;
    // resultDesc.aligned is the device pointer to the scalar
    cudaMemcpy(&h_result, resultDesc.aligned, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Result: " << h_result << std::endl;
    
    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / iterations;
    double gflops = (total_flops / 1e9) / avg_time;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Workload:              64x64 (Add -> Matmul -> Add)" << std::endl;
    std::cout << "Total Operations:      " << total_flops / 1e9 << " GFLOP" << std::endl;
    std::cout << "Average Execution Time: " << avg_time * 1000.0 << " ms" << std::endl;
    std::cout << "Achieved Performance:  " << gflops << " GFLOPS" << std::endl;
    
    cudaFree(d_input);
    // Note: If the function allocated the result, we should free it too.
    if (resultDesc.allocated) cudaFree(resultDesc.allocated);
    free(h_data);
    
    return 0;
}
