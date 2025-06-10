#include <iostream>
#include "kernels.cuh"

__global__ void matrix_addition(const float *A, const float *B, float *C, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

int main()
{
    const int M = 1024;
    const int N = 1024;
    float A[M * N], B[M * N], C[M * N];
    float *Ad, *Bd, *Cd;
    size_t size = M * N * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void **)&Ad, size);
    cudaMalloc((void **)&Bd, size);
    cudaMalloc((void **)&Cd, size);

    // Copy data from host to device
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, C, size, cudaMemcpyHostToDevice);

    // Execution parameter configuration
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil((float)M / blockDim.x), ceil((float)N / blockDim.y));

    // Launch kernel
    matrix_addition<<<gridDim, blockDim>>>(Ad, Bd, Cd, M, N);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    return 0;
}