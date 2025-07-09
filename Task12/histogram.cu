#include<iostream>
#include "utils.h"

const int COARSE_SIZE = 32;

__global__ void histogram(const int* I, int* H, int N, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int privateH[M];
    for (int i=threadIdx.x; i<M; i+=blockDim.x)
    {
        privateH[i] = 0;
    }
    __syncthreads();
    
    for (int i=idx; i<N; i+=gridDim.x * blockDim.x)
    {
        atomicAdd(&privateH[I[i]], 1);
    }
    __syncthreads();

    for (int i=threadIdx.x; i<M; i+=blockDim.x)
    {
        atomicAdd(&H[i], privateH[i]);
    }
}

void launch_histogram(const int* I, int *H, int N, int M)
{
    int *dI, *dH;

    cudaMalloc((void**)&dI, N*sizeof(int));
    cudaMalloc((void**)&dH, M*sizeof(int));

    cudaMemcpy(dI, I, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dH, 0, M*sizeof(int));

    int blockSize = 256;
    int gridSize = 32;

    histogram<<<gridSize, blockSize>>>(dI, dH, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(H, dH, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dI);
    cudaFree(dH);
}

int main()
{
    return 0;
}