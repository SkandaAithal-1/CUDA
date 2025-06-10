#include <iostream>
#include "kernels.cuh"

__global__ void element_wise_square(const float* x, float* y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        y[idx] = x[idx] * x[idx];
    }
}

void l2_norm(const float* x, float* y, int N)
{
    float *xd, *yd;

    cudaMalloc((void**)&xd, N * sizeof(float));
    cudaMalloc((void**)&yd, N * sizeof(float));

    cudaMemcpy(xd, x, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    element_wise_square<<<gridSize, blockSize>>>(xd, yd, N);
    cudaDeviceSynchronize();

    for (int size = N; size > 0; size >>= 1)
    {
        vector_sum<<<gridSize, blockSize>>>(yd, N);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(y, yd, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(xd);
    cudaFree(yd);
}

int main()
{
    int N = 8;
    float x[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float y;
    l2_norm(x, &y, N);

    std::cout << "L2 Norm: " << y << std::endl;
    return 0;
}