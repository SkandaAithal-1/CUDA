#include <iostream>
#include "kernels.cuh"

__global__ void element_wise_vector_prod(const float *a, const float *b, float *res, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        res[idx] = a[idx] * b[idx];
    }
}

void dot_product(const float *a, const float *b, float *res, int N)
{
    float *ad, *bd, *temp_res;
    cudaMalloc((void **)&ad, N * sizeof(float));
    cudaMalloc((void **)&bd, N * sizeof(float));
    cudaMalloc((void **)&temp_res, N * sizeof(float));

    cudaMemcpy(ad, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    element_wise_vector_prod<<<gridSize, blockSize>>>(ad, bd, temp_res, N);
    cudaDeviceSynchronize();

    for (int size = N; size > 0; size >>= 1)
    {
        vector_sum<<<gridSize, blockSize>>>(temp_res, size);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(res, temp_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(temp_res);
}

int main()
{
    int N = 8;
    float a[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float res;
    dot_product(a, b, &res, N);
    std::cout << "Dot product: " << res << std::endl;
    return 0;
}