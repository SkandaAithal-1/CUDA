#include <iostream>
#include "kernels.cuh"

__global__ void transpose(const float *A, float *B, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        B[col * M + row] = A[row * N + col];
    }
}

void invoke_transpose(const float *A, float *B, int M, int N)
{
    float *Ad, *Bd;
    cudaMalloc((void**)&Ad, M * N * sizeof(float));
    cudaMalloc((void**)&Bd, M * N * sizeof(float));

    cudaMemcpy(Ad, A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1)/blockSize.x, (N + blockSize.y - 1)/blockSize.y);

    transpose<<<gridSize, blockSize>>>(Ad, Bd, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, Bd, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(Ad);
    cudaFree(Bd);
}

int main()
{
    int M = 4, N = 4;
    float A[M * N] = {1.0f, 2.0f, 3.0f, 4.0f,
                   5.0f, 6.0f, 7.0f, 8.0f,
                   9.0f, 10.0f, 11.0f, 12.0f,
                   13.0f, 14.0f, 15.0f, 16.0f};
    float B[M * N];
    invoke_transpose(A, B, M, N);
    std::cout << "Transposed Matrix B:" << std::endl;
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<M; j++)
        {
            std::cout << B[i * M + j] << " ";
        }
        std::cout << std::endl;

    }
    return 0;
}