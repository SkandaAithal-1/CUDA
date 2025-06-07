#include<iostream>

__global__ void vector_sum(const float *x, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = N >> 1;
    if (idx < stride)
    {
        x[idx] += x[idx + stride];
    }
}

int main()
{
    int N = 8;
    float x[N] = [1, 2, 3, 4, 5, 6, 7, 8];
    float y;

    cudaMalloc((void **)&xd, N * sizeof(float));

    cudaMemcpy(xd, x, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    for (int size = N; size > 0; size >>= 1){
        vector_sum<<<gridSize, blockSize>>>(xd, size);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(y, xd, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum : " << y << std::endl;
}