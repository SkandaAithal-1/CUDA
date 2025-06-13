#include <iostream>

const int TILE_WIDTH = 256;
const int KERNEL_WIDTH = 5;
__constant__ float kernel[KERNEL_WIDTH];

__global__ void conv1d_unoptim(const float *a, const float *b, float *c, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N - K + 1)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += a[idx + i] * b[i];
        }
        c[idx] = sum;
    }
}

__global__ void conv1d(const float *in, float *out, int N, int K)
{
    int tx = threadIdx.x, ty = threadIdx, y;
    int idx = blockIdx.x * blockDim.x + tx;
    __shared__ float sharedMemIn[TILE_WIDTH];
    extern __constant__ float kernel[];
    if (idx < N)
    {
        sharedMemIn[tx] = in[idx];
        __syncthreads();

        float result = 0.0f;
        for (int i=min(idx, KERNEL_WIDTH); i>=max(idx-K+1, 0); i--){
            out[idx-i] += sharedMemIn[tx] * kernel[i];
        }
    }
}

void launch_conv1d(const float *a, const float *b, float *c, int N, int K){
    float *ad, *cd;
    cudaMalloc((void**)&ad, N * sizeof(float));
    cudaMalloc((void **)&cd, (N - K + 1) * sizeof(float));

    __constant__ float kernel[K];
    cudaMemcpy(ad, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel, b, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = TILE_WIDTH;
    int gridSize = (N + blockSize - 1) / blockSize;

    conv1d<<<gridSize, blockSize>>>(ad, cd, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(c, cd, (N - K + 1)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(ad);
    cudaFree(cd);
}

void launch_conv1d_unoptim(const float *a, const float *b, float *c, int N, int K)
{
    float *ad, *bd, *cd;
    cudaMalloc((void **)&ad, N * sizeof(float));
    cudaMalloc((void **)&bd, K * sizeof(float));
    cudaMalloc((void **)&cd, (N - K + 1) * sizeof(float));

    cudaMemcpy(ad, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, K * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N - K + 1 + blockSize - 1) / blockSize;

    conv1d_unoptim<<<gridSize, blockSize>>>(ad, bd, cd, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(c, cd, (N - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
}

int main()
{
    int N = 10;
    int K = 3;
    float a[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float b[K] = {1, 0, -1};
    float c[N - K + 1];
    float c_optim[N - K + 1] = {0.0f};

    launch_conv1d_unoptim(a, b, c, N, K);

    launch_conv1d(a, b, c_optim, N, K);

    std::cout << "Convolution result: ";
    for (int i = 0; i < N - K + 1; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}