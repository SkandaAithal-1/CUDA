#include <iostream>
#include "utils.h"

// Assumptions : 
// KERNEL_WIDTH is small and can fit in constant memory
// KERNEL_WIDTH is odd

const int INPUT_TILE_WIDTH = 256;
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

__global__ void conv1d(const float *in, float *out, int N)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int outputTileSize = INPUT_TILE_WIDTH - 2 * (KERNEL_WIDTH/2);
    int radius = KERNEL_WIDTH/2;
    int inputIdx = bx * outputTileSize + tx - radius;

    __shared__ float sharedMem[INPUT_TILE_WIDTH];

    if (0 <= inputIdx && inputIdx < N) 
    {
        sharedMem[tx] = in[inputIdx];
    }
    else 
    {
        sharedMem[tx] = 0.0f;
    }
    __syncthreads();

    if (radius <= inputIdx && inputIdx < N - radius)
    {
        if (0 <= tx-radius && tx-radius < outputTileSize)
        {
            float outValue = 0.0f;
            for (int k = 0; k < KERNEL_WIDTH; k++)
            {
                outValue += kernel[k] * sharedMem[tx - radius + k];
            }
            out[inputIdx-radius] = outValue;
        }
    }
}

__global__ void conv1d_output_tile(const float *in, float *out, int N)
{
    int tx = threadIdx.x, bx = blockIdx.x, radius = (KERNEL_WIDTH/2);
    int inputTileSize = OUTPUT_TILE_WIDTH + 2 * radius;
    int outputIdx = bx * OUTPUT_TILE_WIDTH + tx;

    __shared__ float sharedMem[inputTileSize];

    if (outputIdx < radius || outputIdx >= N-radius)
    {
        return;
    }

    if (tx < radius)
    {
        sharedMem[tx] = in[outputIdx-radius];
    }
    if (OUTPUT_TILE_WIDTH-tx >= radius)
    {
        sharedMem[tx+2*radius] = in[outputIdx+radius];
    }
    sharedMem[tx+radius] = in[outputIdx];
    __syncthreads();

    float pValue = 0.0f;
    for (int k=0; k<KERNEL_WIDTH; k++)
    {
        pValue += kernel[k] * sharedMem[tx+k];
    }
    out[outputIdx-radius] = pValue;
}

void launch_conv1d(const float *a, const float *b, float *c, int N){
    float *ad, *cd;
    cudaMalloc((void**)&ad, N * sizeof(float));
    cudaMalloc((void **)&cd, (N - KERNEL_WIDTH + 1) * sizeof(float));

    cudaMemcpy(ad, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel, b, KERNEL_WIDTH * sizeof(float));

    int blockSize = INPUT_TILE_WIDTH;
    int gridSize = (N + blockSize - 1) / blockSize;

    conv1d<<<gridSize, blockSize>>>(ad, cd, N);
    cudaDeviceSynchronize();

    cudaMemcpy(c, cd, (N - KERNEL_WIDTH + 1)*sizeof(float), cudaMemcpyDeviceToHost);
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
    int K = KERNEL_WIDTH;
    float a[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float b[K] = {2, 1, 0, -1, -2};
    float c[N - K + 1];
    float c_optim[N - K + 1] = {0.0f};

    launch_conv1d_unoptim(a, b, c, N, K);

    launch_conv1d(a, b, c_optim, N);

    std::cout << "Convolution result : ";
    print_array(c, N - KERNEL_WIDTH + 1);

    std::cout << "Convolution optimised result: ";
    print_array(c_optim, N - KERNEL_WIDTH + 1);

    return 0;
}