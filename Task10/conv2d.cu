#include <iostream>
#include "utils.h"

const int INPUT_TILE_WIDTH = 32;
int KERNEL_WIDTH = 5;
__constant__ float kernel[KERNEL_WIDTH];

__global__ void conv2d(const float *in, float *out, int N, int M)
{
    int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;
    int radius = (KERNEL_WIDTH)/2;
    int OUTPUT_TILE_WIDTH = INTPUT_TILE_WIDTH - 2 * radius;
    int inputX = bx * OUTPUT_TILE_WIDTH + tx - radius;
    int inputY = by * OUTPUT_TILE_WIDTH + ty - radius;
    __shared__ float sharedMem[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];

    if (0 <= inputX && inputX < N && 0 <= inputY && inputY < M)
    {
        sharedMem[tx][ty] = in[inputX][inputY];
    }
    else
    {
        sharedMem[tx][ty] = 0.0f;
    }
    __syncthreads();

    if (radius <= inputX && inputX < N-radius && radius <= inputY && inputY < M-radius)
    {
        if (radius <= tx && tx < INPUT_TILE_WIDTH-radius && radius <= ty && ty < INPUT_TILE_WIDTH-radius)
        {
            float pValue = 0.0f;
            for (int i=0; i<KERNEL_WIDTH; i++){
                for (int j=0; j<KERNEL_WIDTH; j++){
                    pValue += kernel[i][j] * sharedMem[tx-radius+i][ty-radius+j];
                }
            }
            out[inputX-radius][inputY-radius] = pValue;
        }
    }
}

void launch_conv2d(float *A, float *B, float *C, int N, int M)
{
    float *Ad, *Cd;
    cudaMalloc((void **)&Ad, N*M*sizeof(float));
    cudaMalloc((void **)&Cd, N*M*sizeof(float));

    cudaMemcpy(Ad, A, N*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel, B, KERNEL_WIDTH*sizeof(float));

    dim3 blockSize(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH);
    dim3 gridSize((N + KERNEL_WIDTH + INPUT_TILE_WIDTH - 1)/INPUT_TILE_WIDTH, (M + KERNEL_WIDTH + INPUT_TILE_WIDTH - 1)/INPUT_TILE_WIDTH);

    conv2d<<<gridSize, blockSize>>>(Ad, Cd, N, M);
    cudaDeviceSynchronize();

    cudaMemcpy(C, Cd, (N-KERNEL_WIDTH+1)*(M-KERNEL_WIDTH+1)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(Ad);
    cudaFree(Cd);
}

int main()
{
    return 0;
}