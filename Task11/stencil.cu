#include<iostream>
#include "utils.h"

const int OUT_TILE_WIDTH = 30;
const float c[7] = {0.0f};

// 3D seven point stencil with register tiling and thread coarsening
__global__ void stencil(const float *I, float *O, int N)
{
    int iS = blockIdx.z * OUT_TILE_WIDTH;
    int j = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - 1;

    float xPrev, xNext;
    __shared__ float xCurr[OUT_TILE_WIDTH+2][OUT_TILE_WIDTH+2];

    if (0 <= iS-1 && iS-1 < N)
        xPrev = I[(iS-1)*N*N+j*N+k];
    if (0 <= j && j < N && 0 <= k && k < N)
        xCurr[threadIdx.x][threadIdx.y] = I[i*N*N+j*N+k];

    for (int i=iS; i<iS+OUT_TILE_WIDTH; i++)
    {
        if (0 <= i+1 && i+1 < N)
            xNext = I[(i+1)*N*N+j*N+k];
        if (1 <= i && i < N-1 && 1 <= j && j < N-1 && 1 <= k & k < N-1)
        {
            if (1 <= threadIdx.x && threadIdx.x < OUT_TILE_WIDTH+1 && 1 <= threadIdx.y && threadIdx.y < OUT_TILE_WIDTH+1)
            {
                O[i*N*N+j*N+k] = c[0] * xCurr[threadIdx.x][threadIdx.y] + 
                                 c[1] * xCurr[threadIdx.x-1][threadIdx.y] +
                                 c[2] * xCurr[threadIdx.x+1][threadIdx.y] +
                                 c[3] * xCurr[threadIdx.x][threadIdx.y-1] +
                                 c[4] * xCurr[threadIdx.x][threadIdx.y+1] +
                                 c[5] * xPrev + 
                                 c[6] * xNext;
            }
        }
        if (i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
        {
            if (0 <=i && i < N && 0 <= j && j < N && 0 <= k && k < N)
                O[i*N*N+j*N+k] = xCurr[threadIdx.x][threadIdx.y]; 
        }
        __syncthreads();

        xPrev = xCurr[threadIdx.x][threadIdx.y];
        xCurr[threadIdx.x][threadIdx.y] = xNext;
        __syncthreads();
    }
}

void launch_stencil(const float *I, float *O, int N)
{
    float *dI, *dO;
    cudaMalloc((void**)&dI, N*N*N*sizeof(float));
    cudaMalloc((void**)&dO, N*N*N*sizeof(float));
    
    cudaMemcpy(dI, I, N*N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(OUT_TILE_WIDTH+2, OUT_TILE_WIDTH+2);
    dim3 gridSize((N + 2 + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH,
                  (N + 2 + OUT_TILE_WIDTH - 1)/OUT_TILE_WIDTH)

    stencil<<<gridSize, blockSize>>>(dI, dO, N);
    cudaDeviceSynchronize();

    cudaMemcpy(O, dO, N*N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dI);
    cudaFree(dO);
}

int main()
{
    return 0;
}