#include <iostream>

__global__ void matrix_vector_mult(const float *A, const float *x, float* y, int M, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M){
        for (int i=0; i<N; i++){
            y[row] += A[row * N + i] * x[i];
        }
    }
}

__global__ void matrix_vector_mult_optim(const float *A, const float *x, float *y, int M, int N){
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    // int TILE_WIDTH = bx;
    int row = bx * blockDim.x + tx;
    if (row >= M) return;

    __shared__ float x_shared[bx]; // bx = TILE_WIDTH 
    float sum = 0;
    for (int phase=0; phase < (int)(N/bx); phase++){  // bx = TILE_WIDTH
        // Load x into shared memory
        if (phase * bx + tx < N){
            x_shared[tx] = x[phase * bx + tx]; // bx = TILE_WIDTH
        }else{
            x_shared[tx] = 0;
        }
        __syncthreads();

        // Accumulate the sum for this phase
        for (int i=0; i<bx; i++){ // bx = TILE_WIDTH
            if (phase * bx + i >= N) break;
            sum += A[row * N + phase * bx + i] * x_shared[i]; // bx = TILE_WIDTH
        }
        __syncthreads();
    }
    y[row] = sum;
}

int main()
{
    int M = 100, N = 100;
    float A[M*N], x[N], y[M], y_optim[M];
    float *Ad, *xd, *yd, *y_optimd;
    for (int i=0; i<M*N; i++){A[i] = 1.0f;}
    for (int i=0; i<N; i++){x[i] = 1.0f;}

    // Allocate device memory
    cudaMalloc((void**)&Ad, M*N*sizeof(float));
    cudaMalloc((void**)&xd, N*(sizeof(float)));
    cudaMalloc((void**)&yd, M*(sizeof(float)));
    cudaMalloc((void**)&y_optimd, M*(sizeof(float)));

    // Copy data from host to device
    cudaMemcpy(Ad, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xd, x, N*sizeof(float), cudaMemcpyHostToDevice);

    // Execution parameters configuration
    int blockSize = 256;
    int gridSize = ceil(M / blockSize);

    // Launch kernel
    matrix_vector_mult<<<gridSize, blockSize>>>(Ad, xd, yd, M, N);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(y, yd, M*sizeof(float), cudaMemcpyDeviceToHost);

    // Launch optimized kernel
    matrix_vector_mult_optim<<<gridSize, blockSize>>>(Ad, xd, y_optimd, M, N);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(y_optim, y_optimd, M*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(Ad);
    cudaFree(xd);
    cudaFree(yd);
    cudaFree(y_optimd);

    return 0;
}