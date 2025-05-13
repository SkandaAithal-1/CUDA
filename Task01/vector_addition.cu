#include<iostream>

__global__ void vector_addition(const float *a, const float *b, float *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    int N = 100;
    float A[n], B[n], C[n];
    float *Ad, *Bd, *Cd;

    //Allocate memory on the device 
    cudaMalloc((void**)&Ad, N * (sizeof(float)));
    cudaMalloc((void**)&Bd, N * (sizeof(float)));
    cudaMalloc((void**)&Cd, N * (sizeof(float)));

    // Copy data from host to device
    cudaMemcpy(Ad, A, N * (sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, N * (sizeof(float)), cudaMemcpyHostToDevice);

    // Configure execution parameters
    int blockSize = 256;
    int gridSize = ceil(N / blockSize);  

    // Launch kernel
    vector_addition<<<gridSize, blockSize>>>(Ad, Bd, Cd, N);

    // Copy result from device to host
    cudaMemcpy(C, Cd, N * (sizeof(float)), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    return 0;
}