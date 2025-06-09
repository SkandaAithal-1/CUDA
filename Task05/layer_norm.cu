#include <iostream>
#include <cmath>

__global__ void layer_norm(float *x, float *y, int N, int M, float epsilon)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x;
    int row_idx = bx * blockDim.x + tx;
    if (row_idx < N)
    {
        extern __shared__ float shared_mem[];
        float *row_data = shared_mem;

        float mean = 0.0f;
        float var = 0.0f;

        // Load the row data into shared memory
        for (int col_idx = ty; col_idx < M; col_idx += blockDim.y)
        {
            row_data[col_idx] = x[row_idx * M + col_idx];
        }
        __syncthreads();

        // Calculate mean
        for (int col_idx = 0; col_idx < M; col_idx++)
        {
            mean += row_data[col_idx];
        }
        mean /= M;

        // Calculate variance
        for (int col_idx = 0; col_idx < M; col_idx++)
        {
            var += (row_data[col_idx] - mean) * (row_data[col_idx] - mean);
        }
        var /= M;

        // Normalize row data
        for (int col_idx = ty; col_idx < M; col_idx += blockDim.y)
        {
            y[row_idx * M + col_idx] = (row_data[col_idx] - mean) / sqrtf(var + epsilon);
        }
    }
}

void launch_layer_norm(float *x, float *y, int N, int M, float epsilon)
{
    // Launches the CUDA kernel for layer normalization
    float *xd, *yd;

    // Allocate device memory
    cudaMalloc((void **)&xd, N * M * sizeof(float));
    cudaMalloc((void **)&yd, N * M * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(xd, x, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Execution parameter configuration
    dim3 blockDim(16, 16);
    int gridDim = (N + blockDim.x - 1) / blockDim.x;
    size_t shared_mem_size = M * sizeof(float);

    // Launch the kernel
    layer_norm<<<gridDim, blockDim, shared_mem_size>>>(xd, yd, N, M, epsilon);
    cudaDeviceSynchronize();

    // Copy results to the host
    cudaMemcpy(y, yd, N * M * sizeof(float), cudaMemcpyDeviceToHost);
}

int main()
{
    int N = 3, M = 3;
    float epsilon = 1e-5f;
    float x[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};

    float y[N * M];
    launch_layer_norm(x, y, N, M, epsilon);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            std::cout << y[i * M + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}