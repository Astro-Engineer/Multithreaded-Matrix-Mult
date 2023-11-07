#include <iostream>
#include <ctime>

#define n 1024
#define block_size 32

__global__ void mul_matrix(int *a, int *b, int *c) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int my_x = blockIdx.x * blockDim.x + threadIdx.x;
    int my_y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_c = 0;
    __shared__ int A_s[32][32];
    __shared__ int B_s[32][32];

    for (int i = 0; i < n / block_size; i++) {
        A_s[row][col] = a[my_x * n + (i * block_size + col)];
        B_s[row][col] = b[(i * block_size + row) * n + my_y];
        __syncthreads();

        for (int j = 0; j < block_size; j++) {
            local_c += A_s[row][j] * B_s[j][col];
        }
        __syncthreads();
    }
    c[my_x * n + my_y] = local_c;
}

int main() {
    int *gpu_a, *gpu_b;
    int *d_a, *d_b, *d_c;
    int *h_c = new int[n * n]; 

    gpu_a = new int[n * n];
    gpu_b = new int[n * n];

    for (int i = 0; i < n * n; i++) {
        gpu_a[i] = 1;
        gpu_b[i] = 2;
    }

    cudaMalloc((void**)&d_a, n * n * sizeof(int));
    cudaMalloc((void**)&d_b, n * n * sizeof(int));
    cudaMalloc((void**)&d_c, n * n * sizeof(int));

    cudaMemcpy(d_a, gpu_a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, gpu_b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(n / block_size, n / block_size);
    dim3 dimBlock(block_size, block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start);
    mul_matrix<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel Execution Time: %f ms\n", milliseconds);

    cudaMemcpy(h_c, d_c, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result at C[451][451]: %d\n", h_c[451 * n + 451]);

    delete[] gpu_a;
    delete[] gpu_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
