#include <iostream>
#include <ctime>

#define n 1024
#define block_size 32

__global__ void mul_matrix(int *a, int *b, int *c) {
    int my_x = blockIdx.x * blockDim.x + threadIdx.x;
    int my_y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_c = 0;
    __shared__ int A_s[block_size][block_size];
    __shared__ int B_s[block_size][block_size];

    for (int i = 0; i < n / block_size; i++) {
        A_s[threadIdx.x][threadIdx.y] = a[my_x * n + (i * block_size + threadIdx.y)];
        B_s[threadIdx.x][threadIdx.y] = b[(i * block_size + threadIdx.x) * n + my_y];
        __syncthreads();

        for (int j = 0; j < block_size; j++) {
            local_c += A_s[threadIdx.x][j] * B_s[j][threadIdx.y];
        }
        __syncthreads();
    }
    c[my_x * n + my_y] = local_c;
}

int main() {
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = new int[n * n];
    h_B = new int[n * n];
    h_C = new int[n * n];

    // Initialize input matrices
    for (int i = 0; i < n * n; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, n * n * sizeof(int));
    cudaMalloc((void**)&d_B, n * n * sizeof(int));
    cudaMalloc((void**)&d_C, n * n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid(n / block_size, n / block_size);
    dim3 dimBlock(block_size, block_size);

    // Variables for timing
    timespec start, stop;
    double time;

    // Measure start time
    if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
        perror("clock gettime");
    }

    // Launch the CUDA kernel
    mul_matrix<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Copy the result from the device to the host
    cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Measure stop time
    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
        perror("clock gettime");
    }

    // Calculate the elapsed time
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time taken: %f seconds\n", time);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
