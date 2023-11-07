#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define n 1024

__global__ void mul_matrix(int *a, int *b, int *c) {
    int my_x, my_y;
    my_x = blockIdx.x * blockDim.x + threadIdx.x;
    my_y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_c = 0;
    for (int i = 0; i < n; i++) {
        local_c += a[my_x * n + i] * b[i * n + my_y];
    }
    c[my_x * n + my_y] = local_c; 
}

int main() {
    int i;
    int *a = (int*)malloc(sizeof(int) * n * n);
    int *b = (int*)malloc(sizeof(int) * n * n);
    int *c = (int*)malloc(sizeof(int) * n * n); 

    dim3 dimGrid(64, 64);
    dim3 dimBlock(16, 16);

    for (i = 0; i < n * n; i++) {
        a[i] = 1;
        b[i] = 2;
    }

    int *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void**)&gpu_a, n * n * sizeof(int));
    cudaMalloc((void**)&gpu_b, n * n * sizeof(int));
    cudaMalloc((void**)&gpu_c, n * n * sizeof(int));

    cudaMemcpy(gpu_a, a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mul_matrix<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel Execution Time: %f ms\n", milliseconds);

    cudaMemcpy(c, gpu_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    printf("Result at C[451][451]: %d\n", c[451 * n + 451]);

    free(a);
    free(b);
    free(c); 
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    return 0;
}
