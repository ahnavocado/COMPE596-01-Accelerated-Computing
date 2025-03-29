#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

// Basic global memory matrix multiplication kernel
__global__ void BasicMatrixMulKernel(float* A, float* B, float* C, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < Width && Col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k)
            Pvalue += A[Row * Width + k] * B[k * Width + Col];
        C[Row * Width + Col] = Pvalue;
    }
}

// Shared memory tiled version
__global__ void TiledMatrixMulKernel(float* A, float* B, float* C, int Width) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int p = 0; p < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        if (Row < Width && p * TILE_WIDTH + tx < Width)
            ds_A[ty][tx] = A[Row * Width + p * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0;

        if (p * TILE_WIDTH + ty < Width && Col < Width)
            ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * Width + Col];
        else
            ds_B[ty][tx] = 0.0;

        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }

    if (Row < Width && Col < Width)
        C[Row * Width + Col] = Pvalue;
}

// Utility to initialize matrices with random float values
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = static_cast<float>(rand()) / RAND_MAX;
}

void runMatrixMultiplicationTest(int Width) {
    int size = Width * Width;
    int bytes = size * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_basic = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    randomInit(h_A, size);
    randomInit(h_B, size);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    float ms_basic = 0, ms_tiled = 0;

    // --- Basic kernel
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    BasicMatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_basic, start, stop);

    cudaMemcpy(h_C_basic, d_C, bytes, cudaMemcpyDeviceToHost);

    // --- Tiled kernel
    cudaMemcpy(d_C, h_C_basic, bytes, cudaMemcpyHostToDevice); // Clear C
    cudaEventRecord(start);
    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_tiled, start, stop);

    cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Matrix size: %4d x %4d | Basic: %8.4f ms | Tiled: %8.4f ms\n", Width, Width, ms_basic, ms_tiled);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_basic); free(h_C_tiled);
}

int main() {
    srand(time(NULL));
    printf("Comparing basic and tiled matrix multiplication kernels:\n");
    for (int exp = 5; exp <= 13; ++exp) { 
      int Width = 1 << exp;
        runMatrixMultiplicationTest(Width);
    }
    return 0;
}

