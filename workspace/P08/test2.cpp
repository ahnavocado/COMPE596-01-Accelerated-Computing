/*
 * How to compile:
 *   nvcc -lcusolver -lcudart -o hilbert_solver hilbert_solver.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <chrono>

void generateHilbertMatrix(double* H, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            H[row + col * N] = 1.0 / (row + col + 1); // column-major order
        }
    }
}

void perturbVector(double* vec, int N) {
    for (int i = 0; i < N; ++i) {
        double epsilon = ((double)rand() / RAND_MAX); // 0 < ε < 1
        vec[i] += epsilon;
    }
}

int main() {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

    srand(42); // for reproducibility

    for (int pow = 1; pow <= 12; ++pow) {
        int N = 1 << pow;
        int lda = N, ldb = N;

        double *h_A = new double[N * N];
        double *h_b = new double[N];
        double *h_b_perturbed = new double[N];
        double *h_x = new double[N];
        double *h_x2 = new double[N];

        generateHilbertMatrix(h_A, N);
        for (int i = 0; i < N; ++i) h_b[i] = 1.0;
        for (int i = 0; i < N; ++i) h_b_perturbed[i] = h_b[i];
        perturbVector(h_b_perturbed, N);

        double *d_A, *d_B, *d_work;
        int *d_info;
        int lwork = 0;

        cudaMalloc(&d_A, sizeof(double) * N * N);
        cudaMalloc(&d_B, sizeof(double) * N);
        cudaMalloc(&d_info, sizeof(int));

        cudaMemcpy(d_A, h_A, sizeof(double) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_b, sizeof(double) * N, cudaMemcpyHostToDevice);

        cusolverDnDgetrf_bufferSize(cusolverH, N, N, d_A, lda, &lwork);
        cudaMalloc(&d_work, sizeof(double) * lwork);

        // LU + Solve for original b
        auto start1 = std::chrono::high_resolution_clock::now();
        cusolverDnDgetrf(cusolverH, N, N, d_A, lda, d_work, NULL, d_info);
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_A, lda, NULL, d_B, ldb, d_info);
        cudaDeviceSynchronize();
        auto end1 = std::chrono::high_resolution_clock::now();
        double elapsed1 = std::chrono::duration<double, std::milli>(end1 - start1).count();

        cudaMemcpy(h_x, d_B, sizeof(double) * N, cudaMemcpyDeviceToHost);

        // Reuse LU, solve again for perturbed b
        cudaMemcpy(d_B, h_b_perturbed, sizeof(double) * N, cudaMemcpyHostToDevice);
        auto start2 = std::chrono::high_resolution_clock::now();
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, d_A, lda, NULL, d_B, ldb, d_info);
        cudaDeviceSynchronize();
        auto end2 = std::chrono::high_resolution_clock::now();
        double elapsed2 = std::chrono::duration<double, std::milli>(end2 - start2).count();

        cudaMemcpy(h_x2, d_B, sizeof(double) * N, cudaMemcpyDeviceToHost);

        printf("N = %d\n", N);
        printf("  Time (LU + Solve): %.4f ms\n", elapsed1);
        printf("  Time (Reuse LU):   %.4f ms\n\n", elapsed2);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_info);
        cudaFree(d_work);

        delete[] h_A;
        delete[] h_b;
        delete[] h_b_perturbed;
        delete[] h_x;
        delete[] h_x2;
    }

    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);
    cudaDeviceReset();
    return 0;
}

