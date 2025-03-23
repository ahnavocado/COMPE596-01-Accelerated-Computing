#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define BLOCK_SIZE 512  // 블록 크기 (변경 가능)

__global__ void reduction(float *input, float *output, int len) {
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    // 공유 메모리에 데이터 로드 (배열 범위 체크)
    if (start + t < len)
        partialSum[t] = input[start + t];
    else
        partialSum[t] = 0;
    
    if (start + BLOCK_SIZE + t < len)
        partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
        partialSum[BLOCK_SIZE + t] = 0;

    // Reduction 연산 수행 (stride 방식)
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride) 
            partialSum[t] += partialSum[t + stride];
    }

    // 블록 내 최종 합을 output 배열에 저장
    if (t == 0)
        output[blockIdx.x] = partialSum[0];
}

// CPU에서 순차적으로 합산 수행
float cpu_reduction(float *input, int N) {
    clock_t start, end;
    float sum = 0.0f;

    start = clock();
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    end = clock();

    printf("CPU elapsed time: %f ms\n", ((double)(end - start)) / CLOCKS_PER_SEC * 1000);
    return sum;
}

// GPU Reduction 실행 및 시간 측정
float gpu_reduction(float *h_input, int N) {
    float *d_input, *d_output, *h_output;
    int numBlocks = (N + (2 * BLOCK_SIZE - 1)) / (2 * BLOCK_SIZE);
    cudaEvent_t start, stop;
    float elapsedTime;

    h_output = (float *)malloc(numBlocks * sizeof(float));
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, numBlocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    reduction<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // 최종 Reduction (CPU에서 블록 결과 합산)
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_output[i];
    }

    printf("GPU elapsed time: %f ms\n", elapsedTime);

    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return sum;
}

int main() {
    int N_values[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    int numTests = sizeof(N_values) / sizeof(N_values[0]);

    for (int i = 0; i < numTests; i++) {
        int N = N_values[i];
        printf("\nTesting with N = %d\n", N);

        float *h_input = (float *)malloc(N * sizeof(float));

        // 랜덤 데이터 생성
        for (int j = 0; j < N; j++) {
            h_input[j] = (float)(rand() % 100) / 10.0f; // 0~10 사이 랜덤 float
        }

        float cpu_sum = cpu_reduction(h_input, N);
        float gpu_sum = gpu_reduction(h_input, N);

        printf("CPU Sum: %f\n", cpu_sum);
        printf("GPU Sum: %f\n", gpu_sum);

        free(h_input);
    }

    return 0;
}

