#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define MASK_WIDTH 5
#define MASK_RADIUS 2

__global__ void convolution(unsigned int *inputImage, int *mask, int *outputImage,
                            int channels, int width, int height) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row < height && col < width) {
        for (int ch = 0; ch < channels; ch++) {
            int sum = 0;
            for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
                for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
                    int r = row + i;
                    int c = col + j;

                    if (r >= 0 && r < height && c >= 0 && c < width) {
                        int imageIdx = (r * width + c) * channels + ch;
                        int maskVal = mask[(i + MASK_RADIUS) * MASK_WIDTH + (j + MASK_RADIUS)];
                        sum += inputImage[imageIdx] * maskVal;
                    }
                }
            }
            int outIdx = (row * width + col) * channels + ch;
            outputImage[outIdx] = sum;
        }
    }
}

int main() {
    unsigned int *hostInputImage;
    int *hostOutputImage;
    unsigned int inputLength = 640 * 640 * 3;
    int imageWidth = 640;
    int imageHeight = 640;
    int channels = 3;

    printf("Loading peppers.dat...\n");
    hostInputImage = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
    hostOutputImage = (int *)malloc(inputLength * sizeof(int));

    FILE *f = fopen("snowman.dat", "r");
    unsigned int pixelValue, i = 0;
    while (!feof(f) && i < inputLength) {
        fscanf(f, "%d", &pixelValue);
        hostInputImage[i++] = pixelValue;
    }
    fclose(f);

    // Sobel 5x5 horizontal kernel
    int hostMask[5][5] = {
        {2, 2, 4, 2, 2},
        {1, 1, 2, 1, 1},
        {0, 0, 0, 0, 0},
        {-1, -1, -2, -1, -1},
        {-2, -2, -4, -2, -2}
    };

    unsigned int *deviceInputImage;
    int *deviceOutputImage;
    int *deviceMask;

    cudaMalloc((void**)&deviceInputImage, inputLength * sizeof(unsigned int));
    cudaMalloc((void**)&deviceOutputImage, inputLength * sizeof(int));
    cudaMalloc((void**)&deviceMask, 25 * sizeof(int));
    cudaMemcpy(deviceInputImage, hostInputImage, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMask, hostMask, 25 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH), ceil((float)imageHeight / TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    convolution<<<dimGrid, dimBlock>>>(deviceInputImage, deviceMask, deviceOutputImage,
                                       channels, imageWidth, imageHeight);

    cudaMemcpy(hostOutputImage, deviceOutputImage, inputLength * sizeof(int), cudaMemcpyDeviceToHost);

    f = fopen("peppers.out", "w");
    for (int i = 0; i < inputLength; ++i)
        fprintf(f, "%d\n", hostOutputImage[i]);
    fclose(f);

    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
    cudaFree(deviceMask);
    free(hostInputImage);
    free(hostOutputImage);

    printf("Done! Output saved to peppers.out\n");
    return 0;
}

