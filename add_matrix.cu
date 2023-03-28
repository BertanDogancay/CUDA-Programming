#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define TOLERANCE 0.0001

//add two matrices (two-dimensional) output --> matrix element
__global__ void matrixAdditionKernel(float* C, float* A, float* B, int width) {
    //since it has two dimensions, use both x and y
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    //check first
	if (row < width && col < width) {
        //find index
	int index = row * width + col;
	C[index] = A[index] + B[index];
	}
}

//add two matrix rows (one-dimensional) output --> matrix row
__global__ void matrixRowAdditionKernel(float* C, float* A, float* B, int width) {
    //since it only has one dimension, use x only
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    //check first
    if (row < width) {
        for (int col = 0; col < width; col++) {
            //find index for each
            int index = row * width + col;
            C[index] = A[index] + B[index];
        }
    }
}

//add two matrix columns (one-dimensional) output --> matrix column
__global__ void matrixColAdditionKernel(float* C, float* A, float* B, int width) {
    //since it only has one dimension, use x only
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //check first
    if (col < width) {
        for (int row = 0; row < width; row++) {
            //find index for each
            int index = row * width + col;
            C[index] = A[index] + B[index];
        }
    }
}

__host__ void matrixAdditionHost(float* h_C, const float* h_A, const float* h_B, int width) {
   int size = width * width * sizeof(float);
   float* d_A, * d_B, * d_C;

    //allocate memory for input and output matrices
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //transfer input data to device (GPU)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //set kernel launch config
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (width + threads.y - 1) / threads.y);

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    float gpu_time = 0.0f;

    //start the timer
    cudaEventRecord(start, 0);

    //launch kernels
    matrixAdditionKernel <<<blocks, threads, 0, 0 >>> (d_C, d_A, d_B, width);
    matrixRowAdditionKernel << <blocks, threads, 0, 0 >> > (d_C, d_A, d_B, width);
    matrixColAdditionKernel << <blocks, threads, 0, 0 >> > (d_C, d_A, d_B, width);

    //stop the timer
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // print the GPU times
    printf("time spent executing by the GPU: %.2f ms\n", gpu_time);

    //transfer calculated output data back to host (CPU)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    //release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
}

void testMatrixAddition(float* h_C, const float* h_A, const float* h_B, int width) {
    matrixAdditionHost(h_C, h_A, h_B, width);
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * width + col;
            float diff = h_C[index] - (h_A[index] + h_B[index]);
            if (diff > TOLERANCE || diff < -TOLERANCE) {
                printf("Test FAILED\n");
                return;
            }
        }
    }
    printf("Test PASSED\n");
}

int main(int argc, char* argv[]) {
    int width = 125; //try with different width ex. 500 or 1000
    int size = width * width * sizeof(float);
    float* h_A, * h_B, * h_C;

    //allocate Host (CPU) memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    //initialize random matrices
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * width + col;
            h_A[index] = (float)rand() / RAND_MAX;
            h_B[index] = (float)rand() / RAND_MAX;
        }
    }

    //test it!!
    testMatrixAddition(h_C, h_A, h_B, width);

    //free Host (CPU) memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
