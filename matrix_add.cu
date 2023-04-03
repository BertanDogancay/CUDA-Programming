#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define TOLERANCE 0.00010

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

   printf("Test %d x %d:\n", width, width);

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

    //create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    float gpu_time_1 = 0.0f;

    //start the timer
    cudaEventRecord(start, 0);
    //launch kernel for design 1
    matrixAdditionKernel <<<blocks, threads, 0, 0 >>> (d_C, d_A, d_B, width);
    //stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_1, start, stop);

    cudaDeviceSynchronize();
    float gpu_time_2 = 0.0f;

    cudaEventRecord(start, 0);
    matrixRowAdditionKernel << <blocks, threads, 0, 0 >> > (d_C, d_A, d_B, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_2, start, stop);

    cudaDeviceSynchronize();
    float gpu_time_3 = 0.0f;

    cudaEventRecord(start, 0);
    matrixColAdditionKernel << <blocks, threads, 0, 0 >> > (d_C, d_A, d_B, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_3, start, stop);

    //transfer calculated output data back to host (CPU)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    //free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    float cpu_time = 0.0f;

    cudaEventRecord(start, 0);
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * width + col;
            float diff = h_C[index] - (h_A[index] + h_B[index]);
            if (diff > TOLERANCE || diff < -TOLERANCE) {
                printf("Test FAILED\n\n");
                return;
            }
        }
    }
    printf("Test PASSED\n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);
    printf("Time spent executing by the GPU (Design 1):             %.2f ms\n", gpu_time_1);
    printf("Time spent executing by the GPU (Design 2):             %.2f ms\n", gpu_time_2);
    printf("Time spent executing by the GPU (Design 3):             %.2f ms\n", gpu_time_3);
    printf("Time spent executing by the CPU:                        %.2f ms\n\n", cpu_time);

    //release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //reset the device after execution
    cudaDeviceReset();
}

void initMatrixComputation(int width) {
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

    //call host function
    matrixAdditionHost(h_C, h_A, h_B, width);

    //free Host (CPU) memory
    free(h_A);
    free(h_B);
    free(h_C);
}

int main(int argc, char* argv[]) {
    initMatrixComputation(125);
    initMatrixComputation(250);
    initMatrixComputation(500);
    initMatrixComputation(1000);
    initMatrixComputation(2000);
    
    return 0;
}
