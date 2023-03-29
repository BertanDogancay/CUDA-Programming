#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define TOLERANCE 0.0001

__global__ void matrixMulKernel(float* M, float* N, float* P, int width) {
	//calculate row index of the P element and M
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//calculate column index of the P element and N
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//check first
	if (row < width && col < width) {
		float pVal = 0.0;
		for (int k = 0; k < width; k++) {
			pVal += M[row * width + k] * N[k * width + col];
		}
		P[row * width + col] = pVal;
	}
}

__host__ void matrixMulHost(float* h_M, float* h_N, float* h_P, int width) {
	int size = width * width * sizeof(float);
	float* d_M, * d_N, * d_P;

	printf("Test %d x %d:\n", width, width);

	//create cuda event handles
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//allocate memory for input and output device matrices
	cudaMalloc(&d_M, size);
	cudaMalloc(&d_N, size);
	cudaMalloc(&d_P, size);

	cudaDeviceSynchronize();
	float transfer_time_HtoD = 0.0f;

	cudaEventRecord(start, 0);
	//tranfer input data from host to device
	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_time_HtoD, start, stop);

	//set kernel launch config
	//number of threads = threads.x * threads.y
	//block width = threads.x
	//number of blocks = ((width+threads.x-1) / threads.x) * ((width+threads.y-1) / threads.y)
	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (width + threads.y - 1) / threads.y);

	cudaDeviceSynchronize();
	float gpu_time = 0.0f;

	cudaEventRecord(start, 0);
	//launch kernel
	matrixMulKernel << <blocks, threads, 0, 0 >> > (d_M, d_N, d_P, width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);

	cudaDeviceSynchronize();
	float transfer_time_DtoH = 0.0f;

	cudaEventRecord(start, 0);
	//transfer calculated output data back to host
	cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_time_DtoH, start, stop);

	//free device memory
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);

	cudaDeviceSynchronize();
	float cpu_time = 0.0f;

	cudaEventRecord(start, 0);
	//test is being done using CPU
	//calculate CPU computed value and take the difference of it with the GPU calculated value
	//loop through each element in the matrix with given size
	for (int row = 0; row < width; row++) {
		for (int col = 0; col < width; col++) {
			float sum = 0.0f;
			for (int k = 0; k < width; k++) {
				sum += h_M[row * width + k] * h_N[k * width + col];
			}
			float diff = h_P[row * width + col] - sum;
			if (diff > TOLERANCE || diff < -TOLERANCE) {
				printf("Test FAILED\n");
				return;
			}
		}
	}
	printf("Test PASSED\n");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_time, start, stop);
	printf("Time spent transferring (host-to-device):                %.2f ms\n", transfer_time_HtoD);
	printf("Time spent transferring (device-to-host):                %.2f ms\n", transfer_time_DtoH);
	printf("Total time spent transferring:                           %.2f ms\n", transfer_time_HtoD + transfer_time_DtoH);
	printf("Time spent executing by the GPU:                         %.2f ms\n", gpu_time);
	printf("Time spent executing by the CPU:                         %.2f ms\n\n", cpu_time);

	//release resources
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaDeviceReset();
}

void initializeMats(int width) {
	int size = width * width * sizeof(float);
	float* h_M, * h_N, * h_P;

	//allocate Host (CPU) memory
	h_M = (float*)malloc(size);
	h_N = (float*)malloc(size);
	h_P = (float*)malloc(size);

	//initialize random matrices
	for (int row = 0; row < width; row++) {
		for (int col = 0; col < width; col++) {
			int index = row * width + col;
			h_M[index] = (float)rand() / RAND_MAX;
			h_N[index] = (float)rand() / RAND_MAX;
		}
	}

	matrixMulHost(h_M, h_N, h_P, width);

	//free Host (CPU) memory
	free(h_M);
	free(h_N);
	free(h_P);
}
 
int main(int argc, char* argv[]) {
	initializeMats(125);
	initializeMats(250);
	initializeMats(500);
	initializeMats(1000);
	initializeMats(2000);
	return 0;
}
