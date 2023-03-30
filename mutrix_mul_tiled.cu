#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 2
#define TOLERANCE 0.0001

__global__ void sharedMatrixMulKernel(float* M, float* N, float* P, int width) {
	//create shared variables Mds and Nds
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	//for easy access from registers, pre-define indexes
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	//determine row and column index of P element that the thread is to produce
	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float pVal = 0.0f;

	//loop through phases in the block
	for (int ph = 0; ph < ceil(width / (float)TILE_WIDTH); ++ph)
	{
		//need checking before assigning
		//ph * TILE_WIDTH + tx is the new col val and we need to check if its < width
		if (row < width && ph * TILE_WIDTH + tx < width)
			Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
		else
			Mds[ty][tx] = 0;
		//same here
		//ph * TILE_WIDTH + ty is the new row val
		if (col < width && ph * TILE_WIDTH + ty < width)
			Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
		else
			Nds[ty][tx] = 0;
		//make sure all threads have finished loading the tiles of M and N into Mds and Nds
		__syncthreads();
		//combine all the tiles together
		for (int k = 0; k < TILE_WIDTH; ++k)
			pVal += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}
	if (row < width && col < width)
		P[row * width + col] = pVal;
}

__host__ void sharedMatrixMulHost(float* h_M, float* h_N, float* h_P, int width) {
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
	dim3 threads(TILE_WIDTH, TILE_WIDTH);
	dim3 blocks(ceil(width / (float)threads.x), ceil(width / (float)threads.y));

	cudaDeviceSynchronize();
	float gpu_time = 0.0f;

	cudaEventRecord(start, 0);
	//launch kernel
	sharedMatrixMulKernel << <blocks, threads, 0, 0 >> > (d_M, d_N, d_P, width);
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

void initMatrixComputation(int width) {
	int size = width * width * sizeof(float);
	float* h_M, * h_N, * h_P;

	//allocate host (CPU) memory
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

	sharedMatrixMulHost(h_M, h_N, h_P, width);

	//free host (CPU) memory
	free(h_M);
	free(h_N);
	free(h_P);
}

int main(int argc, char* argv[]) {
	printf("======================== TILE_WIDTH: %d ===============================\n\n", TILE_WIDTH);
	initMatrixComputation(125);
	initMatrixComputation(250);
	initMatrixComputation(500);
	initMatrixComputation(1000);
	initMatrixComputation(2000);
	#undef TILE_WIDTH
    #define TILE_WIDTH 5
	printf("======================== TILE_WIDTH: %d ===============================\n\n", TILE_WIDTH);
	initMatrixComputation(125);
	initMatrixComputation(250);
	initMatrixComputation(500);
	initMatrixComputation(1000);
	initMatrixComputation(2000);
	#undef TILE_WIDTH
	#define TILE_WIDTH 10
	printf("======================== TILE_WIDTH: %d ===============================\n\n", TILE_WIDTH);
	initMatrixComputation(125);
	initMatrixComputation(250);
	initMatrixComputation(500);
	initMatrixComputation(1000);
	initMatrixComputation(2000);
	#undef TILE_WIDTH
	#define TILE_WIDTH 20
	printf("======================== TILE_WIDTH: %d ===============================\n\n", TILE_WIDTH);
	initMatrixComputation(125);
	initMatrixComputation(250);
	initMatrixComputation(500);
	initMatrixComputation(1000);
	initMatrixComputation(2000);
	#undef TILE_WIDTH
	#define TILE_WIDTH 25
	printf("======================== TILE_WIDTH: %d ===============================\n\n", TILE_WIDTH);
	initMatrixComputation(125);
	initMatrixComputation(250);
	initMatrixComputation(500);
	initMatrixComputation(1000);
	initMatrixComputation(2000);

	return 0;
}
