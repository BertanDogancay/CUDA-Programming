#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>

int get_SP_cores(cudaDeviceProp dev_prop) {
	int cores = 0;
	int mp = dev_prop.multiProcessorCount;

	switch (dev_prop.major) {
		case 2: // Fermi
			if (dev_prop.minor == 1) cores = mp * 48;
			else cores = mp * 32;
			break; // Kepler
		case 3:
			cores = mp * 192;
			break;
		case 5: // Maxwell
			cores = mp * 128;
			break;
		case 6: // Pascal
			if ((dev_prop.minor == 1) || (dev_prop.minor == 2)) cores = mp * 128;
			else if (dev_prop.minor == 0) cores = mp * 64;
			else printf("Unknown device type\n");
			break;
		case 7: // Volta and Turing
			if ((dev_prop.minor == 0) || (dev_prop.minor == 5)) cores = mp * 64;
			else if (dev_prop.minor == 2 || dev_prop.minor == 3) cores = 128;
			else printf("Unknown device type\n");
			break;
		case 8: // Ampere
			if (dev_prop.minor == 0) cores = mp * 64;
			else if (dev_prop.minor == 6) cores = mp * 128;
			else if (dev_prop.minor == 9) cores = mp * 128;
			else printf("Unknown device type\n");
			break;
		case 9: // Hopper
			if (dev_prop.minor == 0) cores = mp * 128;
			else printf("Unknown device type\n");
			break;
		default:
			printf("Unknown device type\n");
			break;
	}
	return cores;
}

void get_max_block_dim(cudaDeviceProp dev_drop, int *max_block_dim, int dev_id) {
	max_block_dim[0] = dev_drop.maxThreadsDim[0];
	max_block_dim[1] = dev_drop.maxThreadsDim[1];
	max_block_dim[2] = dev_drop.maxThreadsDim[2];
}

void get_max_grid_dim(cudaDeviceProp dev_drop, int* max_grid_dim, int dev_id) {
	max_grid_dim[0] = dev_drop.maxGridSize[0];
	max_grid_dim[1] = dev_drop.maxGridSize[1];
	max_grid_dim[2] = dev_drop.maxGridSize[2];
}

int main(int argc, char* argv[]) {
	//define
	int max_block_dim[3];
	int max_grid_dim[3];

	//get device count
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	printf("Number of CUDA devices in the system: %d\n\n", dev_count);

	cudaDeviceProp dev_prop;

	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&dev_prop, i);

		printf("Device Number:                          %d\n", i);
		printf("Type:                                   %s\n", dev_prop.name);
		printf("Clock Rate:                             %d KHz\n", dev_prop.clockRate);
		printf("Streaming Multiprocessors Count:        %d\n", dev_prop.multiProcessorCount);
		printf("Number of Cores:                        %d\n", get_SP_cores(dev_prop));
		printf("Warp Size:                              %d\n", dev_prop.warpSize);
		printf("Global Memory:                          %d bytes\n", (int)dev_prop.totalGlobalMem);
		printf("Constant Memory:                        %d bytes\n", (int)dev_prop.totalConstMem);
		printf("Shared Memory Per Block:                %d bytes\n", (int)dev_prop.sharedMemPerBlock);
		printf("Registers Available Per Block:          %d\n", dev_prop.regsPerBlock);
		printf("Max Number of Threads Per Block:        %d\n", dev_prop.maxThreadsPerBlock);

		get_max_block_dim(dev_prop, max_block_dim, i);
		printf("Max Block Dimensions:                   (%d, %d, %d)\n", max_block_dim[0], max_block_dim[1], max_block_dim[2]);

		get_max_grid_dim(dev_prop, max_grid_dim, i);
		printf("Max Grid Dimensions:                    (%d, %d, %d)\n\n", max_grid_dim[0], max_grid_dim[1], max_grid_dim[2]);
	}
}
