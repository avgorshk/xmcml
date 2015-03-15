#ifndef __MCML_GPU_H
#define __MCML_GPU_H

#define BLOCK_SIZE       512
#define NUMBER_OF_BLOCKS 1024
#define COEFF            4294967296

typedef struct __LayerStructGPU
{
	float z0, z1;
	float n;
	float mua;
	float mus;
	float g;
	float cos_crit0, cos_crit1;
} LayerStructGPU;

typedef struct __InStructGPU
{
	float wth;
	float dz, dr, da;
	unsigned int nz, nr, na;
	unsigned int num_layers;
	LayerStructGPU* layers;
} InStructGPU;

typedef struct __OutStructGPU
{
    float rsp;
	unsigned long long int* rd_ra;
	unsigned long long int* a_rz;
	unsigned long long int* t_ra;
} OutStructGPU;

typedef struct __ControlStructGPU
{
	unsigned int mcg59_step;
	unsigned int mcg59_first_id;
	unsigned int mcg59_seed;
	unsigned int num_photons;
	int gpu_devices;
} ControlStructGPU;

int GetNumberOfGPU();

int GetNumberOfGPUThreads();

void CountMcmlGpu(InStructGPU* in, OutStructGPU* out, ControlStructGPU* control);

#endif //__MCML_GPU_H