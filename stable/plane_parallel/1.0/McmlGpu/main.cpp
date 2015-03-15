#include <stdlib.h>
#include <memory.h>
#include <omp.h>
#include <cuda_runtime_api.h>
#include "../McmlGetero/mcml_gpu.h"

float RSpecular(LayerStructGPU* layer);
void CallKernelThreadGPU(InStructGPU in, OutStructGPU out, ControlStructGPU control, int device_id, int pack_size, int layers_size);

int GetNumberOfGPU()
{
	int device_count = 1;
	cudaGetDeviceCount(&device_count);
	return device_count;
}

int GetNumberOfGPUThreads()
{
	return BLOCK_SIZE * NUMBER_OF_BLOCKS;
}

int SetNumberOfThreads(int gpu_devices)
{
	int device_count = 1;
	cudaGetDeviceCount(&device_count);
	if (device_count < gpu_devices)
	{
		omp_set_num_threads(device_count);
		return device_count;
	}
	else
	{
		omp_set_num_threads(gpu_devices);
		return gpu_devices;
	}
}

void KernelThreadGPU(InStructGPU* in, OutStructGPU* out, ControlStructGPU* control, int device_id)
{
	int n = control->num_photons;

	LayerStructGPU* dev_layers = NULL;
	unsigned long long int* dev_a_rz = NULL;
	unsigned long long int* dev_rd_ra = NULL;
	unsigned long long int* dev_t_ra = NULL;

	unsigned int layers_size = (in->num_layers + 2) * sizeof(LayerStructGPU);
	unsigned int rz_size = in->nr * in->nz * sizeof(unsigned long long int);
	unsigned int ra_size = in->nr * in->na * sizeof(unsigned long long int);

	cudaSetDevice(device_id);

	cudaMalloc((void**)&dev_layers, layers_size);
	cudaMalloc((void**)&dev_a_rz, rz_size);
	cudaMalloc((void**)&dev_rd_ra, ra_size);
	cudaMalloc((void**)&dev_t_ra, ra_size);

	cudaMemcpy(dev_layers, in->layers, layers_size, cudaMemcpyHostToDevice);
	cudaMemset(dev_a_rz, 0, rz_size);
	cudaMemset(dev_rd_ra, 0, ra_size);
	cudaMemset(dev_t_ra, 0, ra_size);

	InStructGPU in_gpu;
	in_gpu.da = in->da;
	in_gpu.dr = in->dr;
	in_gpu.dz = in->dz;
	in_gpu.na = in->na;
	in_gpu.nr = in->nr;
	in_gpu.nz = in->nz;
	in_gpu.num_layers = in->num_layers;
	in_gpu.wth = in->wth;
	in_gpu.layers = dev_layers;

	OutStructGPU out_gpu;
	out_gpu.rsp = out->rsp;
	out_gpu.a_rz = dev_a_rz;
	out_gpu.rd_ra = dev_rd_ra;
	out_gpu.t_ra = dev_t_ra;

	//int pack_size = n / (BLOCK_SIZE * NUMBER_OF_BLOCKS);
	int pack_size = 100;

	CallKernelThreadGPU(in_gpu, out_gpu, *control, device_id, pack_size, layers_size);

	cudaMemcpy(out->a_rz, dev_a_rz, rz_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(out->rd_ra, dev_rd_ra, ra_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(out->t_ra, dev_t_ra, ra_size, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_layers);
	cudaFree(dev_a_rz);
	cudaFree(dev_rd_ra);
	cudaFree(dev_t_ra);
}

void CountMcmlGpu(InStructGPU* in, OutStructGPU* out, ControlStructGPU* control)
{
	if (in == NULL || out == NULL || control == NULL)
	{
		return;
	}

	int device_count = SetNumberOfThreads(control->gpu_devices);
	
	control->num_photons /= device_count;
	control->gpu_devices = device_count;

	out->rsp = RSpecular(in->layers);

	OutStructGPU* out_gpu = (OutStructGPU*)malloc(sizeof(OutStructGPU) * device_count);
	int size_rz = sizeof(unsigned long long int) * in->nr * in->nz;
	int size_ra = sizeof(unsigned long long int) * in->nr * in->na;
	for (int i = 0; i < device_count; ++i)
	{
		out_gpu[i].rsp = out->rsp;
		out_gpu[i].a_rz = (unsigned long long int*)malloc(size_rz);
		out_gpu[i].rd_ra = (unsigned long long int*)malloc(size_ra);
		out_gpu[i].t_ra = (unsigned long long int*)malloc(size_ra);
		memset(out_gpu[i].a_rz, 0, size_rz);
		memset(out_gpu[i].rd_ra, 0, size_ra);
		memset(out_gpu[i].t_ra, 0, size_ra);
	}

	#pragma omp parallel
	{
		int device_id = omp_get_thread_num();
		KernelThreadGPU(in, &(out_gpu[device_id]), control, device_id);
	}

	int n_rz = in->nr * in->nz;
	int n_ra = in->nr * in->na;
	for (int i = 0; i < device_count; ++i)
	{
		for (int j = 0; j < n_rz; ++j)
		{
			out->a_rz[j] += out_gpu[i].a_rz[j];
		}
		
		for (int j = 0; j < n_ra; ++j)
		{
			out->rd_ra[j] += out_gpu[i].rd_ra[j];
			out->t_ra[j] += out_gpu[i].t_ra[j];
		}
	}

	for (int i = 0; i < device_count; ++i)
	{
		free(out_gpu[i].a_rz);
		free(out_gpu[i].rd_ra);
		free(out_gpu[i].t_ra);
	}
	free(out_gpu);
}
