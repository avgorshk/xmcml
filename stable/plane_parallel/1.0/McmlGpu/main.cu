#include "./kernel/kernel.cu"

extern __shared__ LayerStructGPU SharedLayerStruct[];
__global__ void GlobalKernel(InStructGPU in, OutStructGPU out, ControlStructGPU control, int device_id, int pack_size)
{
	int index = threadIdx.x;

	if (index < in.num_layers + 2)
	{
		SharedLayerStruct[index].z0 = in.layers[index].z0;
		SharedLayerStruct[index].z1 = in.layers[index].z1;
		SharedLayerStruct[index].n = in.layers[index].n;
		SharedLayerStruct[index].mua = in.layers[index].mua;
		SharedLayerStruct[index].mus = in.layers[index].mus;
		SharedLayerStruct[index].g = in.layers[index].g;
		SharedLayerStruct[index].cos_crit0 = in.layers[index].cos_crit0;
		SharedLayerStruct[index].cos_crit1 = in.layers[index].cos_crit1;
	}
	__syncthreads();
	in.layers = SharedLayerStruct;

	PhotonStructGPU photon;
	uint64 mcg59_x;
	uint64 mcg59_cn;

	int thread_id = device_id * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

	MakeMCG59(mcg59_x, mcg59_cn, control.mcg59_seed, control.mcg59_step);

    InitMCG59(mcg59_x, control.mcg59_first_id + thread_id);
	
	for (int i = 0; i < pack_size; ++i)
	{
		LaunchPhoton(in.layers, &photon, out.rsp);

		do
		{
			if (in.layers[photon.layer_index].mua == 0.0f && in.layers[photon.layer_index].mus == 0.0f)
			{
				HopInGlass(&photon, &in, &out, mcg59_x, mcg59_cn);
			}
			else
			{
				HopDropSpinInTissue(&photon, &in, &out, mcg59_x, mcg59_cn);
			}

			if ((photon.w < in.wth) && (!photon.dead))
			{
				Roulette(&photon, mcg59_x, mcg59_cn);
			}
		} while (!photon.dead);
	}
}

void CallKernelThreadGPU(InStructGPU in, OutStructGPU out, ControlStructGPU control, int device_id, int pack_size, int layers_size)
{
	int n = control.num_photons;
	dim3 blocks = dim3(n / (BLOCK_SIZE * pack_size), 1, 1);
	dim3 threads = dim3(BLOCK_SIZE, 1, 1);

	GlobalKernel<<<blocks, threads, layers_size>>>(in, out, control, device_id, pack_size);
}
