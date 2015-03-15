#include "convert_gpu.h"

typedef struct __ThreadArgs
{
	InputStruct* in;
	OutStruct* out;
	ControlStruct* control;
} ThreadArgs;

void ThreadGPU(ThreadArgs* args)
{
	InputStruct* in = args->in;
	OutStruct* out = args->out;
	ControlStruct* control = args->control;

	ControlStructGPU control_gpu;
	control_gpu.mcg59_first_id = control->mcg59_first_id + control->omp_threads; 
	control_gpu.mcg59_seed = control->mcg59_seed;
	control_gpu.mcg59_step = control->mcg59_step;
	control_gpu.num_photons = control->gpu_photons;
	control_gpu.gpu_devices = control->gpu_devices;

	InStructGPU* in_gpu = CreateInStructGPU(in); 
	OutStructGPU* out_gpu = CreateOutStructGPU(in_gpu);

	CountMcmlGpu(in_gpu, out_gpu, &control_gpu);

	CopyOutStructGPU(out, out_gpu, in_gpu);

	FreeInStructGPU(in_gpu);
	FreeOutStructGPU(out_gpu);
}

void CountMcml(InputStruct* in, OutStruct* out, ControlStruct* control)
{
	int r = in->nr;
	int z = in->nz;
	int a = in->na;

	OutStruct out_gpu;

	ThreadArgs args_gpu;

	if (control->gpu_devices > 0)
	{
		out_gpu.Rsp = 0.0f;
		out_gpu.A_rz = new double*[r];
		for (int i = 0; i < r; ++i)
		{
			out_gpu.A_rz[i] = new double[z];
		}
		out_gpu.Rd_ra = new double*[r];
		for (int i = 0; i < r; ++i)
		{
			out_gpu.Rd_ra[i] = new double[a];
		}
		out_gpu.Tt_ra = new double*[r];
		for (int i = 0; i < r; ++i)
		{
			out_gpu.Tt_ra[i] = new double[a];
		}

		args_gpu.in = in;
		args_gpu.out = &out_gpu;
		args_gpu.control = control;

		ThreadGPU(&args_gpu);

		out->Rsp = out_gpu.Rsp;

		for (int i = 0; i < r; ++i)
		{
			for (int j = 0; j < z; ++j)
			{
				out->A_rz[i][j] = out_gpu.A_rz[i][j];
			}
		}
		for (int i = 0; i < r; ++i)
		{
			for (int j = 0; j < a; ++j)
			{
				out->Rd_ra[i][j] = out_gpu.Rd_ra[i][j];
			}
		}
		for (int i = 0; i < r; ++i)
		{
			for (int j = 0; j < a; ++j)
			{
				out->Tt_ra[i][j] = out_gpu.Tt_ra[i][j];
			}
		}
	}
}