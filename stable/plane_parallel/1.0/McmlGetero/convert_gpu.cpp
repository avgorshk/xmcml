#include "convert_gpu.h"

void CopyLayerStructGPU(LayerStructGPU* layer_gpu, LayerStruct* layer)
{
	layer_gpu->cos_crit0 = layer->cos_crit0;
	layer_gpu->cos_crit1 = layer->cos_crit1;
	layer_gpu->g = layer->g;
	layer_gpu->mua = layer->mua;
	layer_gpu->mus = layer->mus;
	layer_gpu->n = layer->n;
	layer_gpu->z0 = layer->z0;
	layer_gpu->z1 = layer->z1;
}

void CopyInStructGPU(InStructGPU* in_gpu, InputStruct* in)
{
	in_gpu->da = in->da;
	in_gpu->dr = in->dr;
	in_gpu->dz = in->dz;

	in_gpu->na = in->na;
	in_gpu->nr = in->nr;
	in_gpu->nz = in->nz;

	in_gpu->wth = in->Wth;
	in_gpu->num_layers = in->num_layers;

	for (int i = 0; i < in->num_layers + 2; ++i)
	{
		CopyLayerStructGPU(in_gpu->layers + i, in->layerspecs + i);
	}
}

InStructGPU* CreateInStructGPU(InputStruct* in)
{
	InStructGPU* in_gpu = new InStructGPU();
	in_gpu->layers = new LayerStructGPU[in->num_layers + 2];
	CopyInStructGPU(in_gpu, in);
	return in_gpu; 
}

void FreeInStructGPU(InStructGPU* in_gpu)
{
	delete[] in_gpu->layers;
	delete in_gpu;
	in_gpu = 0;
}

OutStructGPU* CreateOutStructGPU(InStructGPU* in_gpu)
{
	OutStructGPU* out_gpu = new OutStructGPU();
	
	int rz = in_gpu->nr * in_gpu->nz;
	int ra = in_gpu->nr * in_gpu->na;
	
	out_gpu->rsp = 0;
	out_gpu->a_rz = new unsigned long long int[rz];
	out_gpu->rd_ra = new unsigned long long int[ra];
	out_gpu->t_ra = new unsigned long long int[ra];

	for (int i = 0; i < rz; ++i)
	{
		out_gpu->a_rz[i] = 0;		
	}

	for (int i = 0; i < ra; ++i)
	{
		out_gpu->rd_ra[i] = 0;		
		out_gpu->t_ra[i] = 0;
	}

	return out_gpu;
}

void CopyOutStructGPU(OutStruct* out, OutStructGPU* out_gpu, InStructGPU* in_gpu)
{
	int r = in_gpu->nr;
	int z = in_gpu->nz;
	int a = in_gpu->na;
	
	out->Rsp = out_gpu->rsp;

	for (int i = 0; i < r; ++i)
	{
		for (int j = 0; j < z; ++j)
		{
			out->A_rz[i][j] = (double)out_gpu->a_rz[i * z + j] / COEFF;
		}
	}

	for (int i = 0; i < r; ++i)
	{
		for (int j = 0; j < a; ++j)
		{
			out->Tt_ra[i][j] = (double)out_gpu->t_ra[i * a + j] / COEFF;
			out->Rd_ra[i][j] = (double)out_gpu->rd_ra[i * a + j] / COEFF;
		}
	}
}

void FreeOutStructGPU(OutStructGPU* out_gpu)
{
	delete[] out_gpu->a_rz;
	delete[] out_gpu->rd_ra;
	delete[] out_gpu->t_ra;
	delete out_gpu;
	out_gpu = 0;
}
