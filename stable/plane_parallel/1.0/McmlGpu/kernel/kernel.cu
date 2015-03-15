#include <sm_12_atomic_functions.h>
#include "kernel.h"
#include "../mcg59/mcg59.cu"

#define SIGN(x) ((x) >= 0 ? 1 : -1)

#define PI       3.14159f 
#define WEIGHT   0.0001f
#define CHANCE   0.1f
#define COS_ZERO 0.99999f
#define COS_90D  0.00001f
#define COEFF    4294967296

__host__ float RSpecular(LayerStructGPU* layer)
{
	float r1, r2;
	float sqrt_r1, sqrt_r2;

	sqrt_r1 = (layer[0].n - layer[1].n) / (layer[0].n + layer[1].n);
	r1 = sqrt_r1 * sqrt_r1;

	if (layer[1].mua == 0.0 && layer[1].mus == 0.0)
	{
		sqrt_r2 = (layer[1].n - layer[2].n) / (layer[1].n + layer[2].n);
		r2 = sqrt_r2 * sqrt_r2;
		r1 = r1 + (1 - r1) * (1 - r1) * r2 / (1 - r1 * r2);
	}

	return r1;
}

__device__ void LaunchPhoton(LayerStructGPU* layer, PhotonStructGPU* photon, float rsp)
{
	photon->w = 1.0f - rsp;
	photon->dead = 0;
	photon->layer_index = 1;
	photon->s = 0.0f;
	photon->sleft = 0.0f;

	photon->x = 0.0f;
	photon->y = 0.0f;
	photon->z = 0.0f;

	photon->ux = 0.0f;
	photon->uy = 0.0f;
	photon->uz = 1.0f;

	if (layer[1].mua == 0.0f && layer[1].mus == 0.0f)
	{
		photon->layer_index = 2;
		photon->z = layer[2].z0;
	}
}

__device__ float SpinTheta(float g, uint64& mcg59_x, uint64 mcg59_cn)
{
	float cost;
    float temp;

    float rnd = NextMCG59(mcg59_x, mcg59_cn);

	if (g == 0.0f)
	{
		cost = 2.0f * rnd - 1.0f;
	}
	else
	{
		temp = (1.0f - g * g) / (1.0f - g + 2 * g * rnd);
		cost = (1.0f + g * g - temp * temp) / (g + g);
	}

    if (cost > 1.0f) cost = 1.0f;
    else if (cost < -1.0f) cost = -1.0f;

	return cost;
}

__device__ void Spin(PhotonStructGPU* photon, float g, uint64& mcg59_x, uint64 mcg59_cn)
{
	float cost, sint;
	float cosp, sinp;

	float ux = photon->ux;
	float uy = photon->uy;
	float uz = photon->uz;
	
	float psi;

    cost = SpinTheta(g, mcg59_x, mcg59_cn);
	sint = sqrtf(1.0f - cost * cost);

	psi = 2.0f * PI * NextMCG59(mcg59_x, mcg59_cn);
	cosp = cosf(psi);
	sinp = sinf(psi);
	
	if (fabs(uz) > COS_ZERO)
	{
		photon->ux = sint * cosp;
		photon->uy = sint * sinp;
		photon->uz = cost * SIGN(uz);
	}
	else
	{
		float temp = sqrtf(1.0f - uz * uz);
		photon->ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost;
		photon->uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost;
		photon->uz = - sint * cosp * temp + uz * cost;
	}
}

__device__ void Hop(PhotonStructGPU* photon)
{
	float s = photon->s;

	photon->x += photon->ux * s;
	photon->y += photon->uy * s;
	photon->z += photon->uz * s;
}

__device__ void StepSizeInGlass(PhotonStructGPU* photon, InStructGPU* in)
{
	float dl_b;
	uint layer_index = photon->layer_index;
	float uz = photon->uz;

	if (uz > 0.0f)
		dl_b = (in->layers[layer_index].z1 - photon->z) / uz;
	else if (uz < 0.0f)
		dl_b = (in->layers[layer_index].z0 - photon->z) / uz;
	else
		dl_b = 0.0f;

	photon->s = dl_b;
}

__device__ void StepSizeInTissue(PhotonStructGPU* photon, InStructGPU* in, uint64& mcg59_x, uint64 mcg59_cn)
{
	uint layer_index = photon->layer_index;
	float mua = in->layers[layer_index].mua;
	float mus = in->layers[layer_index].mus;

	if (photon->sleft == 0.0f)
	{
		float rnd;

		do
		{
			rnd = NextMCG59(mcg59_x, mcg59_cn);
		}
		while (rnd <= 0.0f);

		photon->s =	-logf(rnd) / (mua + mus);
	}
	else
	{
		photon->s = photon->sleft / (mua + mus);
		photon->sleft = 0.0f;
	}
}

__device__ byte HitBoundary(PhotonStructGPU* photon, InStructGPU* in)
{
	float dl_b;
	uint layer_index = photon->layer_index;
	float uz = photon->uz;
	byte hit;

	if (uz > 0.0f)
		dl_b = (in->layers[layer_index].z1 - photon->z) / uz;
	else if (uz < 0.0f)
		dl_b = (in->layers[layer_index].z0 - photon->z) / uz;

	if ((uz != 0.0f) && (photon->s > dl_b))
	{
		float mut = in->layers[layer_index].mua + in->layers[layer_index].mus;
		photon->sleft = (photon->s - dl_b) * mut;
		photon->s = dl_b;
		hit = 1;
	}
	else
		hit = 0;

	return hit;
}

__device__ void Drop(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out)
{
	float dwa;
	float x = photon->x;
	float y = photon->y;
	uint iz, ir;
	uint layer_index = photon->layer_index;
	float mua, mus;
	uint64 ull;

	iz = (uint)(photon->z / in->dz);
	if (iz > in->nz - 1)
		iz = in->nz - 1;

	ir = (uint)(sqrt(x * x + y * y) / in->dr);
	if (ir > in->nr - 1)
		ir = in->nr - 1;

	mua = in->layers[layer_index].mua;
	mus = in->layers[layer_index].mus;
	dwa = photon->w * mua / (mua + mus);
	photon->w -= dwa;

	ull = (uint64)(dwa * COEFF);

	atomicAdd(out->a_rz + ir * in->nz + iz, ull);
}

__device__ void Roulette(PhotonStructGPU* photon, uint64& mcg59_x, uint64 mcg59_cn)
{
	if (photon->w == 0.0f)
		photon->dead = 1;
	else if (NextMCG59(mcg59_x, mcg59_cn) < CHANCE)
		photon->w /= CHANCE;
	else
		photon->dead = 1;
}

__device__ float RFrensel(float n1, float n2, float ca1, float* ca2_ptr)
{
	float r;

	if (n1 == n2)
	{
		*ca2_ptr = ca1;
		r = 0.0f;
	}
	else if (ca1 > COS_ZERO)
	{
		*ca2_ptr = ca1;
		r = (n2 - n1) / (n2 + n1);
		r *= r;
	}
	else if (ca1 < COS_90D)
	{
		*ca2_ptr = 0.0f;
		r = 1.0f;
	}
	else
	{
		float sa1, sa2;
		float ca2;

		sa1 = sqrtf(1.0f - ca1 * ca1);
		sa2 = n1 * sa1 / n2;

		if (sa2 >= 1.0f)
		{
			*ca2_ptr = 0.0f;
			r = 1.0f;
		}
		else
		{
			float cap, cam;
			float sap, sam;

			*ca2_ptr = ca2 = sqrtf(1 - sa2 * sa2);

			cap = ca1 * ca2 - sa1 * sa2;
			cam = ca1 * ca2 + sa1 * sa2;
			sap = sa1 * ca2 + sa2 * ca1;
			sam = sa1 * ca2 - ca1 * sa2;
			r = 0.5f * sam * sam * (cam * cam + cap * cap)/ (sap * sap * cam * cam);
		}
	}

	return r;
}

__device__ void RecordR(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out, float refl)
{
	float x = photon->x;
	float y = photon->y;
	uint ir, ia;
	uint64 ull;

	ir = (uint)(sqrt(x * x + y * y) / in->dr);
	if (ir > in->nr - 1)
		ir = in->nr - 1;

	ia = (uint)(acos(-photon->uz) / in->da);
	if (ia > in->na - 1)
		ia = in->na - 1;

	ull = (uint64)(photon->w * (1.0f - refl) * COEFF);

	atomicAdd(out->rd_ra + ir * in->na + ia, ull);

	photon->w *= refl;
}

__device__ void RecordT(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out, float refl)
{
	float x = photon->x;
	float y = photon->y;
	uint ir, ia;
	uint64 ull;

	ir = (uint)(sqrt(x * x + y * y) / in->dr);
	if (ir > in->nr - 1)
		ir = in->nr - 1;

	ia = (uint)(acos(-photon->uz) / in->da);
	if (ia > in->na - 1)
		ia = in->na - 1;

	ull = (uint64)(photon->w * (1.0f - refl) * COEFF);

	atomicAdd(out->t_ra + ir * in->na + ia, ull);

	photon->w *= refl;
}

__device__ void CrossUpOrNot(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out,
	uint64& mcg59_x, uint64 mcg59_cn)
{
	float uz = photon->uz;
	float uz1;
	float r = 0.0f;
	uint layer_index = photon->layer_index;
	float ni = in->layers[layer_index].n;
	float nt = in->layers[layer_index - 1].n;

	if ( - uz <= in->layers[layer_index].cos_crit0)
		r = 1.0f;
	else
		r = RFrensel(ni, nt, -uz, &uz1);

	if (NextMCG59(mcg59_x, mcg59_cn) > r)
	{
		if (layer_index == 1)
		{
			photon->uz = - uz1;
			RecordR(photon, in, out, 0.0f);
			photon->dead = 1;
		}
		else
		{
			--(photon->layer_index);
			photon->ux *= ni / nt;
			photon->uy *= ni / nt;
			photon->uz = - uz1;
		}
	}
	else
	{
		photon->uz = -uz;
	}
}

__device__ void CrossDnOrNot(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out,
	uint64& mcg59_x, uint64 mcg59_cn)
{
	float uz = photon->uz;
	float uz1;
	float r = 0.0f;
	uint layer_index = photon->layer_index;
	float ni = in->layers[layer_index].n;
	float nt = in->layers[layer_index + 1].n;

	if ( uz <= in->layers[layer_index].cos_crit1)
		r = 1.0f;
	else
		r = RFrensel(ni, nt, uz, &uz1);

	if (NextMCG59(mcg59_x, mcg59_cn) > r)
	{
		if (layer_index == in->num_layers)
		{
			photon->uz = uz1;
			RecordT(photon, in, out, 0.0f);
			photon->dead = 1;
		}
		else
		{
			++(photon->layer_index);
			photon->ux *= ni / nt;
			photon->uy *= ni / nt;
			photon->uz = uz1;
		}
	}
	else
	{
		photon->uz = -uz;
	}
}

__device__ void CrossOrNot(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out,
	uint64& mcg59_x, uint64 mcg59_cn)
{
	if (photon->uz < 0.0f)
		CrossUpOrNot(photon, in, out, mcg59_x, mcg59_cn);
	else
		CrossDnOrNot(photon, in, out, mcg59_x, mcg59_cn);
}

__device__ void HopInGlass(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out,
	uint64& mcg59_x, uint64 mcg59_cn)
{
	if (photon->uz == 0.0f)
	{
		photon->dead = 1;
	}
	else
	{
		StepSizeInGlass(photon, in);
		Hop(photon);
		CrossOrNot(photon, in, out, mcg59_x, mcg59_cn);
	}
}

__device__ void HopDropSpinInTissue(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out,
	uint64& mcg59_x, uint64 mcg59_cn)
{
	StepSizeInTissue(photon, in, mcg59_x, mcg59_cn);

	if (HitBoundary(photon, in))
	{
		Hop(photon);
		CrossOrNot(photon, in, out, mcg59_x, mcg59_cn);
	}
	else
	{
		Hop(photon);
		Drop(photon, in, out);
		Spin(photon, in->layers[photon->layer_index].g, mcg59_x, mcg59_cn);
	}
}

__device__ void HopDropSpin(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out,
	uint64& mcg59_x, uint64 mcg59_cn)
{
    uint layer_index = photon->layer_index;

    if (in->layers[layer_index].mua == 0.0f && in->layers[layer_index].mus == 0.0f)
    {
        HopInGlass(photon, in, out, mcg59_x, mcg59_cn);
    }
    else
    {
        HopDropSpinInTissue(photon, in, out, mcg59_x, mcg59_cn);
    }

    if ((photon->w < in->wth) && (! photon->dead))
    {
        Roulette(photon, mcg59_x, mcg59_cn);
    }
}
