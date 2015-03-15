#ifndef __TYPES_GPU_H
#define __TYPES_GPU_H

typedef unsigned long long int uint64;
typedef unsigned int uint;
typedef unsigned char byte;

typedef struct  __PhotonStructGPU
{
	float x, y, z;
	float ux, uy, uz;
	float w;
	float s;
	float sleft;
	uint layer_index;
	byte dead;
} PhotonStructGPU;

typedef struct  __LayerStructGPU
{
	float z0, z1;
	float n;
	float mua;
	float mus;
	float g;
	float cos_crit0, cos_crit1;
} LayerStructGPU;

typedef struct  __InStructGPU
{
	float wth;
	float dz, dr, da;
	uint nz, nr, na;
	uint num_layers;
	LayerStructGPU* layers;
} InStructGPU;

typedef struct  __OutStructGPU
{
    float rsp;
	uint64* rd_ra;
	uint64* a_rz;
	uint64* t_ra;
} OutStructGPU;

typedef struct  __ControlStructGPU
{
	uint mcg59_step;
	uint mcg59_first_id;
	uint mcg59_seed;
	uint num_photons;
	int gpu_devices;
} ControlStructGPU;

#endif //__TYPES_GPU_H
