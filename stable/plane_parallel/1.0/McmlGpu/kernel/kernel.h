#ifndef __KERNEL_H
#define __KERNEL_H

#include "../types_gpu.h"

#define BLOCK_SIZE       512
#define NUMBER_OF_BLOCKS 1024

float RSpecular(LayerStructGPU* layer);

void LaunchPhoton(LayerStructGPU* layer, PhotonStructGPU* photon, float rsp);

void HopDropSpin(PhotonStructGPU* photon, InStructGPU* in, OutStructGPU* out,
	uint64& mcg59_x, uint64 mcg59_cn);

#endif //__KERNEL_H
