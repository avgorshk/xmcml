#ifndef __CONVERT_GPU_H
#define __CONVERT_GPU_H

#include "mcml_getero.h"
#include "mcml_gpu.h"

InStructGPU* CreateInStructGPU(InputStruct* in);
void FreeInStructGPU(InStructGPU* in_gpu);

OutStructGPU* CreateOutStructGPU(InStructGPU* in_gpu);
void CopyOutStructGPU(OutStruct* out, OutStructGPU* out_gpu, InStructGPU* in_gpu);
void FreeOutStructGPU(OutStructGPU* out_gpu);

#endif //__CONVERT_GPU_H