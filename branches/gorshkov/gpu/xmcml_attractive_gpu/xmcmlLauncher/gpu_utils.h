#ifndef __GPU_UTILS_H
#define __GPU_UTILS_H

#include "..\xmcml\mcml_kernel_types.h"
#include "..\xmcml\mcml_mcg59.h"

InputInfo* CopyInputToGPU(InputInfo* input);
void ReleaseGpuInput(InputInfo* gpuInput);
MCG59* CopyRandomGeneratorsToGPU(MCG59* randomGenerators, int numberOfThreads);
void ReleaseGpuRandomGenerators(MCG59* gpuRandomGenerators);
void SetGpuDevice(int mpi_rank);

#endif //__GPU_UTILS_H
