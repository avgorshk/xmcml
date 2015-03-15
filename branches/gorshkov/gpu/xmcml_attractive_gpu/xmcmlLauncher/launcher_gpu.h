#ifndef __LAUNCHER_GPU_H
#define __LAUNCHER_GPU_H

#include "../xmcml/mcml_kernel.h"
#include "../xmcml/mcml_mcg59.h"

void InitOutput(InputInfo* input, OutputInfo* output);
void FreeOutput(OutputInfo* output);
void LaunchGPU(InputInfo* input, OutputInfo* output, MCG59* randomGenerator);

#endif //__LAUNCHER_GPU_H
