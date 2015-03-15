#ifndef __LAUNCHER_OMP_H
#define __LAUNCHER_OMP_H

#include "..\..\xmcml\xmcml\mcml_kernel_types.h"
#include "..\..\xmcml\xmcml\mcml_mcg59.h"

int GetMaxThreads();
void InitOutput(InputInfo* input, OutputInfo* output);
void FreeOutput(OutputInfo* output);
void LaunchOMP(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, int numThreads);

#endif //__LAUNCHER_OMP_H
