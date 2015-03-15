#ifndef __LAUNCHER_MPI_H
#define __LAUNCHER_MPI_H

#include "..\..\xmcml\xmcml\mcml_kernel_types.h"
#include "..\..\xmcml\xmcml\mcml_mcg59.h"

void LaunchMPI(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, int numThreadsPerProcess);

#endif //__LAUNCHER_MPI_H