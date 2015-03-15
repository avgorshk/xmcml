#ifndef __MCG59_H
#define __MCG59_H

#include "../types_gpu.h"

void MakeMCG59(uint64& x, uint64& cn, uint64 seed, uint step);
void InitMCG59(uint64& x, uint id);
float NextMCG59(uint64& x, uint64 cn);

#endif //__MCG59_H