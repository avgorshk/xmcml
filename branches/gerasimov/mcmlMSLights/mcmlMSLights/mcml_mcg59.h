#ifndef __MCML_MCG59_H
#define __MCML_MCG59_H

#include "mcml_kernel_types.h"

typedef struct __MCG59
{
	uint64 value;
	uint64 offset;
} MCG59;

//Public
void InitMCG59(MCG59* randomGenerator, uint64 seed, uint id, uint step);
double NextMCG59(MCG59* randomGenerator);

#endif //__MCML_MCG59_H
