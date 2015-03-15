#include "mcml_mcg59.h"

#define MCG59_C     302875106592253
#define MCG59_M     576460752303423488
#define MCG59_DEC_M 576460752303423487

__device__ double NextMCG59(MCG59* randomGenerator)
{
	randomGenerator->value = (randomGenerator->value * randomGenerator->offset) & MCG59_DEC_M;
	return (double)(randomGenerator->value) / MCG59_M;
}
