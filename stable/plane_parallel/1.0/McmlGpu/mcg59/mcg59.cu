#include "mcg59.h"

#define MCG59_C     302875106592253
#define MCG59_M     576460752303423488
#define MCG59_DEC_M 576460752303423487

__device__ uint64 CountCN(uint n)
{
	uint64 s = 1;
	uint64 a = MCG59_C;

	while (n > 0)
	{
		if ((n & 1) == 0)
		{
			a *= a;
			n >>= 1;
		}
		else
		{
			s *= a;
			--n;
		}
	}

	return s;
}

__device__ void MakeMCG59(uint64& x, uint64& cn, uint64 seed, uint step)
{
	x = 2 * seed + 1;
	cn = CountCN(step);
}

__device__ void InitMCG59(uint64& x, uint id)
{
	uint64 cn = CountCN(id);
	x = (x * cn) & MCG59_DEC_M;
}

__device__ float NextMCG59(uint64& x, uint64 cn)
{
	x = (x * cn) & MCG59_DEC_M;
	return (float)x / MCG59_M;
}
