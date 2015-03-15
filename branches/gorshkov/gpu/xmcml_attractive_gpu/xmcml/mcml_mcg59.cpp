#include "mcml_mcg59.h"

#define MCG59_C     302875106592253
#define MCG59_M     576460752303423488
#define MCG59_DEC_M 576460752303423487

uint64 RaiseToPower(uint64 argument, uint power)
{
	uint64 result = 1;

	while (power > 0)
	{
		if ((power & 1) == 0)
		{
			argument *= argument;
			power >>= 1;
		}
		else
		{
			result *= argument;
			--power;
		}
	}

	return result;
}

void InitMCG59(MCG59* randomGenerator, uint64 seed, uint id, uint step)
{
	uint64 value = 2 * seed + 1;
	uint64 firstOffset = RaiseToPower(MCG59_C, id);
	value = (value * firstOffset) & MCG59_DEC_M;
	randomGenerator->value = value;
	randomGenerator->offset = RaiseToPower(MCG59_C, step);
}
