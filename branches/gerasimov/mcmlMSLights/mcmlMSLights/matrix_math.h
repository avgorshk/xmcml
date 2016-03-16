#ifndef __MATRIX_MATH
#define __MATRIX_MATH

#include "mcml_kernel_types.h"

#define MATRIX_3D_MODE_XY 0
#define MATRIX_3D_MODE_XZ 1
#define MATRIX_3D_MODE_YZ 2

typedef struct __InputFuncPar
{
	double A;
	double f;
	double a;
}InputFuncPar;

void MSL_2D(Area* area, double* resWeight, InputFuncPar* funcPar);
void MSL_3D(Area* area, double* resWeight, InputFuncPar* funcPar);

#endif