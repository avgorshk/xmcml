#ifndef __MCML_MATH_H
#define __MCML_MATH_H

#include "mcml_types.h"

//Public
double3 SubVector(double3 a, double3 b);
double LengthOfVector(double3 a);
double DotVector(double3 a, double3 b);
double3 NormalizeVector(double3 a);
double3 CrossVector(double3 a, double3 b);
double3 GetPlaneNormal(double3 a, double3 b, double3 c);

#endif //__MCML_MATH_H
