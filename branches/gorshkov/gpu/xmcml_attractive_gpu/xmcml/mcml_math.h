#ifndef __MCML_MATH_H
#define __MCML_MATH_H

#include "mcml_types.h"

//Public
Double3 SubVector(Double3 a, Double3 b);
Double3 DivVector(Double3 a, double b);
double LengthOfVector(Double3 a);
double DotVector(Double3 a, Double3 b);
Double3 NormalizeVector(Double3 a);
Double3 CrossVector(Double3 a, Double3 b);
Double3 GetPlaneNormal(Double3 a, Double3 b, Double3 c);
Double3 GetPlaneSegmentIntersectionPoint(Double3 a, Double3 b, double z);

#endif //__MCML_MATH_H
