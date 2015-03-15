#ifndef __MCML_MATH_H
#define __MCML_MATH_H

#include <math.h>
#include "mcml_types.h"

//Public
floatVec3 SubVector(floatVec3 a, floatVec3 b);
float LengthOfVector(floatVec3 a);
float DotVector(floatVec3 a, floatVec3 b);
floatVec3 NormalizeVector(floatVec3 a);
floatVec3 CrossVector(floatVec3 a, floatVec3 b);
floatVec3 GetPlaneNormal(floatVec3 a, floatVec3 b, floatVec3 c);
floatVec3 GetPlaneSegmentIntersectionPoint(floatVec3 a, floatVec3 b, float z);

#endif //__MCML_MATH_H
