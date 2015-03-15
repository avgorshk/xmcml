#ifndef __CIRCLE_H
#define __CIRCLE_H

#include "..\..\..\xmcml\xmcml\xmcml\mcml_kernel_types.h"

void GenerateCircleYZ(double3 center, double radius, int numberOfSubdividings, Surface* circle);
void GenerateCircleXZ(double3 center, double radius, int numberOfSubdividings, Surface* circle);

#endif //__CIRCLE_H
