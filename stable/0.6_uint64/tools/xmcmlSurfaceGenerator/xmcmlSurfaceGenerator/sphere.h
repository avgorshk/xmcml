#ifndef __SPHERE_H
#define __SPHERE_H

#include "..\..\..\xmcml\xmcml\xmcml\mcml_kernel_types.h"

void GenerateSphere(double3 center, double radius, int numberOfSubdividings, Surface* sphere);
void GenerateTopHalfSphere(double3 center, double radius, int numberOfSubdividings, Surface* sphere);
void GenerateBottomHalfSphere(double3 center, double radius, int numberOfSubdividings, Surface* sphere);

#endif //__SPHERE_H
