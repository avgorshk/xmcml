#ifndef __CYLINDER_H
#define __CYLINDER_H

#include "..\..\..\xmcml\xmcml\mcml_kernel_types.h"

void GenerateCylinderYZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder);
void GenerateCylinderXZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder);
void GenerateTopHalfCylinderYZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder);
void GenerateTopHalfCylinderXZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder);
void GenerateBottomHalfCylinderYZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder);
void GenerateBottomHalfCylinderXZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder);

#endif //__CYLINDER_H
