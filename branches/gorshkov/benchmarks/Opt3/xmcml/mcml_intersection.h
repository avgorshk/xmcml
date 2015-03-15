#ifndef __MCML_INTERSECTION_H
#define __MCML_INTERSECTION_H

#include "mcml_kernel_types.h"

typedef struct __IntersectionInfo
{
    double3 normal;
    double distance;
    int surfaceId;
    int isFindIntersection;
} IntersectionInfo;

//Pubic
IntersectionInfo GetIntersectionInfo(PhotonState* photon, InputInfo* input);

//Private
double GetTriangleIntersectionDistance(double3 origin, double3 direction, double3 a, double3 b, double3 c);
IntersectionInfo ComputeSurfaceIntersection(double3 origin, double3 direction, Surface surface);
IntersectionInfo ComputeIntersection(double3 origin, double3 direction, Surface* surfaces, int numberOfSurfaces);

#endif //__MCML_INTERSECTION_H
