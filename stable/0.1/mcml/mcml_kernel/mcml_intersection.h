#ifndef __MCML_INTERSECTION_H
#define __MCML_INTERSECTION_H

#include "mcml_types.h"

typedef struct __IntersectionInfo
{
    double3 normal;
    double distance;
    int surfaceId;
    int isFindIntersection;
} IntersectionInfo;

//Pubic
IntersectionInfo ComputeIntersection(double3 origin, double3 direction, Surface* surfaces, int numberOfSurfaces);

//Private
double GetTriangleIntersectionDistance(double3 origin, double3 direction, double3* vertices);
IntersectionInfo ComputeSurfaceIntersection(double3 origin, double3 direction, Surface surface);

#endif //__MCML_INTERSECTION_H
