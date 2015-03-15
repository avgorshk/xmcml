#ifndef __MCML_INTERSECTION_H
#define __MCML_INTERSECTION_H

#include "mcml_types.h"

#define EPSILON 1E-5
#define MAX_DISTANCE FLT_MAX
#define BIN_NUMBER 16
#define MAX_TRIANGLES_IN_LEAVES 5

typedef struct __IntersectionInfo
{
    floatVec3 normal;
    float distance;
    int surfaceId;
    int isFindIntersection;
} IntersectionInfo;

//Pubic
IntersectionInfo ComputeIntersection(floatVec3 origin, floatVec3 direction, Surface* surfaces, int numberOfSurfaces);

//Private
float GetTriangleIntersectionDistance(floatVec3 origin, floatVec3 direction, floatVec3 a, floatVec3 b, floatVec3 c);
IntersectionInfo ComputeSurfaceIntersection(floatVec3 origin, floatVec3 direction, Surface& surface);

typedef struct __BVHBoxIntersectionInfo
{
	int node;
	float tnear;
} BVHBoxIntersectionInfo;

bool IntersectAABB(AABB box, floatVec3 origin, floatVec3 direction, float& t_near, float& t_far);
AABB createBox(Triangle* tris, int numOfTriangles, Surface* surfaces);

BVHTree* GenerateBVHTree(Surface* surfaces, int numberOfSurfaces);
IntersectionInfo ComputeBVHIntersection(floatVec3 origin, floatVec3 direction, float step, BVHTree* tree, Surface* surfaces);
IntersectionInfo ComputeBVHIntersectionWithoutStep(floatVec3 origin, floatVec3 direction, BVHTree* tree, Surface* surfaces);

#endif //__MCML_INTERSECTION_H
