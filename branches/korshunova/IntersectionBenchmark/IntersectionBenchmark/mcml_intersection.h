#ifndef __MCML_INTERSECTION_H
#define __MCML_INTERSECTION_H

#include "mcml_types.h"

#define EPSILON 1E-6
#define MAX_DISTANCE 1.0E+256
#define BIN_NUMBER 16
#define MAX_TRIANGLES_IN_LEAVES 5

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
double GetTriangleIntersectionDistance(double3 origin, double3 direction, double3 a, double3 b, double3 c);
IntersectionInfo ComputeSurfaceIntersection(double3 origin, double3 direction, Surface surface);

typedef struct __BVHBoxIntersectionInfo
{
	BVHNode* node;
	double tnear;

	__BVHBoxIntersectionInfo(BVHNode* n, double near) :
			node(n), tnear(near) {}
} BVHBoxIntersectionInfo;

typedef struct __KdBoxIntersectionInfo
{
	KdTree* node;
	double tfar;

	__KdBoxIntersectionInfo(KdTree* n, double far) :
			node(n), tfar(far) {}
} KdBoxIntersectionInfo;

typedef struct __TreeInfo
{
	int Depth; //глубина
	int avDepth; //средн€€ глубина
	int leavesCount;
	int maxTrianglesInLeaf;
	int avTrianglesInLeaf;
} TreeInfo;

TreeInfo evaluateBVH(BVHNode* tree);
TreeInfo evaluateKd(KdTree* tree);

bool IntersectAABB(AABB box, double3 origin, double3 direction, double& t_near, double& t_far);
AABB createBox(Triangle* tris, int numOfTriangles, Surface* surfaces);

BVHTree* GenerateBVHTree(Surface* surfaces, int numberOfSurfaces);
IntersectionInfo ComputeBVHIntersection(double3 origin, double3 direction, double step, BVHTree* tree, Surface* surfaces);
IntersectionInfo ComputeBVHIntersectionWithoutStep(double3 origin, double3 direction, BVHTree* tree, Surface* surfaces);

KdTree* GenerateKdTree(Surface* surfaces, int numberOfSurfaces);
IntersectionInfo ComputeKDIntersectionWithoutStep(double3 origin, double3 direction, KdTree* tree, Surface* surfaces);
IntersectionInfo ComputeKDIntersection(double3 origin, double3 direction, double step, KdTree* tree, Surface* surfaces);
#endif //__MCML_INTERSECTION_H
