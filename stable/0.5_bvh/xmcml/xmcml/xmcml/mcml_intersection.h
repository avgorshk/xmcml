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

typedef struct __Triangle
{
	int surfId;
	int trIndex;
} Triangle;

typedef struct __AABB
{
	double3 Ver1; 
	double3 Ver2; 
} AABB;

typedef struct __BVHNode
{
	AABB Box;
	__BVHNode* rightNode;
	__BVHNode* leftNode;
	Triangle* triangles;
	int numOfTriangles;

	__BVHNode() { 
		rightNode = leftNode = 0;
		triangles = 0;
		numOfTriangles = 0;
	}

	~__BVHNode() {
		if (rightNode != 0)
			delete rightNode;
		if (leftNode != 0)
			delete leftNode;
	}
} BVHNode;

typedef struct __BVHTree
{
public:
	BVHNode* root;
	Triangle* triangles;

	~__BVHTree() {  
		if (root != 0)
			delete root;
		if (triangles != 0)
			delete[] triangles;
	}
} BVHTree;

typedef struct __BVHBoxIntersectionInfo
{
	BVHNode* node;
	double tnear;

	__BVHBoxIntersectionInfo(BVHNode* n, double near) :
			node(n), tnear(near) {}
} BVHBoxIntersectionInfo;

typedef struct __TreeInfo
{
	int Depth;
	int avDepth;
	int leavesCount;
	int maxTrianglesInLeaf;
	int avTrianglesInLeaf;
} TreeInfo;

//Pubic
BVHTree* GenerateBVHTree(Surface* surfaces, int numberOfSurfaces);
IntersectionInfo ComputeBVHIntersection(double3 origin, double3 direction, 
    double step, BVHTree* tree, Surface* surfaces);
IntersectionInfo ComputeBVHIntersectionWithoutStep(double3 origin, double3 direction, 
    BVHTree* tree, Surface* surfaces);

//Private
double GetTriangleIntersectionDistance(double3 origin, double3 direction, double3 a, double3 b, double3 c);

#endif //__MCML_INTERSECTION_H
