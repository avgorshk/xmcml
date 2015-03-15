#ifndef __MCML_INTERSECTION_H
#define __MCML_INTERSECTION_H

#include "mcml_types.h"

#define EPSILON 1E-6
#define MAX_DISTANCE 1.0E+256
#define BIN_NUMBER 16
#define MAX_TRIANGLES_IN_LEAVES 5

typedef struct __BVHBoxIntersectionInfo
{
	int node;
	double tnear;
} BVHBoxIntersectionInfo;

typedef struct __IntersectionInfo
{
    Double3 normal;
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
	Double3 Ver1; 
	Double3 Ver2; 
} AABB;

typedef struct __BVHNode
{
	AABB Box;
    int rightNode;
    int leftNode;
    int offset;
	int numOfTriangles;

	__BVHNode() 
    { 
		rightNode = leftNode = -1;
		numOfTriangles = 0;
        offset = 0;
	}
} BVHNode;

typedef struct __BVHTree
{
public:
    BVHNode* nodes;
    int numOfNodes;
	int root;
	Triangle* triangles;
    int numOfTriangles;

    __BVHTree()
    {
        nodes = 0;
        triangles = 0;
        root = numOfNodes = numOfTriangles = 0;
    }

	void Release() 
    {  
        if (nodes != 0)
            delete[] nodes;
        if (triangles != 0)
            delete[] triangles;
	}
} BVHTree;

BVHTree* GenerateBVHTree(Surface* surfaces, int numberOfSurfaces);

#endif //__MCML_INTERSECTION_H
