#ifndef __MCML_TYPES_H
#define __MCML_TYPES_H

#include <vector_types.h>
#include <vector>

typedef unsigned char byte;
typedef unsigned int uint;
typedef unsigned long long int uint64;

typedef union __floatVec3
{
	struct { float x, y, z; };
	struct { float cell[3]; };
} floatVec3;

typedef struct __Area
{
	floatVec3 corner;
	floatVec3 length;
	int3 partitionNumber;
} Area;

typedef struct __Surface
{
	floatVec3* vertices;     //массив точек(вершин) в трехмерном пространстве
	int numberOfVertices;
    int3* triangles;       //массив треугольников, каждый элемент содержит 
                           //индексы 3-х вершин треугольника в массиве vertices
    int numberOfTriangles; 

    ~__Surface()
    {
        if (vertices != 0)
            delete[] vertices;
        if (triangles != 0)
            delete[] triangles;
    }
} Surface;

typedef struct __Triangle
{
	int surfId;	//номер поверхности
	int trIndex;
} Triangle;

typedef struct __AABB
{
	floatVec3 Ver1; 
	floatVec3 Ver2; 
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

	~__BVHTree() 
    {  
        if (nodes != 0)
            delete[] nodes;
        if (triangles != 0)
            delete[] triangles;
	}
} BVHTree;

#endif //__MCML_TYPES_H
