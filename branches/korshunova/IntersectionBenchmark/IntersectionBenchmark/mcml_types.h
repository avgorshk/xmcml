#ifndef __MCML_TYPES_H
#define __MCML_TYPES_H

typedef unsigned char byte;
typedef unsigned int uint;
typedef unsigned long long int uint64;

typedef union __double3 
{
	struct { double x, y, z; };
	struct { double cell[3]; };
} double3;

typedef union __int3
{
	struct { int x, y, z; };
	struct { int cell[3]; };
} int3;

typedef struct __Area
{
	double3 corner;
	double3 length;
	int3 partitionNumber;
} Area;

typedef struct __Surface
{
	double3* vertices;     //массив точек(вершин) в трехмерном пространстве
	int numberOfVertices;
    int3* triangles;       //массив треугольников, каждый элемент содержит 
                           //индексы 3-х вершин треугольника в массиве vertices
    int numberOfTriangles; 
} Surface;

typedef struct __Triangle
{
	int surfId;	//номер поверхности
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

typedef struct __KdTree
{
	AABB Box;
	__KdTree* rightNode;
	__KdTree* leftNode;
	Triangle* triangles;
	int numOfTriangles;
	int splitAxis;
	double splitPos;

	__KdTree() { 
		rightNode = leftNode = 0;
		triangles = 0;
		numOfTriangles = 0;
	}

	~__KdTree() {
		if (rightNode != 0)
			delete rightNode;
		if (leftNode != 0)
			delete leftNode;
		if (triangles != 0)
			delete[] triangles;
	}
} KdTree;

#endif //__MCML_TYPES_H
