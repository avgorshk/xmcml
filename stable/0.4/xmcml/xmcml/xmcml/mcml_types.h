#ifndef __MCML_TYPES_H
#define __MCML_TYPES_H

typedef unsigned char byte;
typedef unsigned int uint;
typedef unsigned long long int uint64;

typedef struct __double3
{
	double x, y, z;
} double3;

typedef struct __int3
{
	int x, y, z;
} int3;

typedef struct __Area
{
	double3 corner;
	double3 length;
	int3 partitionNumber;
} Area;

typedef struct __Surface
{
	double3* vertices;
	int numberOfVertices;
    int3* triangles;
    int numberOfTriangles;
} Surface;

#endif //__MCML_TYPES_H
