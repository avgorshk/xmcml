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

//Opt1
typedef union __byte3
{
	struct { int x, y, z; };
	struct { int cell[3]; };
} byte3;

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
