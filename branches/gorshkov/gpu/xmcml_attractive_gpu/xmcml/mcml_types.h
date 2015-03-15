#ifndef __MCML_TYPES_H
#define __MCML_TYPES_H

typedef unsigned char byte;
typedef unsigned int uint;
typedef unsigned long long int uint64;

typedef union __double3 
{
	struct { double x, y, z; };
	struct { double cell[3]; };
} Double3;

typedef union __int3
{
	struct { int x, y, z; };
	struct { int cell[3]; };
} Int3;

typedef union __byte3
{
    struct { byte x, y, z; };
    struct { byte cell[3]; };
} Byte3;

typedef struct __Area
{
	Double3 corner;
	Double3 length;
	Int3 partitionNumber;
} Area;

typedef struct __Surface
{
	Double3* vertices;
	int numberOfVertices;
    Int3* triangles;
    int numberOfTriangles;
} Surface;

#endif //__MCML_TYPES_H
