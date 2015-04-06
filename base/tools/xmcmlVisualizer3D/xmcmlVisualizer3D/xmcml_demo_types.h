#ifndef __XMCML_DEMO_TYPES_H
#define __XMCML_DEMO_TYPES_H

typedef unsigned long long int uint64;
typedef unsigned int uint;

typedef struct __float3
{
    float x;
    float y;
    float z;
} float3;

typedef struct __int3
{
    int x;
    int y;
    int z;
} int3;

typedef struct __area
{
    float3 corner;
    float3 length;
    int3 size;
} area;

#endif //__XMCML_DEMO_TYPES_H