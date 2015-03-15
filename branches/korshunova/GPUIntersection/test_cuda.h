#ifndef __TEST_CUDA_H__
#define __TEST_CUDA_H__

#include "mcml_intersection.h"
#include <cuda_runtime.h>

typedef struct __GpuBVH
{
    BVHNode* nodes;
    int numOfNodes;
	int root;
	Triangle* triangles;
    int numOfTriangles;
} GpuBVH;

typedef struct __GPUSurface
{
	floatVec3* vertices;    
	int numberOfVertices;
    int3* triangles;       
    int numberOfTriangles; 
} GPUSurface;

void  GpuBVHIntersections(floatVec3 *origin, floatVec3 *direction, float* step, int numRays, BVHTree* tree, Surface* surfaces, int numSurfaces, IntersectionInfo *results, float &computeTime, float &fullTime);
GpuBVH createGpuBVH(BVHTree* tree);
void releaseGpuBVH(GpuBVH &gpuTree);
GPUSurface* copySurfacesToGPU(Surface* surfaces, int numSurfaces);
void releaseGPUSurfaces(GPUSurface* gpuSurfaces, int numSurfaces);

#endif //__TEST_CUDA_H__
