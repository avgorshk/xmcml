#include "test_cuda.h"
#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

GpuBVH createGpuBVH(BVHTree* tree)
{
    GpuBVH gpuTree;
    int nBytesForTriangles = tree->numOfTriangles * sizeof(Triangle);
    int nBytesForNodes = tree->numOfNodes * sizeof(BVHNode);
    Triangle* gpuTriangles;
    BVHNode* gpuNodes;
    
    cudaMalloc((void**) &gpuTriangles, nBytesForTriangles);
    cudaMemcpy(gpuTriangles, tree->triangles, nBytesForTriangles, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpuNodes, nBytesForNodes);
    cudaMemcpy(gpuNodes, tree->nodes, nBytesForNodes, cudaMemcpyHostToDevice);

    gpuTree.numOfTriangles = tree->numOfTriangles;
    gpuTree.numOfNodes = tree->numOfNodes;
    gpuTree.root = tree->root;
    gpuTree.nodes = gpuNodes;
    gpuTree.triangles = gpuTriangles;
    
    return gpuTree;
}

void releaseGpuBVH(GpuBVH &gpuTree)
{
	cudaFree(gpuTree.triangles);
    cudaFree(gpuTree.nodes);
}

GPUSurface* copySurfacesToGPU(Surface* surfaces, int numSurfaces)
{
    GPUSurface* gpuSurfaces;    

    GPUSurface* tmpSurfaces = new GPUSurface[numSurfaces];
    for (int i = 0; i < numSurfaces; i++)
    {
        tmpSurfaces[i].numberOfVertices = surfaces[i].numberOfVertices;
        tmpSurfaces[i].numberOfTriangles = surfaces[i].numberOfTriangles;
        int nBytesForVertices = surfaces[i].numberOfVertices * sizeof(floatVec3);
        int nBytesForTriangles = surfaces[i].numberOfTriangles * sizeof(int3);

        cudaMalloc((void**) &(tmpSurfaces[i].vertices), nBytesForVertices);
        cudaMemcpy(tmpSurfaces[i].vertices, surfaces[i].vertices, nBytesForVertices, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &(tmpSurfaces[i].triangles), nBytesForTriangles);
        cudaMemcpy(tmpSurfaces[i].triangles, surfaces[i].triangles, nBytesForTriangles, cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**) &gpuSurfaces, numSurfaces * sizeof(GPUSurface));    
    cudaMemcpy(gpuSurfaces, tmpSurfaces, numSurfaces * sizeof(GPUSurface), cudaMemcpyHostToDevice);

    delete[] tmpSurfaces;

    return gpuSurfaces;
}

void releaseGPUSurfaces(GPUSurface* gpuSurfaces, int numSurfaces)
{
    if (gpuSurfaces == 0)
    {
        return;
    }
    GPUSurface* tmp = new GPUSurface[numSurfaces];
    cudaMemcpy(tmp, gpuSurfaces, numSurfaces * sizeof(GPUSurface), cudaMemcpyDeviceToHost);     
    cudaFree(gpuSurfaces);

    for (int i = 0; i < numSurfaces; i++)
    {
        cudaFree(tmp[i].vertices);
        cudaFree(tmp[i].triangles);
    }

    delete[] tmp;
}
