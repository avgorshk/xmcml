#include "gpu_utils.h"

#include <cuda_runtime_api.h>

Surface* CopySurfacesToGPU(Surface* surfaces, int numSurfaces)
{
    Surface* gpuSurfaces;    

    Surface* tmpSurfaces = new Surface[numSurfaces];
    for (int i = 0; i < numSurfaces; i++)
    {
        tmpSurfaces[i].numberOfVertices = surfaces[i].numberOfVertices;
        tmpSurfaces[i].numberOfTriangles = surfaces[i].numberOfTriangles;
        int nBytesForVertices = surfaces[i].numberOfVertices*sizeof(Double3);
        int nBytesForTriangles = surfaces[i].numberOfTriangles*sizeof(Int3);

        cudaMalloc((void**) &(tmpSurfaces[i].vertices), nBytesForVertices);
        cudaMemcpy(tmpSurfaces[i].vertices, surfaces[i].vertices, nBytesForVertices, cudaMemcpyHostToDevice);
        cudaMalloc((void**) &(tmpSurfaces[i].triangles), nBytesForTriangles);
        cudaMemcpy(tmpSurfaces[i].triangles, surfaces[i].triangles, nBytesForTriangles, cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**) &gpuSurfaces, numSurfaces * sizeof(Surface));    
    cudaMemcpy(gpuSurfaces, tmpSurfaces, numSurfaces * sizeof(Surface), cudaMemcpyHostToDevice);

    delete[] tmpSurfaces;

    return gpuSurfaces;
}

void ReleaseGpuSurfaces(Surface* gpuSurfaces, int numSurfaces)
{
    Surface* tmp = new Surface[numSurfaces];
    cudaMemcpy(tmp, gpuSurfaces, numSurfaces * sizeof(Surface), cudaMemcpyDeviceToHost);     
    cudaFree(gpuSurfaces);

    for (int i = 0; i < numSurfaces; i++)
    {
        cudaFree(tmp[i].vertices);
        cudaFree(tmp[i].triangles);
    }

    delete[] tmp;
}

BVHTree* CopyBVHToGPU(BVHTree* tree)
{
    BVHTree tmp;
    int nBytesForTriangles = tree->numOfTriangles*sizeof(Triangle);
    int nBytesForNodes = tree->numOfNodes*sizeof(BVHNode);
    Triangle* gpuTriangles;
    BVHNode* gpuNodes;
    
    cudaMalloc((void**)&gpuTriangles, nBytesForTriangles);
    cudaMemcpy(gpuTriangles, tree->triangles, nBytesForTriangles, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpuNodes, nBytesForNodes);
    cudaMemcpy(gpuNodes, tree->nodes, nBytesForNodes, cudaMemcpyHostToDevice);

    tmp.numOfTriangles = tree->numOfTriangles;
    tmp.numOfNodes = tree->numOfNodes;
    tmp.root = tree->root;
    tmp.nodes = gpuNodes;
    tmp.triangles = gpuTriangles;

    BVHTree* gpuTree;
    cudaMalloc((void**)&gpuTree, sizeof(BVHTree));
    cudaMemcpy(gpuTree, &tmp, sizeof(BVHTree), cudaMemcpyHostToDevice);
    
    return gpuTree;
}

void ReleaseGpuBVH(BVHTree* gpuTree)
{
    BVHTree tmp;
    cudaMemcpy(&tmp, gpuTree, sizeof(BVHTree), cudaMemcpyDeviceToHost);
	cudaFree(tmp.triangles);
    cudaFree(tmp.nodes);
    cudaFree(gpuTree);
}

Area* CopyAreaToGPU(Area* area)
{
    Area* gpuArea;
    cudaMalloc((void**)&gpuArea, sizeof(Area));
    cudaMemcpy(gpuArea, area, sizeof(Area), cudaMemcpyHostToDevice);
    return gpuArea;
}

void ReleaseGpuArea(Area* gpuArea)
{
    cudaFree(gpuArea);
}

CubeDetector* CopyCubeDetectorsToGPU(CubeDetector* cubeDetector, int numberOfCubeDetectors)
{
    CubeDetector* gpuCubeDetector;
    cudaMalloc((void**)&gpuCubeDetector, numberOfCubeDetectors*sizeof(CubeDetector));
    cudaMemcpy(gpuCubeDetector, cubeDetector, numberOfCubeDetectors*sizeof(CubeDetector), cudaMemcpyHostToDevice);
    return gpuCubeDetector;
}

void ReleaseGpuCubeDetectors(CubeDetector* gpuCubeDetector)
{
    cudaFree(gpuCubeDetector);
}

RingDetector* CopyRingDetectorsToGPU(CubeDetector* ringDetector, int numberOfRingDetectors)
{
    RingDetector* gpuRingDetector;
    cudaMalloc((void**)&gpuRingDetector, numberOfRingDetectors*sizeof(RingDetector));
    cudaMemcpy(gpuRingDetector, ringDetector, numberOfRingDetectors*sizeof(RingDetector), cudaMemcpyHostToDevice);
    return gpuRingDetector;
}

void ReleaseGpuRingDetectors(RingDetector* gpuRingDetector)
{
    cudaFree(gpuRingDetector);
}

LayerInfo* CopyLayerInfoToGPU(LayerInfo* layerInfo, int numberOfLayers)
{
    LayerInfo* tmp = new LayerInfo[numberOfLayers];
    for (int i = 0; i < numberOfLayers; ++i)
    {
        tmp[i].absorptionCoefficient = layerInfo[i].absorptionCoefficient;
        tmp[i].anisotropy = layerInfo[i].anisotropy;
        tmp[i].numberOfSurfaces = layerInfo[i].numberOfSurfaces;
        tmp[i].refractiveIndex = layerInfo[i].refractiveIndex;
        tmp[i].scatteringCoefficient = layerInfo[i].scatteringCoefficient;
        
        int* gpuSurfaceId;
        cudaMalloc((void**)&gpuSurfaceId, layerInfo[i].numberOfSurfaces*sizeof(int));
        cudaMemcpy(gpuSurfaceId, layerInfo[i].surfaceId, layerInfo[i].numberOfSurfaces*sizeof(int), cudaMemcpyHostToDevice);
        tmp[i].surfaceId = gpuSurfaceId;
    }

    LayerInfo* gpuLayerInfo;
    cudaMalloc((void**)&gpuLayerInfo, numberOfLayers*sizeof(LayerInfo));
    cudaMemcpy(gpuLayerInfo, tmp, numberOfLayers*sizeof(LayerInfo), cudaMemcpyHostToDevice);

    delete[] tmp;
    return gpuLayerInfo;
}

void ReleaseGpuLayerInfo(LayerInfo* gpuLayerInfo, int numberOfLayers)
{
    LayerInfo* tmp = new LayerInfo[numberOfLayers];
    cudaMemcpy(tmp, gpuLayerInfo, numberOfLayers*sizeof(LayerInfo), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numberOfLayers; ++i)
    {
        cudaFree(tmp[i].surfaceId);
    }

    cudaFree(gpuLayerInfo);
    delete[] tmp;
}

WeightIntegralTable* CopyWeightIntegralTableToGPU(WeightIntegralTable* weightIntegral, int numberOfTables)
{
    WeightIntegralTable* tmp = new WeightIntegralTable[numberOfTables];
    for (int i = 0; i < numberOfTables; ++i)
    {
        tmp[i].anisotropy = weightIntegral[i].anisotropy;
        tmp[i].numberOfElements = weightIntegral[i].numberOfElements;

        double* elements;
        cudaMalloc((void**)&elements, weightIntegral[i].numberOfElements*sizeof(double));
        cudaMemcpy(elements, weightIntegral[i].elements, weightIntegral[i].numberOfElements*sizeof(double), 
            cudaMemcpyHostToDevice);

        tmp[i].elements = elements;
    }

    WeightIntegralTable* gpuWeightIntegral;
    cudaMalloc((void**)&gpuWeightIntegral, numberOfTables*sizeof(WeightIntegralTable));
    cudaMemcpy(gpuWeightIntegral, tmp, numberOfTables*sizeof(WeightIntegralTable), cudaMemcpyHostToDevice);

    delete[] tmp;
    return gpuWeightIntegral;
}

void ReleaseGpuWeightIntegralTable(WeightIntegralTable* gpuWeightIntegral, int numberOfTables)
{
    WeightIntegralTable* tmp = new WeightIntegralTable[numberOfTables];
    cudaMemcpy(tmp, gpuWeightIntegral, numberOfTables*sizeof(WeightIntegralTable), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numberOfTables; ++i) 
        cudaFree(tmp[i].elements);
    cudaFree(gpuWeightIntegral);
    delete[] tmp;
}

InputInfo* CopyInputToGPU(InputInfo* input)
{
    InputInfo tmp;
    tmp.area = CopyAreaToGPU(input->area);
    tmp.attractiveFactor = input->attractiveFactor;
    tmp.bvhTree = CopyBVHToGPU(input->bvhTree);
    tmp.cubeDetector = CopyCubeDetectorsToGPU(input->cubeDetector, input->numberOfCubeDetectors);
    tmp.layerInfo = CopyLayerInfoToGPU(input->layerInfo, input->numberOfLayers);
    tmp.minWeight = input->minWeight;
    tmp.numberOfCubeDetectors = input->numberOfCubeDetectors;
    tmp.numberOfLayers = input->numberOfLayers;
    tmp.numberOfPhotons = input->numberOfPhotons;
    tmp.numberOfRingDetectors = input->numberOfRingDetectors;
    tmp.numberOfSurfaces = input->numberOfSurfaces;
    tmp.numberOfWeightTables = input->numberOfWeightTables;
    tmp.ringDetector = CopyRingDetectorsToGPU(input->cubeDetector, input->numberOfRingDetectors);
    tmp.startPosition = input->startPosition;
    tmp.startDirection = input->startDirection;
    tmp.surface = CopySurfacesToGPU(input->surface, input->numberOfSurfaces);
    tmp.targetPoint = input->targetPoint;
    tmp.timeFinish = input->timeFinish;
    tmp.timeScaleSize = input->timeScaleSize;
    tmp.timeStart = input->timeStart;
    tmp.weightIntegralPrecision = input->weightIntegralPrecision;
    tmp.weightTable = CopyWeightIntegralTableToGPU(input->weightTable, input->numberOfWeightTables);
    tmp.weightTablePrecision = input->weightTablePrecision;

    InputInfo* gpuInput;
    cudaMalloc((void**)&gpuInput, sizeof(InputInfo));
    cudaMemcpy(gpuInput, &tmp, sizeof(InputInfo), cudaMemcpyHostToDevice);
    
    return gpuInput;
}

void ReleaseGpuInput(InputInfo* gpuInput)
{
    InputInfo tmp;
    cudaMemcpy(&tmp, gpuInput, sizeof(InputInfo), cudaMemcpyDeviceToHost);
    
    ReleaseGpuArea(tmp.area);
    ReleaseGpuBVH(tmp.bvhTree);
    ReleaseGpuCubeDetectors(tmp.cubeDetector);
    ReleaseGpuLayerInfo(tmp.layerInfo, tmp.numberOfLayers);
    ReleaseGpuRingDetectors(tmp.ringDetector);
    ReleaseGpuSurfaces(tmp.surface, tmp.numberOfSurfaces);
    ReleaseGpuWeightIntegralTable(tmp.weightTable, tmp.numberOfWeightTables);

    cudaFree(gpuInput);
}

MCG59* CopyRandomGeneratorsToGPU(MCG59* randomGenerators, int numberOfThreads)
{
    MCG59* gpuRandomGenerators;
    cudaMalloc((void**)&gpuRandomGenerators, numberOfThreads*sizeof(MCG59));
    cudaMemcpy(gpuRandomGenerators, randomGenerators, numberOfThreads*sizeof(MCG59), cudaMemcpyHostToDevice);
    return gpuRandomGenerators;
}

void ReleaseGpuRandomGenerators(MCG59* gpuRandomGenerators)
{
    cudaFree(gpuRandomGenerators);
}

void SetGpuDevice(int mpi_rank)
{
    int numberOfGpu;
    cudaGetDeviceCount(&numberOfGpu);
    cudaSetDevice(mpi_rank%numberOfGpu);
}