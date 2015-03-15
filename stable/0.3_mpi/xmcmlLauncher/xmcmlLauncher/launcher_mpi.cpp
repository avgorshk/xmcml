#include "launcher_mpi.h"

#include <stdlib.h>
#include <memory.h>
#include <mpi.h>

#include "..\..\xmcml\xmcml\mcml_kernel_types.h"
#include "launcher_omp.h"

void SendInt3ToAll(int3* ptr)
{
    MPI_Bcast(&(ptr->x), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(ptr->y), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(ptr->z), 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void SendDouble3ToAll(double3* ptr)
{
    MPI_Bcast(&(ptr->x), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(ptr->y), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(ptr->z), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void SendAreaToAll(Area* area)
{
    SendInt3ToAll(&(area->partitionNumber));
    SendDouble3ToAll(&(area->corner));
    SendDouble3ToAll(&(area->length));
}

void SendLayerInfoToAll(LayerInfo* layerInfo, int numberOfLayers, int pid)
{
    for (int i = 0; i < numberOfLayers; ++i)
    {
        MPI_Bcast(&(layerInfo[i].absorptionCoefficient), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(layerInfo[i].anisotropy), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(layerInfo[i].refractiveIndex), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(layerInfo[i].scatteringCoefficient), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(layerInfo[i].numberOfSurfaces), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (pid > 0)
        {
            layerInfo[i].surfaceId = new int[layerInfo[i].numberOfSurfaces];
        }

        MPI_Bcast(layerInfo[i].surfaceId, layerInfo[i].numberOfSurfaces, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void SendSurfaceToAll(Surface* surface, int numberOfSurfaces, int pid)
{
    for (int i = 0; i < numberOfSurfaces; ++i)
    {
        MPI_Bcast(&(surface[i].numberOfTriangles), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (pid > 0)
        {
            surface[i].triangles = new int3[surface[i].numberOfTriangles];
        }
        
        for (int j = 0; j < surface[i].numberOfTriangles; ++j)
        {
            SendInt3ToAll(&(surface[i].triangles[j]));
        }
        
        MPI_Bcast(&(surface[i].numberOfVertices), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (pid > 0)
        {
            surface[i].vertices = new double3[surface[i].numberOfVertices];
        }
        
        for (int j = 0; j < surface[i].numberOfVertices; ++j)
        {
            SendDouble3ToAll(&(surface[i].vertices[j]));
        }
    }
}

void SendDetectorToAll(Detector* detector, int numberOfDetectors)
{
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        SendDouble3ToAll(&(detector[i].center));
        SendDouble3ToAll(&(detector[i].length));
    }
}

void SendInputToAll(InputInfo* input, int pid)
{
    MPI_Bcast(&(input->minWeight), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfDetectors), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfLayers), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfPhotons), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfSurfaces), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid > 0)
    {
        input->area = new Area;
        input->detector = new Detector[input->numberOfDetectors];
        input->layerInfo = new LayerInfo[input->numberOfLayers];
        input->surface = new Surface[input->numberOfSurfaces];
    }

    SendAreaToAll(input->area);
    SendDetectorToAll(input->detector, input->numberOfDetectors);
    SendLayerInfoToAll(input->layerInfo, input->numberOfLayers, pid);
    SendSurfaceToAll(input->surface, input->numberOfSurfaces, pid);
}

void ReceiveOutputFromAll(OutputInfo* output, int pid)
{
    double* absorption = NULL;
    double* weightInDetector = NULL;

    if (pid == 0)
    {
        absorption = new double[output->absorptionSize];
        weightInDetector = new double[output->numberOfDetectors];
    }

    MPI_Reduce(output->absorption, absorption, output->absorptionSize, 
        MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(output->weightInDetector, weightInDetector, output->numberOfDetectors, 
        MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (pid == 0)
    {
        delete[] output->absorption;
        output->absorption = absorption;
        delete[] output->weightInDetector;
        output->weightInDetector = weightInDetector;
    }
}

void LaunchMPI(InputInfo* input, OutputInfo* output, int numThreadsPerProcess)
{
    if (numThreadsPerProcess <= 0)
    {
        numThreadsPerProcess = GetMaxThreads();
    }

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int numThreads = mpi_size * numThreadsPerProcess;

    MCG59* randomGenerator = new MCG59[numThreadsPerProcess];
    for (int i = 0; i < numThreadsPerProcess; ++i)
    {
        InitMCG59(&(randomGenerator[i]), 777, mpi_rank * numThreadsPerProcess + i, numThreads);
    }

    SendInputToAll(input, mpi_rank);

    int numberOfPhotonsPerProcess = input->numberOfPhotons / mpi_size;
    int remainder = input->numberOfPhotons - numberOfPhotonsPerProcess * mpi_size;
    if (remainder > mpi_rank)
    {
        ++numberOfPhotonsPerProcess;
    }
    
    int specifiedNumberOfPhotons = input->numberOfPhotons;
    input->numberOfPhotons = numberOfPhotonsPerProcess;
    
    LaunchOMP(input, output, randomGenerator, numThreadsPerProcess);

    ReceiveOutputFromAll(output, mpi_rank);

    input->numberOfPhotons = specifiedNumberOfPhotons;
}