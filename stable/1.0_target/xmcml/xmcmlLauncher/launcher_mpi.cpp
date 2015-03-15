#include "launcher_mpi.h"

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#include "../xmcml/mcml_kernel_types.h"
#include "../xmcml/mcml_intersection.h"
#include "logger.h"
#include "writer.h"
#include "launcher_omp.h"

void mpi_sum(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype)
{
    uint64* in = (uint64*)invec;
    uint64* inout = (uint64*)inoutvec;
    for (int i = 0; i < *len; ++i)
    {
        inout[i] += in[i];
    }
}

MPI_Datatype CreateMCG59TypeForMPI()
{
    int blocklens[3] = {1, 1, 1};
    MPI_Aint indices[3] = {0, sizeof(uint64), sizeof(MCG59)};
    MPI_Datatype old_types[3] = {MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG, MPI_UB};
    MPI_Datatype MPI_MCG59_Type;
    MPI_Type_create_struct(3, blocklens, indices, old_types, &MPI_MCG59_Type);
    MPI_Type_commit(&MPI_MCG59_Type);
    return MPI_MCG59_Type;
}

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

void SendCubeDetectorToAll(CubeDetector* detector, int numberOfDetectors)
{
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        SendDouble3ToAll(&(detector[i].center));
        SendDouble3ToAll(&(detector[i].length));
		MPI_Bcast(&(detector[i].targetLayer), 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void SendRingDetectorToAll(RingDetector* detector, int numberOfDetectors)
{
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        SendDouble3ToAll(&(detector[i].center));
		MPI_Bcast(&(detector[i].smallRadius), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(detector[i].bigRadius), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(detector[i].targetLayer), 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void SendInputToAll(InputInfo* input, int pid)
{
    MPI_Bcast(&(input->minWeight), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfCubeDetectors), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&(input->numberOfRingDetectors), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfLayers), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfPhotons), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->numberOfSurfaces), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->timeStart), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->timeFinish), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(input->timeScaleSize), 1, MPI_INT, 0, MPI_COMM_WORLD);
	SendDouble3ToAll(&(input->startPosition));
	SendDouble3ToAll(&(input->startDirection));

    if (pid > 0)
    {
        input->area = new Area;
        input->cubeDetector = new CubeDetector[input->numberOfCubeDetectors];
		input->ringDetector = new RingDetector[input->numberOfRingDetectors];
        input->layerInfo = new LayerInfo[input->numberOfLayers];
        input->surface = new Surface[input->numberOfSurfaces];
    }

    SendAreaToAll(input->area);
    SendCubeDetectorToAll(input->cubeDetector, input->numberOfCubeDetectors);
	SendRingDetectorToAll(input->ringDetector, input->numberOfRingDetectors);
    SendLayerInfoToAll(input->layerInfo, input->numberOfLayers, pid);
    SendSurfaceToAll(input->surface, input->numberOfSurfaces, pid);
}

void SendRandomGeneratorToAll(MCG59** randomGenerator, int numThreadsPerProcess, int pid)
{
    MCG59* buffer = new MCG59[numThreadsPerProcess];

    MPI_Datatype MPI_MCG59_Type = CreateMCG59TypeForMPI();

    MPI_Scatter(*randomGenerator, numThreadsPerProcess, MPI_MCG59_Type, buffer, 
        numThreadsPerProcess, MPI_MCG59_Type, 0, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_MCG59_Type);
    
    if (pid == 0)
    {
        delete[] (*randomGenerator);
    }
    *randomGenerator = buffer;
}

void ReceiveOutputFromAll(OutputInfo* output, int pid)
{
    double* absorption = NULL;
    double* weightInDetector = NULL;
    uint64* trajectory = NULL;
    uint64 numberOfPhotonsPerDetector = 0;
    uint64 numberOfPhotonsPerTime = 0;
    double weightPerTime = 0.0;

    MPI_Op ullSumOp;
    MPI_Op_create((MPI_User_function*)mpi_sum, 1, &ullSumOp);

    if (pid == 0)
    {
        absorption = new double[output->gridSize];
        weightInDetector = new double[output->numberOfDetectors];
    }

    MPI_Reduce(output->absorption, absorption, output->gridSize, 
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

    for (int i = 0; i < output->numberOfDetectors; ++i)
    {
        if (pid == 0)
        {
            trajectory = new uint64[output->gridSize];
            numberOfPhotonsPerDetector = 0;
        }
        
        MPI_Reduce(output->detectorTrajectory[i].trajectory, trajectory, output->gridSize, 
            MPI_UNSIGNED_LONG_LONG, ullSumOp, 0, MPI_COMM_WORLD);
        MPI_Reduce(&(output->detectorTrajectory[i].numberOfPhotons), &numberOfPhotonsPerDetector, 1,
            MPI_UNSIGNED_LONG_LONG, ullSumOp, 0, MPI_COMM_WORLD);

        if (pid == 0)
        {
            delete[] output->detectorTrajectory[i].trajectory;
            output->detectorTrajectory[i].trajectory = trajectory;
            output->detectorTrajectory[i].numberOfPhotons = numberOfPhotonsPerDetector;
        }

        for (int j = 0; j < output->detectorTrajectory[i].timeScaleSize; ++j)
        {
            if (pid == 0)
            {
                numberOfPhotonsPerTime = 0;
                weightPerTime = 0.0;
            }

            MPI_Reduce(&(output->detectorTrajectory[i].timeScale[j].numberOfPhotons), 
                &numberOfPhotonsPerTime, 1, MPI_UNSIGNED_LONG_LONG, ullSumOp, 0, MPI_COMM_WORLD);
            MPI_Reduce(&(output->detectorTrajectory[i].timeScale[j].weight), 
                &weightPerTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (pid == 0)
            {
                output->detectorTrajectory[i].timeScale[j].numberOfPhotons = numberOfPhotonsPerTime;
                output->detectorTrajectory[i].timeScale[j].weight = weightPerTime;
            }
        }
    }

    MPI_Op_free(&ullSumOp);
}

MCG59* ReceiveRandomGeneratorFromAll(MCG59* randomGenerator, int numThreadsPerProcess)
{
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    MCG59* result = NULL;
    if (mpi_rank == 0)
    {
        result = new MCG59[mpi_size * numThreadsPerProcess];
    }

    MPI_Datatype MPI_MCG59_Type = CreateMCG59TypeForMPI();

    MPI_Gather(randomGenerator, numThreadsPerProcess, MPI_MCG59_Type, result, numThreadsPerProcess, 
        MPI_MCG59_Type, 0, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_MCG59_Type);

    return result;
}

void DoBackup(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, int numThreads, uint64 numberOfPhotonsCount)
{
    bool isOk;
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    ReceiveOutputFromAll(output, mpi_rank);
    MCG59* commonRandomGenerator = ReceiveRandomGeneratorFromAll(randomGenerator, numThreads);

    if (mpi_rank == 0)
    {
        output->numberOfPhotons += mpi_size * numberOfPhotonsCount;
        input->numberOfPhotons = output->numberOfPhotons;
        
        isOk = WriteBackupToFile(input, output, commonRandomGenerator, numThreads, mpi_size);

        char message[128];
        sprintf(message, "%llu photons done", output->numberOfPhotons);
        WriteMessageToLog(message);
        sprintf(message, "%s", isOk ? "Backup file is written" : "Backup file is NOT written");
        WriteMessageToLog(message);
    }
    else
    {
        output->numberOfPhotons = 0;
        output->specularReflectance = 0.0;
        memset(output->absorption, 0, output->gridSize * sizeof(double));
        memset(output->weightInDetector, 0, output->numberOfDetectors * sizeof(double));
        for (int i = 0; i < output->numberOfDetectors; ++i)
        {
            output->detectorTrajectory[i].numberOfPhotons = 0;
            memset(output->detectorTrajectory[i].trajectory, 0, 
                output->detectorTrajectory[i].trajectorySize * sizeof(uint64));
            for (int j = 0; j < output->detectorTrajectory[i].timeScaleSize; ++j)
            {
                output->detectorTrajectory[i].timeScale[j].numberOfPhotons = 0;
                output->detectorTrajectory[i].timeScale[j].weight = 0.0;
            }
        }
    }
}

void LaunchOMPWithBackups(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, 
    int numThreads, uint backupPortionSize)
{
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    uint64 numberOfPhotonsPerProcess = input->numberOfPhotons;
    uint64 numberOfBackupPortions = (numberOfPhotonsPerProcess - 1) / backupPortionSize;
    for (uint64 i = 0; i < numberOfBackupPortions; ++i)
    {
        input->numberOfPhotons = backupPortionSize;
        LaunchOMP(input, output, randomGenerator, numThreads);
        DoBackup(input, output, randomGenerator, numThreads, backupPortionSize);
    }
    
    input->numberOfPhotons = numberOfPhotonsPerProcess - numberOfBackupPortions * backupPortionSize;
    LaunchOMP(input, output, randomGenerator, numThreads);

    if (mpi_rank == 0)
    {
        output->numberOfPhotons += mpi_size * input->numberOfPhotons;
    }
}

void LaunchMPI(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, 
    int numThreadsPerProcess, uint backupPortionSize)
{
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    SendInputToAll(input, mpi_rank);
    if (mpi_rank > 0)
    {
        InitOutput(input, output);
    }

    SendRandomGeneratorToAll(&randomGenerator, numThreadsPerProcess, mpi_rank);

    input->bvhTree = GenerateBVHTree(input->surface, input->numberOfSurfaces);

    uint64 numberOfPhotonsPerProcess = input->numberOfPhotons / mpi_size;
    uint64 remainder = input->numberOfPhotons - numberOfPhotonsPerProcess * mpi_size;
    if (remainder > mpi_rank)
    {
        ++numberOfPhotonsPerProcess;
    }

    uint64 specifiedNumberOfPhotons = input->numberOfPhotons;
    input->numberOfPhotons = numberOfPhotonsPerProcess;

    LaunchOMPWithBackups(input, output, randomGenerator, numThreadsPerProcess, backupPortionSize);

    ReceiveOutputFromAll(output, mpi_rank);

    input->numberOfPhotons = specifiedNumberOfPhotons;

    delete[] randomGenerator;
}