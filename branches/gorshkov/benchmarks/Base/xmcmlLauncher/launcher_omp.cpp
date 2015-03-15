#include "launcher_omp.h"

#include <stdlib.h>
#include <memory.h>
#include <omp.h>

#include "../xmcml/mcml_kernel.h"

void InitOutput(InputInfo* input, OutputInfo* output)
{
    output->numberOfPhotons = input->numberOfPhotons;

    int gridSize = input->area->partitionNumber.x * 
        input->area->partitionNumber.y * input->area->partitionNumber.z;
    output->gridSize = gridSize;
    output->commonTrajectory = new uint64[gridSize];
    memset(output->commonTrajectory, 0, gridSize * sizeof(uint64));
    
    output->numberOfDetectors = input->numberOfCubeDetectors;
    output->weightInDetector = new double[input->numberOfCubeDetectors];
    memset(output->weightInDetector, 0, input->numberOfCubeDetectors * sizeof(double));
    
    output->detectorTrajectory = new DetectorTrajectory[input->numberOfCubeDetectors];
    for (int i = 0; i < input->numberOfCubeDetectors; ++i)
    {
        output->detectorTrajectory[i].numberOfPhotons = 0;
        output->detectorTrajectory[i].trajectorySize = gridSize;
        output->detectorTrajectory[i].trajectory = new uint64[gridSize];
        memset(output->detectorTrajectory[i].trajectory, 0, gridSize * sizeof(uint64));
    }
}

void FreeOutput(OutputInfo* output)
{
    if (output != NULL)
    {
        if (output->commonTrajectory != NULL)
        {
            delete[] output->commonTrajectory;
        }
        if (output->weightInDetector != NULL)
        {
            delete[] output->weightInDetector;
        }
        if (output->detectorTrajectory != NULL)
        {
            for (int i = 0; i < output->numberOfDetectors; ++i)
            {
                if (output->detectorTrajectory[i].trajectory != NULL)
                {
                    delete[] output->detectorTrajectory[i].trajectory;
                }
            }
            delete[] output->detectorTrajectory;
        }
    }
}

int GetMaxThreads()
{
    return omp_get_max_threads();
}

void LaunchOMP(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, int numThreads)
{
    omp_set_num_threads(numThreads);

    double specularReflectance = ComputeSpecularReflectance(input->layerInfo);
    OutputInfo* threadOutputs = new OutputInfo[numThreads];

    #pragma omp parallel 
    {
        int threadId = omp_get_thread_num();

        InitOutput(input, &(threadOutputs[threadId]));
        threadOutputs[threadId].specularReflectance = specularReflectance;

        uint64* trajectory = new uint64[threadOutputs[threadId].gridSize];
            
        for (uint64 i = threadId; i < input->numberOfPhotons; i += numThreads)
        {
            ComputePhoton(specularReflectance, input, &(threadOutputs[threadId]), 
                &(randomGenerator[threadId]), trajectory);
        }

        delete[] trajectory;
    }

    output->specularReflectance = specularReflectance;
    for (int i = 0; i < numThreads; ++i)
    {
        for (int j = 0; j < output->gridSize; ++j)
        {
            output->commonTrajectory[j] += threadOutputs[i].commonTrajectory[j];
        }
        for (int k = 0; k < output->numberOfDetectors; ++k)
        {
            output->weightInDetector[k] += threadOutputs[i].weightInDetector[k];
            output->detectorTrajectory[k].numberOfPhotons += 
                threadOutputs[i].detectorTrajectory[k].numberOfPhotons;
            for (int j = 0; j < output->detectorTrajectory[k].trajectorySize; ++j)
            {
                output->detectorTrajectory[k].trajectory[j] += 
                    threadOutputs[i].detectorTrajectory[k].trajectory[j];
            }
        }
    }

    for (int i = 0; i < numThreads; ++i)
    {
        FreeOutput(threadOutputs + i);
    }
    delete[] threadOutputs;
}
