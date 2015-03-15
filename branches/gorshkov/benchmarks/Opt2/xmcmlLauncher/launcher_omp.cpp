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

//Opt2
void InitThreadOutput(OutputInfo* dst, OutputInfo* src)
{
    dst->gridSize = src->gridSize;
    dst->detectorTrajectory = src->detectorTrajectory;
    dst->numberOfDetectors = src->numberOfDetectors;
    dst->numberOfPhotons = src->numberOfPhotons;
    dst->specularReflectance = src->specularReflectance;
    dst->weightInDetector = src->weightInDetector;
    
    dst->commonTrajectory = new uint64[dst->gridSize];
    memset(dst->commonTrajectory, 0, dst->gridSize*sizeof(uint64));
}

//Opt2
void FreeThreadOutput(OutputInfo* output)
{
    if (output->commonTrajectory != NULL)
    {
        delete[] output->commonTrajectory;
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

        //Opt2
        InitThreadOutput(&(threadOutputs[threadId]), output);

        //Opt1
        PhotonTrjectory trajectory;
            
        for (uint64 i = threadId; i < input->numberOfPhotons; i += numThreads)
        {
            ComputePhoton(specularReflectance, input, &(threadOutputs[threadId]), 
                &(randomGenerator[threadId]), &trajectory);
        }
    }

    output->specularReflectance = specularReflectance;
    for (int i = 0; i < numThreads; ++i)
    {
        for (int j = 0; j < output->gridSize; ++j)
        {
            output->commonTrajectory[j] += threadOutputs[i].commonTrajectory[j];
        }
    }

    for (int i = 0; i < numThreads; ++i)
    {
        FreeThreadOutput(threadOutputs + i); //Opt2
    }
    delete[] threadOutputs;
}
