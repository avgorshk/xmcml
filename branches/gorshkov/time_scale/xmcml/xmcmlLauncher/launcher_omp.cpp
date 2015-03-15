#include "launcher_omp.h"

#include <stdlib.h>
#include <memory.h>
#include <omp.h>

#include "..\..\xmcml\xmcml\mcml_kernel.h"

void InitOutput(InputInfo* input, OutputInfo* output)
{
    output->numberOfPhotons = 0;

    int gridSize = input->area->partitionNumber.x * 
        input->area->partitionNumber.y * input->area->partitionNumber.z;
    output->gridSize = gridSize;
    output->absorption = new double[gridSize];
    memset(output->absorption, 0, gridSize * sizeof(double));
    
    int numberOfDetectors = input->numberOfDetectors;
    output->numberOfDetectors = numberOfDetectors;
    output->weightInDetector = new double[numberOfDetectors];
    memset(output->weightInDetector, 0, numberOfDetectors * sizeof(double));
    
    output->detectorTrajectory = new DetectorTrajectory[numberOfDetectors];
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        output->detectorTrajectory[i].numberOfPhotons = 0;
        output->detectorTrajectory[i].trajectorySize = gridSize;
        output->detectorTrajectory[i].trajectory = new uint64[gridSize];
        memset(output->detectorTrajectory[i].trajectory, 0, gridSize * sizeof(uint64));
        output->detectorTrajectory[i].timeScaleSize = input->timeScaleSize;
        output->detectorTrajectory[i].timeScale = new TimeInfo[input->timeScaleSize];

        double timeStep = (input->timeFinish - input->timeStart) / input->timeScaleSize;
        if (timeStep < 0.0) timeStep = 0.0;
        for (int j = 0; j < input->timeScaleSize - 1; ++j)
        {
            output->detectorTrajectory[i].timeScale[j].numberOfPhotons = 0;
            output->detectorTrajectory[i].timeScale[j].timeStart = 
                input->timeStart + j * timeStep;
            output->detectorTrajectory[i].timeScale[j].timeFinish = 
                output->detectorTrajectory[i].timeScale[j].timeStart + timeStep;
            output->detectorTrajectory[i].timeScale[j].trajectorySize = gridSize;
            output->detectorTrajectory[i].timeScale[j].trajectory = new uint64[gridSize];
            memset(output->detectorTrajectory[i].timeScale[j].trajectory, 0, gridSize * sizeof(uint64));
        }
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].numberOfPhotons = 0;
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].timeStart = 
            input->timeStart + (input->timeScaleSize - 1) * timeStep;
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].timeFinish = 
            input->timeFinish;
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].trajectorySize = gridSize;
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].trajectory = new uint64[gridSize];
        memset(output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].trajectory, 
            0, gridSize * sizeof(uint64));
    }
}

void FreeOutput(OutputInfo* output)
{
    if (output != NULL)
    {
        if (output->absorption != NULL)
        {
            delete[] output->absorption;
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
                if (output->detectorTrajectory[i].timeScale != NULL)
                {
                    for (int j = 0; j < output->detectorTrajectory[i].timeScaleSize; ++j)
                    {
                        if (output->detectorTrajectory[i].timeScale[j].trajectory != NULL)
                        {
                            delete[] output->detectorTrajectory[i].timeScale[j].trajectory;
                        }
                    }
                    delete[] output->detectorTrajectory[i].timeScale;
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
            output->absorption[j] += threadOutputs[i].absorption[j];
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
            for (int j = 0; j < output->detectorTrajectory[k].timeScaleSize; ++j)
            {
                output->detectorTrajectory[k].timeScale[j].numberOfPhotons +=
                    threadOutputs[i].detectorTrajectory[k].timeScale[j].numberOfPhotons;
                for (int l = 0; l < output->detectorTrajectory[k].timeScale[j].trajectorySize; ++l)
                {
                    output->detectorTrajectory[k].timeScale[j].trajectory[l] +=
                        threadOutputs[i].detectorTrajectory[k].timeScale[j].trajectory[l];
                }
            }
        }
    }

    for (int i = 0; i < numThreads; ++i)
    {
        FreeOutput(threadOutputs + i);
    }
    delete[] threadOutputs;
}
