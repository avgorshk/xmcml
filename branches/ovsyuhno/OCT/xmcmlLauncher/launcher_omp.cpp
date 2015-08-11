#include "launcher_omp.h"

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <omp.h>

#include "../xmcml/mcml_kernel.h"

void InitOutput(InputInfo* input, OutputInfo* output)
{
    output->numberOfPhotons = 0;

    int gridSize = input->area->partitionNumber.x * 
        input->area->partitionNumber.y * input->area->partitionNumber.z;
    output->gridSize = gridSize;
    output->absorption = new double[gridSize];
    memset(output->absorption, 0, gridSize * sizeof(double));
    
	int numberOfDetectors = input->numberOfCubeDetectors + input->numberOfRingDetectors;
    output->numberOfDetectors = numberOfDetectors;
    
    output->detectorInfo = new DetectorInfo[numberOfDetectors];
    for (int i = 0; i < numberOfDetectors; ++i)
    {
		output->detectorInfo[i].weight = 0;
        output->detectorInfo[i].numberOfPhotons = 0;
        output->detectorInfo[i].trajectorySize = gridSize;
        output->detectorInfo[i].trajectory = new uint64[gridSize];
        memset(output->detectorInfo[i].trajectory, 0, gridSize * sizeof(uint64));
        output->detectorInfo[i].timeScaleSize = input->timeScaleSize;
        output->detectorInfo[i].timeScale = new TimeInfo[input->timeScaleSize];

        double timeStep = (input->timeFinish - input->timeStart) / input->timeScaleSize;
        if (timeStep < 0.0) timeStep = 0.0;
        for (int j = 0; j < input->timeScaleSize - 1; ++j)
        {
            output->detectorInfo[i].timeScale[j].numberOfPhotons = 0;
            output->detectorInfo[i].timeScale[j].weight = 0.0;
            output->detectorInfo[i].timeScale[j].timeStart = 
                input->timeStart + j * timeStep;
            output->detectorInfo[i].timeScale[j].timeFinish = 
                output->detectorInfo[i].timeScale[j].timeStart + timeStep;
        }
        output->detectorInfo[i].timeScale[input->timeScaleSize - 1].numberOfPhotons = 0;
        output->detectorInfo[i].timeScale[input->timeScaleSize - 1].weight = 0.0;
        output->detectorInfo[i].timeScale[input->timeScaleSize - 1].timeStart = 
            input->timeStart + (input->timeScaleSize - 1) * timeStep;
        output->detectorInfo[i].timeScale[input->timeScaleSize - 1].timeFinish = 
            input->timeFinish;
		output->detectorInfo[i].targetRange = 0.0;
    }
}

void InitThreadOutput(OutputInfo* dst, OutputInfo* src)
{
    dst->gridSize = src->gridSize;
    dst->detectorInfo = src->detectorInfo;
    dst->numberOfDetectors = src->numberOfDetectors;
    dst->numberOfPhotons = src->numberOfPhotons;
    dst->specularReflectance = src->specularReflectance;
	//dst->detectorInfo->weight = src->detectorInfo->weight;
    
    dst->absorption = new double[dst->gridSize];
    memset(dst->absorption, 0, dst->gridSize*sizeof(double));
}

void FreeOutput(OutputInfo* output)
{
    if (output != NULL)
    {
        if (output->absorption != NULL)
        {
            delete[] output->absorption;
        }
        if (output->detectorInfo != NULL)
        {
            for (int i = 0; i < output->numberOfDetectors; ++i)
            {
                if (output->detectorInfo[i].trajectory != NULL)
                {
                    delete[] output->detectorInfo[i].trajectory;
                }
                if (output->detectorInfo[i].timeScale != NULL)
                {
                    delete[] output->detectorInfo[i].timeScale;
                }
            }
            delete[] output->detectorInfo;
        }
    }
}

void FreeThreadOutput(OutputInfo* output)
{
    if (output->absorption != NULL)
    {
        delete[] output->absorption;
    }
}

int GetMaxThreads()
{
    return omp_get_max_threads();
}

void LaunchOMP(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, int numThreads)
{
	int debind = 0;
    omp_set_num_threads(numThreads);

    double specularReflectance = ComputeSpecularReflectance(input->layerInfo);
    
    OutputInfo* threadOutputs = new OutputInfo[numThreads];
    for (int i = 0; i < numThreads; ++i)
    {
        InitThreadOutput(&(threadOutputs[i]), output);
        threadOutputs[i].specularReflectance = specularReflectance;
    }

    PhotonTrajectory* trajectory = new PhotonTrajectory[numThreads];

    #pragma omp parallel for schedule(dynamic)
    for (long long int i = 0; i < input->numberOfPhotons; ++i)
    {
        int threadId = omp_get_thread_num();
        ComputePhoton(specularReflectance, input, &(threadOutputs[threadId]), 
            &(randomGenerator[threadId]), &(trajectory[threadId]), &debind);
    }

    output->specularReflectance = specularReflectance;
    for (int i = 0; i < numThreads; ++i)
    {
        for (int j = 0; j < output->gridSize; ++j)
        {
            output->absorption[j] += threadOutputs[i].absorption[j];
        }
    }

    for (int i = 0; i < numThreads; ++i)
    {
        FreeThreadOutput(threadOutputs + i);
    }

    delete[] threadOutputs;
    delete[] trajectory;
}
