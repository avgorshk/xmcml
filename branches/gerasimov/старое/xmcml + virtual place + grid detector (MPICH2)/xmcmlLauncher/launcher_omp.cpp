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
	int gridDetectorSize = input->area->partitionNumber.x *
		input->area->partitionNumber.y;
    output->gridSize = gridSize;
	output->gridDetectorSize = gridDetectorSize;
    output->absorption = new double[gridSize];
	output->scatteringMap = new double[gridSize];
    memset(output->absorption, 0, gridSize * sizeof(double));
	memset(output->scatteringMap, 0, gridSize * sizeof(double));
    
	int numberOfDetectors = input->numberOfCubeDetectors + input->numberOfRingDetectors;
    output->numberOfDetectors = numberOfDetectors;
    output->weightInDetector = new double[numberOfDetectors];
	output->weightInGridDetector = new double[gridDetectorSize];
	memset(output->weightInDetector, 0, numberOfDetectors * sizeof(double));
	memset(output->weightInGridDetector, 0, gridDetectorSize * sizeof(double));

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
            output->detectorTrajectory[i].timeScale[j].weight = 0.0;
            output->detectorTrajectory[i].timeScale[j].timeStart = 
                input->timeStart + j * timeStep;
            output->detectorTrajectory[i].timeScale[j].timeFinish = 
                output->detectorTrajectory[i].timeScale[j].timeStart + timeStep;
        }
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].numberOfPhotons = 0;
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].weight = 0.0;
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].timeStart = 
            input->timeStart + (input->timeScaleSize - 1) * timeStep;
        output->detectorTrajectory[i].timeScale[input->timeScaleSize - 1].timeFinish = 
            input->timeFinish;
		output->detectorTrajectory[i].otherRange = 0.0;
		output->detectorTrajectory[i].targetRange = 0.0;
    }
}

void InitThreadOutput(OutputInfo* dst, OutputInfo* src)
{
    dst->gridSize = src->gridSize;
	dst->gridDetectorSize = src->gridDetectorSize;
    dst->detectorTrajectory = src->detectorTrajectory;
    dst->numberOfDetectors = src->numberOfDetectors;
    dst->numberOfPhotons = src->numberOfPhotons;
    dst->specularReflectance = src->specularReflectance;
    dst->weightInDetector = src->weightInDetector;
	dst->weightInGridDetector = src->weightInGridDetector;
    
    dst->absorption = new double[dst->gridSize];
	dst->scatteringMap = new double[dst->gridSize];
    memset(dst->absorption, 0, dst->gridSize*sizeof(double));
	memset(dst->scatteringMap, 0 ,dst->gridSize*sizeof(double));
}

void FreeOutput(OutputInfo* output)
{
    if (output != NULL)
    {
        if (output->absorption != NULL)
        {
            delete[] output->absorption;
        }
		if (output->scatteringMap != NULL)
		{
			delete[] output->scatteringMap;
		}
        if (output->weightInDetector != NULL)
        {
            delete[] output->weightInDetector;
        }
		if (output->weightInGridDetector != NULL)
		{
			delete[] output->weightInGridDetector;
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
                    delete[] output->detectorTrajectory[i].timeScale;
                }
            }
            delete[] output->detectorTrajectory;
        }
    }
}

void FreeThreadOutput(OutputInfo* output)
{
    if (output->absorption != NULL)
    {
        delete[] output->absorption;
    }
	if (output->scatteringMap != NULL)
	{
		delete[] output->scatteringMap;
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
            &(randomGenerator[threadId]), &(trajectory[threadId]));
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
        for (int j = 0; j < output->gridSize; ++j)
        {
            output->scatteringMap[j] += threadOutputs[i].scatteringMap[j];
        }
    }

    for (int i = 0; i < numThreads; ++i)
    {
        FreeThreadOutput(threadOutputs + i);
    }

    delete[] threadOutputs;
    delete[] trajectory;
}
