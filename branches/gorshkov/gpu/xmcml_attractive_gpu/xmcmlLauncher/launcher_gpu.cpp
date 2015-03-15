#include "launcher_gpu.h"

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <omp.h>
#include <cuda_runtime_api.h>

#include "../xmcml/mcml_kernel.h"
#include "gpu_utils.h"

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
                    delete[] output->detectorTrajectory[i].timeScale;
                }
            }
            delete[] output->detectorTrajectory;
        }
    }
}

void UpdateOutput(InputInfo* input, OutputInfo* output, GpuThreadOutput* cpuOutput, int numberOfPhotons)
{
    for (int i = 0; i < numberOfPhotons; ++i)
    {
        UpdateTotalTrajectory(output, input->area, &(cpuOutput[i].trajectory));
        if (cpuOutput[i].detectorId >= 0)
        {
            UpdateDetectorTrajectory(output, input->area, &(cpuOutput[i].trajectory), cpuOutput[i].detectorId);
            UpdateWeightInDetector(output, cpuOutput[i].weight, cpuOutput[i].detectorId);
            UpdateDetectorTimeScale(output, cpuOutput[i].time, cpuOutput[i].weight, cpuOutput[i].detectorId);
        }
    }
}

void LaunchGPU(InputInfo* input, OutputInfo* output, MCG59* randomGenerator)
{
    int numberOfThreads = GetMaxThreads();
    InputInfo* gpuInput = CopyInputToGPU(input);
    MCG59* gpuRandomGenerator = CopyRandomGeneratorsToGPU(randomGenerator, numberOfThreads);
    
    GpuThreadOutput *gpuOutput1, *gpuOutput2;
    cudaMalloc((void**)&gpuOutput1, numberOfThreads*sizeof(GpuThreadOutput));
    cudaMalloc((void**)&gpuOutput2, numberOfThreads*sizeof(GpuThreadOutput));
    GpuThreadOutput* gpuOutput[] = {gpuOutput1, gpuOutput2};

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStream_t stream[] = {stream1, stream2};

    GpuThreadOutput* cpuOutput;
    cudaHostAlloc((void**)&cpuOutput, numberOfThreads*sizeof(GpuThreadOutput), cudaHostAllocDefault);

    double specularReflectance = ComputeSpecularReflectance(input->layerInfo);
    output->specularReflectance = specularReflectance;

    uint64 numberOfPhotonBlocks = input->numberOfPhotons/numberOfThreads;

    if (input->numberOfPhotons <= numberOfThreads)
    {
        ComputePhotonBlock(specularReflectance, gpuInput, gpuRandomGenerator, gpuOutput1, input->numberOfPhotons, stream1);
        cudaMemcpy(cpuOutput, gpuOutput1, numberOfThreads*sizeof(GpuThreadOutput), cudaMemcpyDeviceToHost);
        UpdateOutput(input, output, cpuOutput, input->numberOfPhotons);
    }
    else
    {
        byte index = 0;
        ComputePhotonBlock(specularReflectance, gpuInput, gpuRandomGenerator, gpuOutput[index], numberOfThreads, stream[index]);

        for (uint64 i = 1; i < numberOfPhotonBlocks; ++i)
        {
            ComputePhotonBlock(specularReflectance, gpuInput, gpuRandomGenerator, gpuOutput[(~index)&1], 
                numberOfThreads, stream[(~index)&1]);
            cudaMemcpyAsync(cpuOutput, gpuOutput[index], numberOfThreads*sizeof(GpuThreadOutput), 
                cudaMemcpyDeviceToHost, stream[index]);
            cudaStreamSynchronize(stream[index]);
            UpdateOutput(input, output, cpuOutput, numberOfThreads);
            index = (~index) & 1;
        }

        if (numberOfPhotonBlocks*numberOfThreads < input->numberOfPhotons)
        {
            ComputePhotonBlock(specularReflectance, gpuInput, gpuRandomGenerator, gpuOutput[(~index)&1], 
                input->numberOfPhotons - numberOfPhotonBlocks*numberOfThreads, stream[(~index)&1]);
            cudaMemcpyAsync(cpuOutput, gpuOutput[index], numberOfThreads*sizeof(GpuThreadOutput), 
                cudaMemcpyDeviceToHost, stream[index]);
            cudaStreamSynchronize(stream[index]);
            UpdateOutput(input, output, cpuOutput, numberOfThreads);
            index = (~index) & 1;

            cudaMemcpyAsync(cpuOutput, gpuOutput[index], 
                (input->numberOfPhotons - numberOfPhotonBlocks*numberOfThreads)*sizeof(GpuThreadOutput), 
                cudaMemcpyDeviceToHost, stream[index]);
            cudaStreamSynchronize(stream[index]);
            UpdateOutput(input, output, cpuOutput, input->numberOfPhotons - numberOfPhotonBlocks*numberOfThreads);
        }
        else
        {
            cudaMemcpyAsync(cpuOutput, gpuOutput[index], numberOfThreads*sizeof(GpuThreadOutput), 
                cudaMemcpyDeviceToHost, stream[index]);
            cudaStreamSynchronize(stream[index]);
            UpdateOutput(input, output, cpuOutput, numberOfThreads);
        }
    }

    ReleaseGpuInput(gpuInput);
    ReleaseGpuRandomGenerators(gpuRandomGenerator);
    cudaFree(gpuOutput1);
    cudaFree(gpuOutput2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(cpuOutput);
}
