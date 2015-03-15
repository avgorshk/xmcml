#include "launcher_omp.h"

#include <memory.h>
#include <omp.h>

#include "..\..\xmcml\xmcml\mcml_kernel.h"

void InitOutput(InputInfo* input, OutputInfo* output)
{
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
        output->detectorTrajectory[i].trajectory = new double[gridSize];
        memset(output->detectorTrajectory[i].trajectory, 0, gridSize * sizeof(double));
    }
}

int GetMaxThreads()
{
    return omp_get_max_threads();
}

void LaunchOMP(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, int numThreads)
{
    omp_set_num_threads(numThreads);

    InitOutput(input, output);
    output->specularReflectance = ComputeSpecularReflectance(input->layerInfo);

    #pragma omp parallel 
    {
        int gridSize = input->area->partitionNumber.x * input->area->partitionNumber.y *
            input->area->partitionNumber.z;
        double* trajectory = new double[gridSize];

        int threadId = omp_get_thread_num();
        for (int i = threadId; i < input->numberOfPhotons; i += numThreads)
        {
            ComputePhoton(output->specularReflectance, input, output, &(randomGenerator[threadId]), 
                trajectory);
        }

        delete[] trajectory;
    }
}
