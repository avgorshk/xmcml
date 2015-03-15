#include "launcher_omp.h"

#include <memory.h>
#include <omp.h>

#include "..\..\xmcml\xmcml\mcml_kernel.h"

void InitOutput(InputInfo* input, OutputInfo* output)
{
    int absorptionSize = input->area->partitionNumber.x * 
        input->area->partitionNumber.y * input->area->partitionNumber.z;
    output->absorptionSize = absorptionSize;
    output->absorption = new double[absorptionSize];
    memset(output->absorption, 0, absorptionSize * sizeof(double));
    int numberOfDetectors = input->numberOfDetectors;
    output->numberOfDetectors = numberOfDetectors;
    output->weightInDetector = new double[numberOfDetectors];
    memset(output->weightInDetector, 0, numberOfDetectors * sizeof(double));
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
        int threadId = omp_get_thread_num();
        for (int i = threadId; i < input->numberOfPhotons; i += numThreads)
        {
            ComputePhoton(output->specularReflectance, input, output, &(randomGenerator[threadId]));
        }
    }
}
