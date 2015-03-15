#include "launcher.h"

#include <memory.h>
#include <omp.h>

#include "..\..\xmcml\xmcml\mcml_mcg59.h"
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
    output->weigthInDetector = new double[numberOfDetectors];
    memset(output->weigthInDetector, 0, numberOfDetectors * sizeof(double));
}

void Launch(InputInfo* input, OutputInfo* output, int numThreads)
{
    if (numThreads > 0)
    {
        omp_set_num_threads(numThreads);
    }
    else
    {
        numThreads = omp_get_max_threads();
    }

    MCG59* randomGenerator = new MCG59[numThreads];
    for (int i = 0; i < numThreads; ++i)
    {
        InitMCG59(&(randomGenerator[i]), 777, i, numThreads);
    }

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
