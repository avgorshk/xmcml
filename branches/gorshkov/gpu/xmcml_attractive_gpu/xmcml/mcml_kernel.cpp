#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>

#include "mcml_kernel.h"
#include "mcml_math.h"

double ComputeSpecularReflectance(LayerInfo* layer)
{
	double directReflection1, directReflection2;
	double temp;

	temp = (layer[0].refractiveIndex - layer[1].refractiveIndex) / 
		(layer[0].refractiveIndex + layer[1].refractiveIndex);
	directReflection1 = temp * temp;

	if (layer[1].absorptionCoefficient == 0.0 && layer[1].scatteringCoefficient == 0.0)
	{
		temp = (layer[1].refractiveIndex - layer[2].refractiveIndex) / 
			(layer[1].refractiveIndex + layer[2].refractiveIndex);
		directReflection2 = temp * temp;

		temp = 1 - directReflection1;
		directReflection1 = directReflection1 + temp * temp * directReflection2 / 
			(1 - directReflection1 * directReflection2);
	}

	return directReflection1;
}

void UpdateWeightInDetector(OutputInfo* output, double photonWeight, int detectorId)
{
	output->weightInDetector[detectorId] += photonWeight; 
}

void UpdateDetectorTrajectory(OutputInfo* output, Area* area, PhotonTrajectory* trajectory, int detectorId)
{
    int index;
    uint64* detectorTrajectory = output->detectorTrajectory[detectorId].trajectory;

    for (int i = 0; i < trajectory->position; ++i)
    {
        index = trajectory->x[i] * area->partitionNumber.y * area->partitionNumber.z + 
            trajectory->y[i] * area->partitionNumber.z + trajectory->z[i];
        
        ++(detectorTrajectory[index]);
    }

    ++(output->detectorTrajectory[detectorId].numberOfPhotons);
}

void UpdateTotalTrajectory(OutputInfo* output, Area* area, PhotonTrajectory* trajectory)
{
    int index;
    double* totalTrajectory = output->absorption;

    for (int i = 0; i < trajectory->position; ++i)
    {
        index = trajectory->x[i] * area->partitionNumber.y * area->partitionNumber.z + 
            trajectory->y[i] * area->partitionNumber.z + trajectory->z[i];
        
        ++(totalTrajectory[index]);
    }
}

void UpdateDetectorTimeScale(OutputInfo* output, double time, double weight, int detectorId)
{
    int timeScaleSize = output->detectorTrajectory[detectorId].timeScaleSize;
    TimeInfo* timeScale = output->detectorTrajectory[detectorId].timeScale;
    for (int i = 0; i < timeScaleSize - 1; ++i)
    {
        if ((time >= timeScale[i].timeStart) && (time < timeScale[i].timeFinish))
        {
            ++(timeScale[i].numberOfPhotons);
            timeScale[i].weight += weight;
            return;
        }
    }
    if (time >= timeScale[timeScaleSize - 1].timeStart)
    {
        ++(timeScale[timeScaleSize - 1].numberOfPhotons);
        timeScale[timeScaleSize - 1].weight += weight;
    }
}
