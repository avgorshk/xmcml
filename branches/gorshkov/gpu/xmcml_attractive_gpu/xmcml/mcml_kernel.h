#ifndef __MCML_KERNEL_H
#define __MCML_KERNEL_H

#include "mcml_kernel_types.h"
#include "mcml_mcg59.h"
#include "mcml_intersection.h"
#include "mcml_integration.h"

#include <cuda_runtime_api.h>

//Public
int GetMaxThreads();
double ComputeSpecularReflectance(LayerInfo* layer);
void ComputePhotonBlock(double specularReflectance, InputInfo* gpuInput, MCG59* gpuRandomGenerator, 
    GpuThreadOutput* gpuOutput, int numberOfPhotons, cudaStream_t stream);

//Private
void UpdateWeightInDetector(OutputInfo* output, double photonWeight, int detectorId);
void UpdateDetectorTrajectory(OutputInfo* output, Area* area, PhotonTrajectory* trajectory, int detectorId);
void UpdateTotalTrajectory(OutputInfo* output, Area* area, PhotonTrajectory* trajectory);
void UpdateDetectorTimeScale(OutputInfo* output, double time, double weight, int detectorId);

#endif