#ifndef __MCML_KERNEL_H
#define __MCML_KERNEL_H

#include "mcml_kernel_types.h"
#include "mcml_mcg59.h"
#include "mcml_intersection.h"

//Public
double ComputeSpecularReflectance(LayerInfo* layer);
void ComputePhoton(double specularReflectance, InputInfo* input, OutputInfo* output, 
    MCG59* randomGenerator, uint64* trajectory);

//Private
void LaunchPhoton(PhotonState* photon, double specularReflectance);
void MovePhoton(PhotonState* photon);
int GetAreaIndex(double3 photonPosition, Area* area);
void CrossBoundary(PhotonState* photon, InputInfo* input, OutputInfo* output, 
    IntersectionInfo* intersection, MCG59* randomGenerator, uint64* trajectory);
void ComputePhotonDirection(PhotonState* photon, double anisotropy, MCG59* randomGenerator);
double ComputeCosineTheta(double anisotropy, MCG59* randomGenerator);
double ComputeTransmitCosine(double incidentRefractiveIndex, double transmitRefractiveIndex, 
	double incidentCos);
double ComputeFrenselReflectance(double incidentRefractiveIndex, double transmitRefractiveIndex,
	double incidentCos, double transmitCos);
int FindIntersectionLayer(InputInfo* input, int surfaceId, int currectLayerId);
double3 ReflectVector(double3 incidentVector, double3 normalVector);
double3 RefractVector(double incidentRefractiveIndex, double transmitRefractiveIndex, 
    double3 incidentVector, double3 normalVector);
void ComputeStepSizeInTissue(PhotonState* photon, InputInfo* info, MCG59* randomGenerator);
void ComputeDroppedWeightOfPhoton(PhotonState* photon, InputInfo* input, OutputInfo* output, int droppedIndex);
double GetCriticalCos(LayerInfo* layer, int currectLayerId, int intersectionLayerId);
int UpdateWeightInDetectors(PhotonState* photon, InputInfo* input, OutputInfo* output);
void UpdatePhotonTrajectory(PhotonState* photon, InputInfo* input, uint64* trajectory, 
    double3 previousPhotonPosition);
void UpdateDetectorTrajectory(OutputInfo* output, uint64* trajectory, int detectorId);
void MovePhotonAndUpdateTrajectory(PhotonState* photon, InputInfo* input, uint64* trajectory);
void MovePhotonOnMinDistanceAndUpdateTrajectory(PhotonState* photon, InputInfo* input, uint64* trajectory);
void UpdateDetectorTimeScale(OutputInfo* output, PhotonState* photon, uint64* trajectory, int detectorId);

#endif