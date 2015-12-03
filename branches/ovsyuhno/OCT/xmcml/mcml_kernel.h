#ifndef __MCML_KERNEL_H
#define __MCML_KERNEL_H

#include "mcml_kernel_types.h"
#include "mcml_mcg59.h"
#include "mcml_intersection.h"
#include "mcml_integration.h"

//Public
double ComputeSpecularReflectance(LayerInfo* layer);
void ComputePhoton(double specularReflectance, InputInfo* input, OutputInfo* output, 
    MCG59* randomGenerator, PhotonTrajectory* trajectory, int* debind);

//Private
void LaunchPhoton(InputInfo* input, PhotonState* photon, double specularReflectance, MCG59* randomGenerator);
void ComputeStartDirection(InputInfo* input, PhotonState* photon, MCG59* randomGenerator);
void ComputeStartPosition(MCG59* randomGenerator, PhotonState* photon, double standardDeviation);
void MovePhoton(PhotonState* photon);
int GetAreaIndex(double3 photonPosition, Area* area);
void CrossBoundary(PhotonState* photon, InputInfo* input, OutputInfo* output, 
    IntersectionInfo* intersection, MCG59* randomGenerator, PhotonTrajectory* trajectory, int* debind);
void ComputePhotonDirectionWithBiasing(PhotonState* photon, InputInfo* input, MCG59* randomGenerator);
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
void UpdateWeightInDetector(OutputInfo* output, double photonWeight, int detectorId);
void UpdatePhotonTrajectory(PhotonState* photon, InputInfo* input, PhotonTrajectory* trajectory, 
    double3 previousPhotonPosition);
void UpdateDetectorTrajectory(OutputInfo* output, Area* area, PhotonTrajectory* trajectory, int detectorId);
void MovePhotonAndUpdateTrajectory(PhotonState* photon, InputInfo* input, PhotonTrajectory* trajectory);
void MovePhotonOnMinDistanceAndUpdateTrajectory(PhotonState* photon, InputInfo* input, PhotonTrajectory* trajectory);
void UpdateDetectorTimeScale(OutputInfo* output, PhotonState* photon, int detectorId);
int GetDetectorId(PhotonState* photon, InputInfo* input);
int GetWeightTableIndex(InputInfo* input, double anisotropy);
double GetNewPhotonWeight(InputInfo* input, PhotonState* photon, double3 attractiveVector, double p);
double3 GetLocalAttractiveVector(PhotonState* photon, InputInfo* input);
double3 GetNewPhotonDirection(PhotonState* photon, double anisotropy, MCG59* randomGenerator);
void UpdateDetectorRanges(OutputInfo* output, PhotonState* photon, int detectorId);

#endif