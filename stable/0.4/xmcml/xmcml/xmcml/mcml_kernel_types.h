#ifndef __MCML_KERNEL_TYPES_H
#define __MCML_KERNEL_TYPES_H

#include "mcml_types.h"

typedef struct __Detector
{
    double3 center;
    double3 length;
} Detector;

typedef struct __DetectorTrajectory
{
    double* trajectory;
    int trajectorySize;
    int numberOfPhotons;
} DetectorTrajectory;

typedef struct __LayerInfo
{
	double refractiveIndex;
	double absorptionCoefficient;
	double scatteringCoefficient;
	double anisotropy;
    int* surfaceId;
    int numberOfSurfaces;
} LayerInfo;

typedef struct __InputInfo
{
	int numberOfPhotons;
	int numberOfLayers;
	Area* area;
    LayerInfo* layerInfo;
	Surface* surface;
    int numberOfSurfaces;
    Detector* detector;
    int numberOfDetectors;
	double minWeight;
} InputInfo;

typedef struct __OutputInfo
{
	double specularReflectance;
	double* absorption;
    int gridSize;
    double* weightInDetector;
    DetectorTrajectory* detectorTrajectory;
    int numberOfDetectors;
} OutputInfo;

typedef struct __PhotonState
{
	double3 position;
	double3 direction;
	double step;
    int layerId;
	double weight;
	bool isDead;
} PhotonState;

#endif //__MCML_KERNEL_TYPES_H
