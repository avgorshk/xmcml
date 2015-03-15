#ifndef __MCML_KERNEL_TYPES_H
#define __MCML_KERNEL_TYPES_H

#include "mcml_types.h"
#include "mcml_intersection.h"

typedef struct __Detector
{
    double3 center;
    double3 length;
} Detector;

typedef struct __DetectorTrajectory
{
    uint64 numberOfPhotons;
    uint64* trajectory;
    int trajectorySize;
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
	uint64 numberOfPhotons;
	Area* area;
    LayerInfo* layerInfo;
    int numberOfLayers;
	Surface* surface;
    int numberOfSurfaces;
    Detector* detector;
    int numberOfDetectors;
	double minWeight;
    BVHTree* bvhTree;
} InputInfo;

typedef struct __OutputInfo
{
    uint64 numberOfPhotons;
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