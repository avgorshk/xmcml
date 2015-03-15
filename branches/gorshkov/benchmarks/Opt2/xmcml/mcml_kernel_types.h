#ifndef __MCML_KERNEL_TYPES_H
#define __MCML_KERNEL_TYPES_H

#include "mcml_types.h"

#define MAX_TRAJECTORY_SIZE 2048 //Opt1

typedef struct __CubeDetector
{
    double3 center;
    double3 length;
} CubeDetector;

typedef struct __DetectorTrajectory
{
    uint64 numberOfPhotons;
    uint64* trajectory;
    int trajectorySize;
} DetectorTrajectory;

typedef struct __PhotonTrajectory
{
    byte x[MAX_TRAJECTORY_SIZE];
    byte y[MAX_TRAJECTORY_SIZE];
    byte z[MAX_TRAJECTORY_SIZE];
    uint size;
} PhotonTrjectory;

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
    CubeDetector* cubeDetector;
    int numberOfCubeDetectors;
	double minWeight;
	double3 startPosition;
	double3 startDirection;
} InputInfo;

typedef struct __OutputInfo
{
    uint64 numberOfPhotons;
	double specularReflectance;
	uint64* commonTrajectory;
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
