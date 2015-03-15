#ifndef __MCML_KERNEL_TYPES_H
#define __MCML_KERNEL_TYPES_H

#include "mcml_types.h"
#include "mcml_intersection.h"

#define MAX_LAYERS 8

typedef struct __TimeInfo
{
    double timeStart;
    double timeFinish;
    uint64 numberOfPhotons;
    double weight;
} TimeInfo;

typedef struct __CubeDetector
{
    double3 center;
    double3 length;
	int targetLayer;
} CubeDetector;

typedef struct __RingDetector
{
	double3 center;
	double smallRadius;
	double bigRadius;
	int targetLayer;
} RingDetector;

typedef struct __DetectorTrajectory
{
    uint64 numberOfPhotons;
    uint64* trajectory;
    int trajectorySize;
    TimeInfo* timeScale;
    int timeScaleSize;
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
    CubeDetector* cubeDetector;
    int numberOfCubeDetectors;
	RingDetector* ringDetector;
	int numberOfRingDetectors;
	double minWeight;
    BVHTree* bvhTree;
    double timeStart;
    double timeFinish;
    int timeScaleSize;
	double3 startPosition;
	double3 startDirection;
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
	bool visitedLayers[MAX_LAYERS];
	double weight;
    double time;
	bool isDead;
} PhotonState;

#endif //__MCML_KERNEL_TYPES_H
