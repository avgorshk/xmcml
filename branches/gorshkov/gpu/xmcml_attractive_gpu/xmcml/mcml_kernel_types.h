#ifndef __MCML_KERNEL_TYPES_H
#define __MCML_KERNEL_TYPES_H

#include "mcml_types.h"
#include "mcml_intersection.h"

#define MAX_LAYERS 8
#define MAX_TRAJECTORY_SIZE 20480

typedef struct __TimeInfo
{
    double timeStart;
    double timeFinish;
    uint64 numberOfPhotons;
    double weight;
} TimeInfo;

typedef struct __CubeDetector
{
    Double3 center;
    Double3 length;
	int targetLayer;
} CubeDetector;

typedef struct __RingDetector
{
	Double3 center;
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

typedef struct __WeightIntegralTable
{
    double anisotropy;
    int numberOfElements;
    double* elements;
} WeightIntegralTable;

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
	Double3 startPosition;
	Double3 startDirection;
    uint weightIntegralPrecision;
    uint weightTablePrecision;
    double attractiveFactor;
    WeightIntegralTable* weightTable;
    int numberOfWeightTables;
    Double3 targetPoint;
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

typedef struct __PhotonTrajectory
{
    byte x[MAX_TRAJECTORY_SIZE];
    byte y[MAX_TRAJECTORY_SIZE];
    byte z[MAX_TRAJECTORY_SIZE];
    int position;
} PhotonTrajectory;

typedef struct __GpuThreadOutput
{
    int detectorId;
    double weight;
    double time;
    PhotonTrajectory trajectory;
} GpuThreadOutput;

typedef struct __PhotonState
{
	Double3 position;
	Double3 direction;
	double step;
    int layerId;
	bool visitedLayers[MAX_LAYERS];
	double weight;
    double time;
	bool isDead;
} PhotonState;

#endif //__MCML_KERNEL_TYPES_H