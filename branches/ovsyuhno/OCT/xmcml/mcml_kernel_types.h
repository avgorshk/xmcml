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
    double3 center;
    double3 length;
	bool filterLayers[MAX_LAYERS];
	double permissibleAngle;
} CubeDetector;

typedef struct __RingDetector
{
	double3 center;
	double smallRadius;
	double bigRadius;
	bool filterLayers[MAX_LAYERS];
	double permissibleAngle;
} RingDetector;

typedef struct __DetectorInfo 
{
    uint64 numberOfPhotons;
    uint64* trajectory;
    int trajectorySize;
    TimeInfo* timeScale;
    int timeScaleSize;
	double targetRange;
	double weight;
	double permissibleAngle;
} DetectorInfo;

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

typedef struct __StartDirectionInfo
{
	int startDirectionMode;	//1 - use Gauss direction, 0 - use start direction(for all photons)
	double3 startDirection;
	double standardDeviation;
	double distance;
}StartDirectionInfo;

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
	StartDirectionInfo startDirectionInfo; 
    byte useBiasing;
    uint weightIntegralPrecision;
    uint weightTablePrecision;
    double attractiveFactor;
    WeightIntegralTable* weightTable;
    int numberOfWeightTables;
    double3 targetPoint;
	unsigned char targetRangeLayers[MAX_LAYERS];
	double standardDeviation;
} InputInfo;

typedef struct __OutputInfo
{
    uint64 numberOfPhotons;
	double specularReflectance;
	double* absorption;
    int gridSize;
    DetectorInfo* detectorInfo;
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
	double targetRange;
	double otherRange;
	bool isDead;
} PhotonState;

typedef struct __PhotonTrajectory
{
    byte x[MAX_TRAJECTORY_SIZE];
    byte y[MAX_TRAJECTORY_SIZE];
    byte z[MAX_TRAJECTORY_SIZE];
    int position;
} PhotonTrajectory;

#endif //__MCML_KERNEL_TYPES_H
