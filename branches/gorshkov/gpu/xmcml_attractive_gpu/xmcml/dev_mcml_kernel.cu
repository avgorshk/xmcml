#include "mcml_kernel_types.h"
#include "dev_mcml_intersection.cu"
#include "dev_mcml_mcg59.cu"

#define THREADS 128
#define BLOCKS 256

#define PI             3.14159265358979323846
#define COSINE_OF_ZERO (1.0 - 1.0E-12)
#define COSINE_OF_90D  1.0E-12
#define MIN_DISTANCE   1.0E-8

int GetMaxThreads()
{
    return THREADS*BLOCKS;
}

__device__ void LaunchPhoton(InputInfo* input, PhotonState* photon, double specularReflectance)
{
	photon->weight = 1.0 - specularReflectance;
	photon->isDead = false;
	photon->layerId = 1;
	photon->step = 0.0;
    photon->time = 0.0;
	
	photon->position.x = input->startPosition.x;
	photon->position.y = input->startPosition.y;
	photon->position.z = input->startPosition.z + MIN_DISTANCE;

	photon->direction.x = input->startDirection.x;
	photon->direction.y = input->startDirection.y;
	photon->direction.z = input->startDirection.z;

	for (int i = 0; i < input->numberOfLayers; ++i)
		photon->visitedLayers[i] = false;
	photon->visitedLayers[photon->layerId] = true;
}

__device__ void MovePhoton(PhotonState* photon)
{
    photon->position.x += photon->step*photon->direction.x;
    photon->position.y += photon->step*photon->direction.y;
    photon->position.z += photon->step*photon->direction.z;
}

__device__ Byte3 GetAreaIndexVector(Double3 photonPosition, Area* area)
{	
    Byte3 result;
    result.x = 255;
    result.y = 255;
    result.z = 255;

    double indexX = area->partitionNumber.x * (photonPosition.x - area->corner.x) / area->length.x;
    double indexY = area->partitionNumber.y * (photonPosition.y - area->corner.y) / area->length.y;
    double indexZ = area->partitionNumber.z * (photonPosition.z - area->corner.z) / area->length.z;
	
    bool isPhotonInArea = !((indexX < 0.0 || indexX >= area->partitionNumber.x) || 
        (indexY < 0.0 || indexY >= area->partitionNumber.y) ||
        (indexZ < 0.0 || indexZ >= area->partitionNumber.z));
    
    if (isPhotonInArea)
	{
        result.x = (byte)indexX;
        result.y = (byte)indexY;
        result.z = (byte)indexZ;
	}
	
    return result;
}

__device__ int GetAreaIndex(Double3 photonPosition, Area* area)
{	
    double indexX = area->partitionNumber.x * (photonPosition.x - area->corner.x) / area->length.x;
    double indexY = area->partitionNumber.y * (photonPosition.y - area->corner.y) / area->length.y;
    double indexZ = area->partitionNumber.z * (photonPosition.z - area->corner.z) / area->length.z;
	
    bool isNotPhotonInArea = (indexX < 0.0 || indexX >= area->partitionNumber.x) || 
        (indexY < 0.0 || indexY >= area->partitionNumber.y) ||
        (indexZ < 0.0 || indexZ >= area->partitionNumber.z);
    
    if (isNotPhotonInArea)
	{
		return -1;
	}
	
	return (int)indexX * area->partitionNumber.y * area->partitionNumber.z + 
        (int)indexY * area->partitionNumber.z + (int)indexZ;
}

__device__ void UpdatePhotonTrajectory(PhotonState* photon, InputInfo* input, GpuThreadOutput* output, 
    Double3 previousPhotonPosition)
{
    Double3 startPosition, finishPosition;
    if (photon->position.z > previousPhotonPosition.z)
    {
        startPosition = previousPhotonPosition;
        finishPosition = photon->position;
    }
    else if (photon->position.z < previousPhotonPosition.z)
    {
        startPosition = photon->position;
        finishPosition = previousPhotonPosition;
    }
    else
    {
        return;
    }

    double step = input->area->length.z / input->area->partitionNumber.z;
    int planeIndex = (int)((startPosition.z - input->area->corner.z + step / 2.0) / step);
    double plane = input->area->corner.z + planeIndex * step + step / 2.0;
    while ((plane <= finishPosition.z) && (plane < input->area->corner.z + input->area->length.z))
    {
        Double3 intersectionPoint = gpuGetPlaneSegmentIntersectionPoint(startPosition, finishPosition, plane);
        
        Byte3 areaIndexVector = GetAreaIndexVector(intersectionPoint, input->area);
        if (areaIndexVector.x == 255 && areaIndexVector.y == 255 && areaIndexVector.z == 255)
            break;

        if (output->trajectory.position >= MAX_TRAJECTORY_SIZE) printf("Assert trajectory size!\n");

        output->trajectory.x[output->trajectory.position] = areaIndexVector.x;
        output->trajectory.y[output->trajectory.position] = areaIndexVector.y;
        output->trajectory.z[output->trajectory.position] = areaIndexVector.z;
        ++(output->trajectory.position);
        
        plane += step;
    }
}

__device__ void MovePhotonAndUpdateTrajectory(PhotonState* photon, InputInfo* input, GpuThreadOutput* output)
{
    Double3 previousPhotonPosition = photon->position;
    photon->time += photon->step*input->layerInfo[photon->layerId].refractiveIndex;
    MovePhoton(photon);
    UpdatePhotonTrajectory(photon, input, output, previousPhotonPosition);
}

__device__ int FindIntersectionLayer(InputInfo* input, int surfaceId, int currectLayerId)
{
    for (int i = 0; i < input->numberOfLayers; ++i)
    {        
        for (int j = 0; j < input->layerInfo[i].numberOfSurfaces; ++j)
        {
            if (input->layerInfo[i].surfaceId[j] == surfaceId && i != currectLayerId)
            {
                return i;
            }
        }
    }

    return -1;
}

__device__ double GetCriticalCos(LayerInfo* layer, int currectLayerId, int intersectionLayerId)
{
    double currentRefractiveIndex = layer[currectLayerId].refractiveIndex;
    double intersectionRefractiveIndex = layer[intersectionLayerId].refractiveIndex;
    
    double criticalCos;
    if (currentRefractiveIndex > intersectionRefractiveIndex)
    {
        criticalCos = sqrt(1.0 - intersectionRefractiveIndex * intersectionRefractiveIndex / 
            (currentRefractiveIndex * currentRefractiveIndex));
    }
    else
    {
        criticalCos = 0.0;
    }

    return criticalCos;
}

__device__ double ComputeCosineTheta(double anisotropy, MCG59* randomGenerator)
{
	double cosTheta;
	double temp;
	double randomValue = NextMCG59(randomGenerator);

	if (anisotropy != 0.0)
	{
		temp = (1 - anisotropy * anisotropy) / (1 - anisotropy + 2 * anisotropy * randomValue);
		cosTheta = (1 + anisotropy * anisotropy - temp * temp) / (2 * anisotropy);
	}
	else
	{
		cosTheta = 2.0 * randomValue - 1.0;
	}
    
    if (cosTheta > 1.0) cosTheta = 1.0;
	else if (cosTheta < -1.0) cosTheta = -1.0;

	return cosTheta;
}

__device__ double ComputeTransmitCosine(double incidentRefractiveIndex, double transmitRefractiveIndex, 
	double incidentCos)
{
	double transmitCos;

	if (incidentRefractiveIndex == transmitRefractiveIndex)
	{
		transmitCos = incidentCos;
	} 
	else if (incidentCos > COSINE_OF_ZERO)
	{
		transmitCos = incidentCos;
	}
	else if (incidentCos < COSINE_OF_90D)
	{
		transmitCos = 0.0;
	}
	else
	{
		double incidentSin = sqrt(1.0 - incidentCos * incidentCos);
		double transmitSin = incidentRefractiveIndex * incidentSin / transmitRefractiveIndex;
		if (transmitSin >= 1.0)
		{
			transmitCos = 0.0;
		}
		else
		{
			transmitCos = sqrt(1.0 - transmitSin * transmitSin);
		}
	}

	return transmitCos;
}

__device__ double ComputeFrenselReflectance(double incidentRefractiveIndex, double transmitRefractiveIndex,
	double incidentCos, double transmitCos)
{
	double reflectance;

	if (incidentRefractiveIndex == transmitRefractiveIndex)
	{
		reflectance = 0.0;
	} 
	else if (incidentCos > COSINE_OF_ZERO)
	{
		reflectance = (transmitRefractiveIndex - incidentRefractiveIndex) / 
			(transmitRefractiveIndex + incidentRefractiveIndex);
		reflectance *= reflectance;
	}
	else if (incidentCos < COSINE_OF_90D)
	{
		reflectance = 1.0;
	}
	else
	{

		double incidentSin = sqrt(1.0 - incidentCos * incidentCos);
		double transmitSin = sqrt(1.0 - transmitCos * transmitCos);
		double cosSum = incidentCos * transmitCos - incidentSin * transmitSin;
		double cosDifference = incidentCos * transmitCos + incidentSin * transmitSin;
		double sinSum = incidentSin * transmitCos + incidentCos * transmitSin;
		double sinDifference = incidentSin * transmitCos - incidentCos * transmitSin;
		reflectance = 0.5 * sinDifference * sinDifference * (cosDifference * cosDifference +
			cosSum * cosSum) / (sinSum * sinSum * cosDifference * cosDifference);
	}

	return reflectance;
}

__device__ int GetDetectorId(PhotonState* photon, InputInfo* input)
{
	for (int i = 0; i < input->numberOfCubeDetectors; ++i)
    {
        Double3 halfLength = {input->cubeDetector[i].length.x / 2.0, input->cubeDetector[i].length.y / 2.0,
            input->cubeDetector[i].length.z / 2.0};
        bool isPhotonInDetector = (photon->position.x >= (input->cubeDetector[i].center.x - halfLength.x)) &&
            (photon->position.x < (input->cubeDetector[i].center.x + halfLength.x)) &&
            (photon->position.y >= (input->cubeDetector[i].center.y - halfLength.y)) &&
            (photon->position.y < (input->cubeDetector[i].center.y + halfLength.y)) &&
            (photon->position.z >= (input->cubeDetector[i].center.z - halfLength.z)) &&
            (photon->position.z < (input->cubeDetector[i].center.z + halfLength.z));
		bool isPhotonVisitedTargetLayer = photon->visitedLayers[input->cubeDetector[i].targetLayer];
        if (isPhotonInDetector && isPhotonVisitedTargetLayer)
        {  
            return i;
        }
    }

	for (int i = 0; i < input->numberOfRingDetectors; ++i)
	{
		double distance = sqrt(((photon->position.x - input->ringDetector[i].center.x) * 
			(photon->position.x - input->ringDetector[i].center.x)) +
			((photon->position.y - input->ringDetector[i].center.y) * 
			(photon->position.y - input->ringDetector[i].center.y)));
		bool isPhotonInDetector = ((distance >= input->ringDetector[i].smallRadius) && 
			(distance < input->ringDetector[i].bigRadius));
		bool isPhotonVisitedTargetLayer = photon->visitedLayers[input->ringDetector[i].targetLayer];
		if (isPhotonInDetector && isPhotonVisitedTargetLayer)
        {
            return input->numberOfCubeDetectors + i;
        }
	}

	return -1;
}

__device__ Double3 ReflectVector(Double3 incidentVector, Double3 normalVector)
{
    Double3 scaledNormalVector = gpuNormalizeVector(normalVector);
    Double3 scaledIncidentVector = gpuNormalizeVector(incidentVector);
    
    double tmp = 2 * gpuDotVector(scaledIncidentVector, scaledNormalVector);
    scaledNormalVector.x *= tmp;
    scaledNormalVector.y *= tmp;
    scaledNormalVector.z *= tmp;

    return gpuNormalizeVector(gpuSubVector(scaledIncidentVector, scaledNormalVector));
}

__device__ Double3 RefractVector(double incidentRefractiveIndex, double transmitRefractiveIndex, 
    Double3 incidentVector, Double3 normalVector)
{
    Double3 scaledIncidentVector = gpuNormalizeVector(incidentVector);
    scaledIncidentVector.x *= incidentRefractiveIndex;
    scaledIncidentVector.y *= incidentRefractiveIndex;
    scaledIncidentVector.z *= incidentRefractiveIndex;

    Double3 scaledNormalVector = gpuNormalizeVector(normalVector);

    double tmp1 = gpuDotVector(scaledIncidentVector, scaledNormalVector);
    double tmp2 = sqrt((transmitRefractiveIndex * transmitRefractiveIndex - 
        incidentRefractiveIndex * incidentRefractiveIndex) / (tmp1 * tmp1) + 1);
    tmp2 = (tmp2 - 1) * tmp1;

    Double3 transmitVector;
    transmitVector.x = scaledIncidentVector.x + tmp2 * scaledNormalVector.x;
    transmitVector.y = scaledIncidentVector.y + tmp2 * scaledNormalVector.y;
    transmitVector.z = scaledIncidentVector.z + tmp2 * scaledNormalVector.z;

    return gpuNormalizeVector(transmitVector);
}

__device__ void MovePhotonOnMinDistance(PhotonState* photon)
{
    photon->position.x += MIN_DISTANCE * photon->direction.x;
    photon->position.y += MIN_DISTANCE * photon->direction.y;
    photon->position.z += MIN_DISTANCE * photon->direction.z;
}

__device__ void MovePhotonOnMinDistanceAndUpdateTrajectory(PhotonState* photon, InputInfo* input, GpuThreadOutput* output)
{
    Double3 previousPhotonPosition = photon->position;
    MovePhotonOnMinDistance(photon);
    UpdatePhotonTrajectory(photon, input, output, previousPhotonPosition);
}

__device__ void CrossBoundary(PhotonState* photon, InputInfo* input, GpuThreadOutput* output, 
    IntersectionInfo* intersection, MCG59* randomGenerator)
{
    double reflectance;
    double transmitCos;
    double incidentCos = gpuDotVector(photon->direction, intersection->normal);
    if (incidentCos < 0.0)
    {
        incidentCos = -incidentCos;
    }

    int layerId = photon->layerId;
    int intersectionLayerId = FindIntersectionLayer(input, intersection->surfaceId, layerId);
    double criticalCos = GetCriticalCos(input->layerInfo, layerId, intersectionLayerId);
    if (incidentCos <= criticalCos)
    {
        reflectance = 1.0;
    }
    else
    {
        transmitCos = ComputeTransmitCosine(input->layerInfo[layerId].refractiveIndex, 
            input->layerInfo[intersectionLayerId].refractiveIndex, incidentCos);
        reflectance = ComputeFrenselReflectance(input->layerInfo[layerId].refractiveIndex,
            input->layerInfo[intersectionLayerId].refractiveIndex, incidentCos, transmitCos);
    }

    if (NextMCG59(randomGenerator) > reflectance) //transmitting
    {
        if (intersectionLayerId == 0)
        {
            int detectorId = GetDetectorId(photon, input);
            if (detectorId >= 0)
            {
                output->detectorId = detectorId;
                output->weight = photon->weight;
                output->time = photon->time;
            }
            photon->isDead = true;
        }
        else
        {
            photon->direction = RefractVector(input->layerInfo[layerId].refractiveIndex, 
                input->layerInfo[intersectionLayerId].refractiveIndex, photon->direction,
                intersection->normal);
            photon->layerId = intersectionLayerId;
			photon->visitedLayers[photon->layerId] = true;
            MovePhotonOnMinDistanceAndUpdateTrajectory(photon, input, output);
        }
    }
    else //reflection
    {
        photon->direction = ReflectVector(photon->direction, intersection->normal);
        MovePhotonOnMinDistanceAndUpdateTrajectory(photon, input, output);
    }
}

__device__ void ComputeStepSizeInTissue(PhotonState* photon, InputInfo* info, MCG59* randomGenerator)
{
    int layerId = photon->layerId;
    double absorptionCoefficient = info->layerInfo[layerId].absorptionCoefficient;
    double scatteringCoefficient = info->layerInfo[layerId].scatteringCoefficient;
    double randomValue;
        
    do
    {
        randomValue = NextMCG59(randomGenerator);
    } while (randomValue == 0.0);

    photon->step = -log(randomValue) / (absorptionCoefficient + scatteringCoefficient);
}

__device__ void ComputeDroppedWeightOfPhoton(PhotonState* photon, InputInfo* input)
{
	int layerId = photon->layerId;
	double absorptionCoefficient = input->layerInfo[layerId].absorptionCoefficient;
	double scatteringCoefficient = input->layerInfo[layerId].scatteringCoefficient;
	double deltaWeight = photon->weight * absorptionCoefficient / (absorptionCoefficient + scatteringCoefficient);
	photon->weight -= deltaWeight;
}

__device__ Double3 GetLocalAttractiveVector(PhotonState* photon, InputInfo* input)
{
    LayerInfo* layer = input->layerInfo + photon->layerId;
    
    Double3 r = gpuSubVector(photon->position, input->startPosition);
    Double3 rd = gpuSubVector(input->targetPoint, input->startPosition);

    double D = 1/(3*((1 - layer->anisotropy)*layer->scatteringCoefficient + layer->absorptionCoefficient));
    double ltr = 3*D;
    double delta = 0.74*ltr;

    Double3 r1 = rd;
    r1.z += ltr;

    Double3 r2 = rd;
    r2.z -= ltr + 2*delta;

    Double3 tmp1 = gpuSubVector(r, r1);
    double l1 = gpuLengthOfVector(tmp1);
    tmp1 = gpuDivVector(tmp1, l1*l1*l1);
    
    Double3 tmp2 = gpuSubVector(r, r2);
    double l2 = gpuLengthOfVector(tmp2);
    tmp2 = gpuDivVector(tmp2, l2*l2*l2);

    return gpuNormalizeVector(gpuSubVector(tmp2, tmp1));
}

__device__ int GetWeightTableIndex(InputInfo* input, double anisotropy)
{
    int index = -1;
    for (int i = 0; i < input->numberOfWeightTables; ++i)
    {
        if (anisotropy == input->weightTable[i].anisotropy)
            index = i;
    }
    return index;
}

__device__ double GetNewPhotonWeight(InputInfo* input, PhotonState* photon, Double3 attractiveVector, double p)
{
    double anisotropy = input->layerInfo[photon->layerId].anisotropy;
    Double3 initDirection = photon->direction;
    int weightTableIndex = GetWeightTableIndex(input, anisotropy);

    double a = -1.0;
    double b = 1.0;
    int n = (input->weightTable[weightTableIndex].numberOfElements - 1);
    double step = (b - a) / n;
    double dot = gpuDotVector(initDirection, attractiveVector);
    int elementIndex = (int)floor((dot - a) / step + 0.5);
    return photon->weight * input->weightTable[weightTableIndex].elements[elementIndex] / p;
}

__device__ Double3 GetNewPhotonDirection(PhotonState* photon, double anisotropy, MCG59* randomGenerator)
{
    Double3 result;
    double temp;
	double ux = photon->direction.x;
	double uy = photon->direction.y;
	double uz = photon->direction.z;

	double cosTheta = ComputeCosineTheta(anisotropy, randomGenerator);
	double sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	double psi = 2.0 * PI * NextMCG59(randomGenerator);
	double cosPsi = cos(psi);
	double sinPsi = sin(psi);

	if (abs(uz) <= COSINE_OF_ZERO)
	{
		temp = sqrt(1.0 - uz * uz);
		result.x = sinTheta * (ux * uz * cosPsi - uy * sinPsi) / temp + ux * cosTheta;
		result.y = sinTheta * (uy * uz * cosPsi + ux * sinPsi) / temp + uy * cosTheta;
		result.z = -sinTheta * cosPsi * temp + uz * cosTheta;
	}
	else
	{
		temp = (uz >= 0.0) ? 1.0 : -1.0;
		result.x = sinTheta * cosPsi;
		result.y = sinTheta * sinPsi;
		result.z = temp * cosTheta;
	}

    return result;
}

__device__ void ComputePhotonDirectionWithBiasing(PhotonState* photon, InputInfo* input, MCG59* randomGenerator)
{
    double anisotropy = input->layerInfo[photon->layerId].anisotropy;
    Double3 attractiveVector = GetLocalAttractiveVector(photon, input);
    Double3 newDirection;
    
    double p = 0.0;
    do
    {
        newDirection = GetNewPhotonDirection(photon, anisotropy, randomGenerator);
        p = 0.1 + 0.9*((1 + gpuDotVector(newDirection, attractiveVector))/2);
        p = pow(p, input->attractiveFactor);
    } while(p <= NextMCG59(randomGenerator));

    photon->weight = GetNewPhotonWeight(input, photon, attractiveVector, p);
    photon->direction = newDirection;
}

__global__ void GpuComputePhoton(double specularReflectance, InputInfo* input, GpuThreadOutput* output, 
    MCG59* randomGenerator, int numberOfPhotons)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < numberOfPhotons)
    {
        PhotonState photon;
    
        output += tid;
        randomGenerator += tid;

        output->detectorId = -1;
        output->trajectory.position = 0;
    
        LaunchPhoton(input, &photon, specularReflectance);
        while (!photon.isDead)
        {
            int layerId = photon.layerId;
            bool isPhotonInGlass = (input->layerInfo[layerId].absorptionCoefficient == 0.0) &&
                (input->layerInfo[layerId].scatteringCoefficient == 0.0);
        
            if (isPhotonInGlass)
            {
                IntersectionInfo intersection = GpuComputeBVHIntersection(photon.position, 
                    photon.direction, MAX_DISTANCE, input->bvhTree, input->surface);

                if (intersection.isFindIntersection)
                {
                    photon.step = intersection.distance - 0.01 * MIN_DISTANCE;
                    MovePhotonAndUpdateTrajectory(&photon, input, output);
                
                    int areaIndex = GetAreaIndex(photon.position, input->area);
                    if (areaIndex >= 0)
                    {
                        CrossBoundary(&photon, input, output, &intersection, randomGenerator);
                    }
                    else
                    {
                        photon.isDead = true;
                    }
                }
                else
                {
                    photon.isDead = true;
                }
            }
            else
            {
                ComputeStepSizeInTissue(&photon, input, randomGenerator);

                IntersectionInfo intersection = GpuComputeBVHIntersection(photon.position, photon.direction,
                    photon.step, input->bvhTree, input->surface);

                if (intersection.isFindIntersection)
                {
                    photon.step = intersection.distance - 0.01 * MIN_DISTANCE;   
                    MovePhotonAndUpdateTrajectory(&photon, input, output);

                    int areaIndex = GetAreaIndex(photon.position, input->area);
                    if (areaIndex >= 0)
                    {
                        CrossBoundary(&photon, input, output, &intersection, randomGenerator); 
                    }
                    else
                    {
                        photon.isDead = true;
                    }
                }
                else
                {
                    MovePhotonAndUpdateTrajectory(&photon, input, output);

                    int areaIndex = GetAreaIndex(photon.position, input->area);
                    if (areaIndex >= 0)
                    {
                        ComputeDroppedWeightOfPhoton(&photon, input);
                        ComputePhotonDirectionWithBiasing(&photon, input, randomGenerator);
                    }
                    else
                    {
                        photon.isDead = true;
                    }
                }
            }

            if (photon.weight < input->minWeight)
            {
                photon.isDead = true;
            }
        }
    }
}

void ComputePhotonBlock(double specularReflectance, InputInfo* gpuInput, MCG59* gpuRandomGenerator, 
    GpuThreadOutput* gpuOutput, int numberOfPhotons, cudaStream_t stream)
{
    int blocks = (numberOfPhotons + THREADS - 1)/THREADS;
    GpuComputePhoton<<<blocks, THREADS, 0, stream>>>(specularReflectance, gpuInput, gpuOutput, gpuRandomGenerator, numberOfPhotons);
}