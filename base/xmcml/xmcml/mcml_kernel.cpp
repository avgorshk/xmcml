#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>

#include "mcml_kernel.h"
#include "mcml_math.h"

#define PI             3.14159265358979323846
#define COSINE_OF_ZERO (1.0 - 1.0E-12)
#define COSINE_OF_90D  1.0E-12
#define MIN_DISTANCE   1.0E-8

double ComputeSpecularReflectance(LayerInfo* layer)
{
	double directReflection1, directReflection2;
	double temp;

	temp = (layer[0].refractiveIndex - layer[1].refractiveIndex) / 
		(layer[0].refractiveIndex + layer[1].refractiveIndex);
	directReflection1 = temp * temp;

	if (layer[1].absorptionCoefficient == 0.0 && layer[1].scatteringCoefficient == 0.0)
	{
		temp = (layer[1].refractiveIndex - layer[2].refractiveIndex) / 
			(layer[1].refractiveIndex + layer[2].refractiveIndex);
		directReflection2 = temp * temp;

		temp = 1 - directReflection1;
		directReflection1 = directReflection1 + temp * temp * directReflection2 / 
			(1 - directReflection1 * directReflection2);
	}

	return directReflection1;
}

void ComputePhoton(double specularReflectance, InputInfo* input, OutputInfo* output, 
    MCG59* randomGenerator, PhotonTrajectory* trajectory)
{
    PhotonState photon;

    trajectory->position = 0;
    
    LaunchPhoton(input, &photon, specularReflectance);
    while (!photon.isDead)
    {
        int layerId = photon.layerId;
        bool isPhotonInGlass = (input->layerInfo[layerId].absorptionCoefficient == 0.0) &&
            (input->layerInfo[layerId].scatteringCoefficient == 0.0);
        
        if (isPhotonInGlass)
        {
            IntersectionInfo intersection = ComputeBVHIntersectionWithoutStep(photon.position, 
                photon.direction, input->bvhTree, input->surface);

            if (intersection.isFindIntersection)
            {
                photon.step = intersection.distance - 0.01 * MIN_DISTANCE;
                MovePhotonAndUpdateTrajectory(&photon, input, trajectory);
                
                int areaIndex = GetAreaIndex(photon.position, input->area);
                if (areaIndex >= 0)
                {
                    CrossBoundary(&photon, input, output, &intersection, randomGenerator, trajectory);
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

            IntersectionInfo intersection = ComputeBVHIntersection(photon.position, photon.direction,
                photon.step, input->bvhTree, input->surface);

            if (intersection.isFindIntersection)
            {
                photon.step = intersection.distance - 0.01 * MIN_DISTANCE;   
                MovePhotonAndUpdateTrajectory(&photon, input, trajectory);

                int areaIndex = GetAreaIndex(photon.position, input->area);
                if (areaIndex >= 0)
                {
                    CrossBoundary(&photon, input, output, &intersection, randomGenerator, trajectory); 
                }
                else
                {
                    photon.isDead = true;
                }
            }
            else
            {
                MovePhotonAndUpdateTrajectory(&photon, input, trajectory);

                int areaIndex = GetAreaIndex(photon.position, input->area);
                if (areaIndex >= 0)
                {
                    ComputeDroppedWeightOfPhoton(&photon, input, output, areaIndex);
                    if (input->useBiasing)
                        ComputePhotonDirectionWithBiasing(&photon, input, randomGenerator);
                    else
                        photon.direction = GetNewPhotonDirection(&photon, 
                            input->layerInfo[photon.layerId].anisotropy, randomGenerator);
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

void LaunchPhoton(InputInfo* input, PhotonState* photon, double specularReflectance)
{
	photon->weight = 1.0 - specularReflectance;
	photon->isDead = false;
	photon->layerId = 1;
	photon->step = 0.0;
    photon->time = 0.0;
	photon->targetRange = 0.0;
	photon->otherRange = 0.0;
	
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

void MovePhoton(PhotonState* photon)
{
    photon->position.x += photon->step * photon->direction.x;
    photon->position.y += photon->step * photon->direction.y;
    photon->position.z += photon->step * photon->direction.z;
}

void MovePhotonAndUpdateTrajectory(PhotonState* photon, InputInfo* input, PhotonTrajectory* trajectory)
{
    double3 previousPhotonPosition = photon->position;
	photon->time += photon->step*input->layerInfo[photon->layerId].refractiveIndex;
	
	if (input->targetRangeLayers[photon->layerId])
	{
		photon->targetRange += photon->step;
	}
	else
	{
		photon->otherRange += photon->step;
	}

    MovePhoton(photon);
    UpdatePhotonTrajectory(photon, input, trajectory, previousPhotonPosition);
}

void MovePhotonOnMinDistance(PhotonState* photon)
{
    photon->position.x += MIN_DISTANCE * photon->direction.x;
    photon->position.y += MIN_DISTANCE * photon->direction.y;
    photon->position.z += MIN_DISTANCE * photon->direction.z;
}

void MovePhotonOnMinDistanceAndUpdateTrajectory(PhotonState* photon, InputInfo* input, PhotonTrajectory* trajectory)
{
    double3 previousPhotonPosition = photon->position;
    MovePhotonOnMinDistance(photon);
    UpdatePhotonTrajectory(photon, input, trajectory, previousPhotonPosition);
}

int GetAreaIndex(double3 photonPosition, Area* area)
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

byte3 GetAreaIndexVector(double3 photonPosition, Area* area)
{	
    byte3 result;
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

void CrossBoundary(PhotonState* photon, InputInfo* input, OutputInfo* output, 
    IntersectionInfo* intersection, MCG59* randomGenerator, PhotonTrajectory* trajectory)
{
    double reflectance;
    double transmitCos;
    double incidentCos = DotVector(photon->direction, intersection->normal);
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
				UpdateWeightInDetector(output, photon->weight, detectorId);
                UpdateDetectorTrajectory(output, input->area, trajectory, detectorId);
                UpdateDetectorTimeScale(output, photon, detectorId);
				UpdateDetectorRanges(output, photon, detectorId);
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
            MovePhotonOnMinDistanceAndUpdateTrajectory(photon, input, trajectory);
        }
    }
    else //reflection
    {
        photon->direction = ReflectVector(photon->direction, intersection->normal);
        MovePhotonOnMinDistanceAndUpdateTrajectory(photon, input, trajectory);
    }
}

double GetAverageAnisotropy(LayerInfo* layer, int currentLayerId)
{
    double lengths[] = {2.1, 3.2, 6.9, 2.5, 5.8, 60.0};
    double totalLength = 0;
    double g = 0;
    for (int i = 0; i < currentLayerId; ++i)
    {
        g += layer[i + 1].anisotropy*lengths[i];
        totalLength += lengths[i];
    }
    return g/totalLength;
}

double GetAverageScatteringCoefficient(LayerInfo* layer, int currentLayerId)
{
    double lengths[] = {2.1, 3.2, 6.9, 2.5, 5.8, 60.0};
    double totalLength = 0;
    double mus = 0;
    for (int i = 0; i < currentLayerId; ++i)
    {
        mus += layer[i + 1].scatteringCoefficient*lengths[i];
        totalLength += lengths[i];
    }
    return mus/totalLength;
}

double GetAverageAbsorptionCoefficient(LayerInfo* layer, int currentLayerId)
{
    double lengths[] = {2.1, 3.2, 6.9, 2.5, 5.8, 60.0};
    double totalLength = 0;
    double mua = 0;
    for (int i = 0; i < currentLayerId; ++i)
    {
        mua += layer[i + 1].absorptionCoefficient*lengths[i];
        totalLength += lengths[i];
    }
    return mua/totalLength;
}

double3 GetLocalAttractiveVector(PhotonState* photon, InputInfo* input)
{
    LayerInfo* layer = input->layerInfo + photon->layerId;
    
    /*double g = GetAverageAnisotropy(input->layerInfo, photon->layerId);
    double mus = GetAverageScatteringCoefficient(input->layerInfo, photon->layerId);
    double mua = GetAverageAbsorptionCoefficient(input->layerInfo, photon->layerId);*/
    double g = layer->anisotropy;
    double mus = layer->scatteringCoefficient;
    double mua = layer->absorptionCoefficient;

    double3 r = SubVector(photon->position, input->startPosition);
    double3 rd = SubVector(input->targetPoint, input->startPosition);

    double D = 1/(3*((1 - g)*mus + mua));
    double ltr = 3*D;
    double delta = 0.74*ltr;
    //double delta = 2.0/3.0*ltr;
    //double delta = 0.71*ltr;

    double3 r1 = rd;
    r1.z += ltr;

    double3 r2 = rd;
    r2.z -= ltr + 2*delta;

    double3 tmp1 = SubVector(r, r1);
    double l1 = LengthOfVector(tmp1);
    tmp1 = DivVector(tmp1, l1*l1*l1);
    
    double3 tmp2 = SubVector(r, r2);
    double l2 = LengthOfVector(tmp2);
    tmp2 = DivVector(tmp2, l2*l2*l2);

    return NormalizeVector(SubVector(tmp2, tmp1));
}

//double3 ComputeLocalAttractiveVectorPart(double3 r, double3 rn, double alpha)
//{
//    double3 vector = SubVector(r, rn);
//    double length = LengthOfVector(vector);
//    double coeff = exp(-alpha*length)*(alpha*length + 1)/(length*length*length);
//    return MulVector(vector, coeff);
//}
//
//double3 GetLocalAttractiveVector(PhotonState* photon, InputInfo* input)
//{
//    LayerInfo* layer = input->layerInfo + photon->layerId;
//    
//    double g = GetAverageAnisotropy(input->layerInfo, photon->layerId);
//    double mus = GetAverageScatteringCoefficient(input->layerInfo, photon->layerId);
//    double mua = GetAverageAbsorptionCoefficient(input->layerInfo, photon->layerId);
//
//    double3 r = SubVector(photon->position, input->startPosition);
//    double3 rd = SubVector(input->targetPoint, input->startPosition);
//
//    double alpha = sqrt(3*mua*((1 - g)*mus + mua));
//    double D = 1/(3*((1 - g)*mus + mua));
//    double ltr = 3*D;
//    double delta = 0.74*ltr;
//    //double delta = 2.0/3.0*ltr;
//    //double delta = 0.71*ltr;
//
//    double3 r1 = rd;
//    r1.z += ltr;
//
//    double3 r2 = rd;
//    r2.z -= ltr + 2*delta;
//
//    return NormalizeVector(SubVector(ComputeLocalAttractiveVectorPart(r, r2, alpha), 
//        ComputeLocalAttractiveVectorPart(r, r1, alpha)));
//}

void ComputePhotonDirectionWithBiasing(PhotonState* photon, InputInfo* input, MCG59* randomGenerator)
{
    double anisotropy = input->layerInfo[photon->layerId].anisotropy;
    double3 attractiveVector = GetLocalAttractiveVector(photon, input);
    double3 newDirection;
    
    double p = 0.0;
    do
    {
        newDirection = GetNewPhotonDirection(photon, anisotropy, randomGenerator);
        p = 0.1 + 0.9*((1 + DotVector(newDirection, attractiveVector))/2);
        p = pow(p, input->attractiveFactor);
    } while(p <= NextMCG59(randomGenerator));

    photon->weight = GetNewPhotonWeight(input, photon, attractiveVector, p);
    photon->direction = newDirection;
}

double GetNewPhotonWeight(InputInfo* input, PhotonState* photon, double3 attractiveVector, double p)
{
    double anisotropy = input->layerInfo[photon->layerId].anisotropy;
    double3 initDirection = photon->direction;
    int weightTableIndex = GetWeightTableIndex(input, anisotropy);

    double a = -1.0;
    double b = 1.0;
    int n = (input->weightTable[weightTableIndex].numberOfElements - 1);
    double step = (b - a) / n;
    double dot = DotVector(initDirection, attractiveVector);
    int elementIndex = (int)floor((dot - a) / step + 0.5);
    return photon->weight * input->weightTable[weightTableIndex].elements[elementIndex] / p;
}

int GetWeightTableIndex(InputInfo* input, double anisotropy)
{
    int index = -1;
    for (int i = 0; i < input->numberOfWeightTables; ++i)
    {
        if (anisotropy == input->weightTable[i].anisotropy)
            index = i;
    }
    return index;
}

double3 GetNewPhotonDirection(PhotonState* photon, double anisotropy, MCG59* randomGenerator)
{
    double3 result;
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

double ComputeCosineTheta(double anisotropy, MCG59* randomGenerator)
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

double ComputeTransmitCosine(double incidentRefractiveIndex, double transmitRefractiveIndex, 
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

double ComputeFrenselReflectance(double incidentRefractiveIndex, double transmitRefractiveIndex,
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

int FindIntersectionLayer(InputInfo* input, int surfaceId, int currectLayerId)
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

double3 ReflectVector(double3 incidentVector, double3 normalVector)
{
    double3 scaledNormalVector = NormalizeVector(normalVector);
    double3 scaledIncidentVector = NormalizeVector(incidentVector);
    
    double tmp = 2 * DotVector(scaledIncidentVector, scaledNormalVector);
    scaledNormalVector.x *= tmp;
    scaledNormalVector.y *= tmp;
    scaledNormalVector.z *= tmp;

    return NormalizeVector(SubVector(scaledIncidentVector, scaledNormalVector));
}

double3 RefractVector(double incidentRefractiveIndex, double transmitRefractiveIndex, 
    double3 incidentVector, double3 normalVector)
{
    double3 scaledIncidentVector = NormalizeVector(incidentVector);
    scaledIncidentVector.x *= incidentRefractiveIndex;
    scaledIncidentVector.y *= incidentRefractiveIndex;
    scaledIncidentVector.z *= incidentRefractiveIndex;

    double3 scaledNormalVector = NormalizeVector(normalVector);

    double tmp1 = DotVector(scaledIncidentVector, scaledNormalVector);
    double tmp2 = sqrt((transmitRefractiveIndex * transmitRefractiveIndex - 
        incidentRefractiveIndex * incidentRefractiveIndex) / (tmp1 * tmp1) + 1);
    tmp2 = (tmp2 - 1) * tmp1;

    double3 transmitVector;
    transmitVector.x = scaledIncidentVector.x + tmp2 * scaledNormalVector.x;
    transmitVector.y = scaledIncidentVector.y + tmp2 * scaledNormalVector.y;
    transmitVector.z = scaledIncidentVector.z + tmp2 * scaledNormalVector.z;

    return NormalizeVector(transmitVector);
}

void ComputeStepSizeInTissue(PhotonState* photon, InputInfo* info, MCG59* randomGenerator)
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

void ComputeDroppedWeightOfPhoton(PhotonState* photon, InputInfo* input, OutputInfo* output, int droppedIndex)
{
	int layerId = photon->layerId;
	double absorptionCoefficient = input->layerInfo[layerId].absorptionCoefficient;
	double scatteringCoefficient = input->layerInfo[layerId].scatteringCoefficient;
	double deltaWeight = photon->weight * absorptionCoefficient / (absorptionCoefficient + scatteringCoefficient);

	photon->weight -= deltaWeight;
    output->absorption[droppedIndex] += 1.0 / scatteringCoefficient;
}

double GetCriticalCos(LayerInfo* layer, int currectLayerId, int intersectionLayerId)
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

int GetDetectorId(PhotonState* photon, InputInfo* input)
{
	for (int i = 0; i < input->numberOfCubeDetectors; ++i)
    {
        double3 halfLength = {input->cubeDetector[i].length.x / 2.0, input->cubeDetector[i].length.y / 2.0,
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

void UpdateWeightInDetector(OutputInfo* output, double photonWeight, int detectorId)
{
    #pragma omp atomic
	output->weightInDetector[detectorId] += photonWeight;
}

void UpdatePhotonTrajectory(PhotonState* photon, InputInfo* input, PhotonTrajectory* trajectory, 
    double3 previousPhotonPosition)
{
    double3 startPosition, finishPosition;
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
        double3 intersectionPoint = GetPlaneSegmentIntersectionPoint(startPosition, finishPosition, plane);
        
        byte3 areaIndexVector = GetAreaIndexVector(intersectionPoint, input->area);
        if (areaIndexVector.x == 255 && areaIndexVector.y == 255 && areaIndexVector.z == 255)
            break;

        assert(trajectory->position < MAX_TRAJECTORY_SIZE);

        trajectory->x[trajectory->position] = areaIndexVector.x;
        trajectory->y[trajectory->position] = areaIndexVector.y;
        trajectory->z[trajectory->position] = areaIndexVector.z;
        ++(trajectory->position);
        
        plane += step;
    }
}

void UpdateDetectorTrajectory(OutputInfo* output, Area* area, PhotonTrajectory* trajectory, int detectorId)
{
    int index;
    uint64* detectorTrajectory = output->detectorTrajectory[detectorId].trajectory;

    for (int i = 0; i < trajectory->position; ++i)
    {
        index = trajectory->x[i] * area->partitionNumber.y * area->partitionNumber.z + 
            trajectory->y[i] * area->partitionNumber.z + trajectory->z[i];
        
        #pragma omp atomic
        ++(detectorTrajectory[index]);
    }

    #pragma omp atomic
    ++(output->detectorTrajectory[detectorId].numberOfPhotons);
}

void UpdateDetectorTimeScale(OutputInfo* output, PhotonState* photon, int detectorId)
{
    int timeScaleSize = output->detectorTrajectory[detectorId].timeScaleSize;
    TimeInfo* timeScale = output->detectorTrajectory[detectorId].timeScale;
    for (int i = 0; i < timeScaleSize - 1; ++i)
    {
        if ((photon->time >= timeScale[i].timeStart) && (photon->time < timeScale[i].timeFinish))
        {
            #pragma omp atomic
            ++(timeScale[i].numberOfPhotons);

            #pragma omp atomic
            timeScale[i].weight += photon->weight;

            return;
        }
    }
    if (photon->time >= timeScale[timeScaleSize - 1].timeStart)
    {
        #pragma omp atomic
        ++(timeScale[timeScaleSize - 1].numberOfPhotons);

        #pragma omp atomic
        timeScale[timeScaleSize - 1].weight += photon->weight;
    }
}

void UpdateDetectorRanges(OutputInfo* output, PhotonState* photon, int detectorId)
{
	double totalRange  = photon->targetRange + photon->otherRange;
	double targetRange = photon->weight * (photon->targetRange / totalRange);
	double otherRange  = photon->weight * (photon->otherRange  / totalRange);	
	
	#pragma omp atomic
	output->detectorTrajectory[detectorId].targetRange += targetRange;

	#pragma omp atomic
	output->detectorTrajectory[detectorId].otherRange  += otherRange;
}
