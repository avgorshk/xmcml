#include <math.h>
#include <memory.h>

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

void LaunchPhoton(PhotonState* photon, double specularReflectance)
{
	photon->weight = 1.0 - specularReflectance;
	photon->isDead = false;
	photon->layerId = 1;
	photon->step = 0.0;
    photon->time = 0.0;
	
	photon->position.x = 0.0;
	photon->position.y = 0.0;
	photon->position.z = MIN_DISTANCE;

	photon->direction.x = 0.0;
	photon->direction.y = 0.0;
	photon->direction.z = 1.0;
}

void ComputePhoton(double specularReflectance, InputInfo* input, OutputInfo* output, 
    MCG59* randomGenerator, uint64* trajectory)
{
    PhotonState photon;

    int trajectorySize = input->area->partitionNumber.x * input->area->partitionNumber.y * 
        input->area->partitionNumber.z;
    memset(trajectory, 0, trajectorySize * sizeof(uint64));
    
    LaunchPhoton(&photon, specularReflectance);
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
                    ComputePhotonDirection(&photon, input->layerInfo[photon.layerId].anisotropy, 
                        randomGenerator);
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

void MovePhoton(PhotonState* photon)
{
    photon->position.x += photon->step * photon->direction.x;
    photon->position.y += photon->step * photon->direction.y;
    photon->position.z += photon->step * photon->direction.z;
}

void MovePhotonAndUpdateTrajectory(PhotonState* photon, InputInfo* input, uint64* trajectory)
{
    double3 previousPhotonPosition = photon->position;
    photon->time += photon->step * input->layerInfo[photon->layerId].refractiveIndex;
    MovePhoton(photon);
    UpdatePhotonTrajectory(photon, input, trajectory, previousPhotonPosition);
}

void MovePhotonOnMinDistance(PhotonState* photon)
{
    photon->position.x += MIN_DISTANCE * photon->direction.x;
    photon->position.y += MIN_DISTANCE * photon->direction.y;
    photon->position.z += MIN_DISTANCE * photon->direction.z;
}

void MovePhotonOnMinDistanceAndUpdateTrajectory(PhotonState* photon, InputInfo* input, uint64* trajectory)
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

void CrossBoundary(PhotonState* photon, InputInfo* input, OutputInfo* output, 
    IntersectionInfo* intersection, MCG59* randomGenerator, uint64* trajectory)
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
            int detectorId = UpdateWeightInDetectors(photon, input, output);
            if (detectorId >= 0)
            {
                UpdateDetectorTrajectory(output, trajectory, detectorId);
                UpdateDetectorTimeScale(output, photon, trajectory, detectorId);
            }
            photon->isDead = true;
        }
        else
        {
            photon->direction = RefractVector(input->layerInfo[layerId].refractiveIndex, 
                input->layerInfo[intersectionLayerId].refractiveIndex, photon->direction,
                intersection->normal);
            photon->layerId = intersectionLayerId;
            MovePhotonOnMinDistanceAndUpdateTrajectory(photon, input, trajectory);
        }
    }
    else //reflection
    {
        photon->direction = ReflectVector(photon->direction, intersection->normal);
        MovePhotonOnMinDistanceAndUpdateTrajectory(photon, input, trajectory);
    }
}

void ComputePhotonDirection(PhotonState* photon, double anisotropy, MCG59* randomGenerator)
{
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
		photon->direction.x = sinTheta * (ux * uz * cosPsi - uy * sinPsi) / temp + ux * cosTheta;
		photon->direction.y = sinTheta * (uy * uz * cosPsi + ux * sinPsi) / temp + uy * cosTheta;
		photon->direction.z = - sinTheta * cosPsi * temp + uz * cosTheta;
	}
	else
	{
		temp = (uz >= 0.0) ? 1.0 : -1.0;
		photon->direction.x = sinTheta * cosPsi;
		photon->direction.y = sinTheta * sinPsi;
		photon->direction.z = temp * cosTheta;
	}
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

int UpdateWeightInDetectors(PhotonState* photon, InputInfo* input, OutputInfo* output)
{
    for (int i = 0; i < input->numberOfDetectors; ++i)
    {
        double3 halfLength = {input->detector[i].length.x / 2.0, input->detector[i].length.y / 2.0,
            input->detector[i].length.z / 2.0};
        bool isPhotonInDetector = (photon->position.x >= (input->detector[i].center.x - halfLength.x)) &&
            (photon->position.x <= (input->detector[i].center.x + halfLength.x)) &&
            (photon->position.y >= (input->detector[i].center.y - halfLength.y)) &&
            (photon->position.y <= (input->detector[i].center.y + halfLength.y)) &&
            (photon->position.z >= (input->detector[i].center.z - halfLength.z)) &&
            (photon->position.z <= (input->detector[i].center.z + halfLength.z));
        if (isPhotonInDetector)
        {
            output->weightInDetector[i] += photon->weight;   
            return i;
        }
    }

    return -1;
}

void UpdatePhotonTrajectory(PhotonState* photon, InputInfo* input, uint64* trajectory, 
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
        int areaIndex = GetAreaIndex(intersectionPoint, input->area);
        if (areaIndex < 0)
        {
            break;
        }

        ++(trajectory[areaIndex]);
        
        plane += step;
    }
}

void UpdateDetectorTrajectory(OutputInfo* output, uint64* trajectory, int detectorId)
{
    uint64* detectorTrajectory = output->detectorTrajectory[detectorId].trajectory;
    int trajectorySize = output->detectorTrajectory[detectorId].trajectorySize;
    for (int i = 0; i < trajectorySize; ++i)
    {
        detectorTrajectory[i] += trajectory[i];
    }
    ++(output->detectorTrajectory[detectorId].numberOfPhotons);
}

void UpdateDetectorTimeScale(OutputInfo* output, PhotonState* photon, uint64* trajectory, int detectorId)
{
    int timeScaleSize = output->detectorTrajectory[detectorId].timeScaleSize;
    TimeInfo* timeScale = output->detectorTrajectory[detectorId].timeScale;
    for (int i = 0; i < timeScaleSize - 1; ++i)
    {
        if ((photon->time >= timeScale[i].timeStart) && (photon->time < timeScale[i].timeFinish))
        {
            ++(timeScale[i].numberOfPhotons);
            int trajectorySize = output->detectorTrajectory[detectorId].timeScale[i].trajectorySize;
            uint64* timeTrajectory = output->detectorTrajectory[detectorId].timeScale[i].trajectory;
            for (int j = 0; j < trajectorySize; ++j)
            {
                timeTrajectory[j] += trajectory[j];
            }
            return;
        }
    }
    if (photon->time >= timeScale[timeScaleSize - 1].timeStart)
    {
        int trajectorySize = output->detectorTrajectory[detectorId].timeScale[timeScaleSize - 1].trajectorySize;
        uint64* timeTrajectory = output->detectorTrajectory[detectorId].timeScale[timeScaleSize - 1].trajectory;
        for (int j = 0; j < trajectorySize; ++j)
        {
            timeTrajectory[j] += trajectory[j];
        }
        ++(timeScale[timeScaleSize - 1].numberOfPhotons);
    }
}
