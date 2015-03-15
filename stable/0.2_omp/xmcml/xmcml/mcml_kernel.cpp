#include <math.h>
#include "mcml_kernel.h"
#include "mcml_math.h"

#define PI             3.14159265358979323846
#define COSINE_OF_ZERO (1.0 - 1.0E-12)
#define COSINE_OF_90D  1.0E-12
#define MIN_DISTANCE   1.0E-10

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
	
	photon->position.x = 0.0;
	photon->position.y = 0.0;
	photon->position.z = MIN_DISTANCE;

	photon->direction.x = 0.0;
	photon->direction.y = 0.0;
	photon->direction.z = 1.0;
}

void ComputePhoton(double specularReflectance, InputInfo* input, OutputInfo* output, MCG59* randomGenerator)
{
    PhotonState photon;
    
    LaunchPhoton(&photon, specularReflectance);
    while (!photon.isDead)
    {
        int layerId = photon.layerId;
        bool isPhotonInGlass = (input->layerInfo[layerId].absorptionCoefficient == 0.0) &&
            (input->layerInfo[layerId].scatteringCoefficient == 0.0);
        
        if (isPhotonInGlass)
        {
            IntersectionInfo intersection = GetIntersectionInfo(&photon, input);
            if (intersection.isFindIntersection)
            {
                photon.step = intersection.distance;
                MovePhoton(&photon);
                
                int areaIndex = GetAreaIndex(&photon, input->area);
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

            IntersectionInfo intersection = GetIntersectionInfo(&photon, input);
            bool isHitBoundary = intersection.isFindIntersection && (photon.step >= intersection.distance);

            if (isHitBoundary)
            {
                photon.step = intersection.distance;   
                MovePhoton(&photon);

                int areaIndex = GetAreaIndex(&photon, input->area);
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
                MovePhoton(&photon);

                int areaIndex = GetAreaIndex(&photon, input->area);
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

IntersectionInfo GetIntersectionInfo(PhotonState* photon, InputInfo* input)
{
    int layerId = photon->layerId;
    int numberOfSurfaces = input->layerInfo[layerId].numberOfSurfaces;
    Surface* surfaces = new Surface[numberOfSurfaces];

    for (int i = 0; i < numberOfSurfaces; ++i)
    {
        surfaces[i] = input->surface[input->layerInfo[layerId].surfaceId[i]];
    }

    IntersectionInfo intersection = ComputeIntersection(photon->position, photon->direction,
        surfaces, numberOfSurfaces);

    if (intersection.isFindIntersection)
    {
        intersection.surfaceId = input->layerInfo[layerId].surfaceId[intersection.surfaceId];
    }

    delete[] surfaces;

    return intersection;
}

void MovePhoton(PhotonState* photon)
{
    photon->position.x += photon->step * photon->direction.x;
    photon->position.y += photon->step * photon->direction.y;
    photon->position.z += photon->step * photon->direction.z;
}

void MovePhotonOnMinDistance(PhotonState* photon)
{
    photon->position.x += MIN_DISTANCE * photon->direction.x;
    photon->position.y += MIN_DISTANCE * photon->direction.y;
    photon->position.z += MIN_DISTANCE * photon->direction.z;
}

int GetAreaIndex(PhotonState* photon, Area* area)
{	
    int indexX = (int)(area->partitionNumber.x * (photon->position.x - area->corner.x) / area->length.x);
    int indexY = (int)(area->partitionNumber.y * (photon->position.y - area->corner.y) / area->length.y);
    int indexZ = (int)(area->partitionNumber.z * (photon->position.z - area->corner.z) / area->length.z);
	
    bool isNotPhotonInArea = (indexX < 0 || indexX >= area->partitionNumber.x) || 
        (indexY < 0 || indexY >= area->partitionNumber.y) ||
        (indexZ < 0 || indexZ >= area->partitionNumber.z);
    
    if (isNotPhotonInArea)
	{
		return -1;
	}
	
	return indexX * area->partitionNumber.y * area->partitionNumber.z + indexY * area->partitionNumber.z + indexZ;
}

void CrossBoundary(PhotonState* photon, InputInfo* input, OutputInfo* output, 
    IntersectionInfo* intersection, MCG59* randomGenerator)
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
            UpdateWeightInDetectors(photon, input, output);
            photon->isDead = true;
        }
        else
        {
            photon->direction = RefractVector(input->layerInfo[layerId].refractiveIndex, 
                input->layerInfo[intersectionLayerId].refractiveIndex, photon->direction,
                intersection->normal);
            photon->layerId = intersectionLayerId;
            MovePhotonOnMinDistance(photon);
        }
    }
    else //reflection
    {
        photon->direction = ReflectVector(photon->direction, intersection->normal);
        MovePhotonOnMinDistance(photon);
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

    #pragma omp atomic
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

void UpdateWeightInDetectors(PhotonState* photon, InputInfo* input, OutputInfo* output)
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
            #pragma omp atomic
            output->weigthInDetector[i] += photon->weight;
            
            return;
        }
    }
}