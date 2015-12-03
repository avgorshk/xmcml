#include <string.h>

#include "../tinyxml/inc/tinyxml.h"
#include "reader.h"
#include "sections.h"
#include <map>

#include "parser.h"

#define DEFAULT_INTEGRAL_PRECISION 128
#define DEFAULT_TABLE_PRECISION    262144
#define DEFAULT_ATTRACTIVE_FACTOR  1.0
#define Pi 3.141592

#ifdef _WIN32
	#define atoll(S) _atoi64(S)
#endif

bool ParseInputFile(char* fileName, InputInfo* input)
{
	input->standardDeviation = 1;

    input->timeScaleSize = 1;
    input->timeStart = 0.0;
    input->timeFinish = 0.0;

	input->startPosition.x = 0.0;
	input->startPosition.y = 0.0;
	input->startPosition.z = 0.0;

	input->startDirectionInfo.startDirectionMode = 0;
	input->startDirectionInfo.startDirection.x = 0.0;
	input->startDirectionInfo.startDirection.x = 0.0;
	input->startDirectionInfo.startDirection.z = 1.0;
	input->startDirectionInfo.standardDeviation = 0.0;
	input->startDirectionInfo.distance = 1.0;

    input->useBiasing = 0;
    input->weightIntegralPrecision = DEFAULT_INTEGRAL_PRECISION;
    input->weightTablePrecision = DEFAULT_TABLE_PRECISION;
    input->attractiveFactor = DEFAULT_ATTRACTIVE_FACTOR;
	input->weightTable = NULL;
	input->numberOfWeightTables = 0;

    input->targetPoint.x = 0.0;
    input->targetPoint.y = 0.0;
    input->targetPoint.z = 0.0;

	input->numberOfCubeDetectors = 0;
	input->cubeDetector = NULL;
	input->numberOfRingDetectors = 0;
	input->ringDetector = NULL;

	memset(input->targetRangeLayers, 0, MAX_LAYERS*sizeof(unsigned char));

    TiXmlDocument document(fileName);
    bool isLoadOk = document.LoadFile();
    if (!isLoadOk)
    {
        return false;
    }
    
    TiXmlElement* root = document.FirstChildElement();
    TiXmlElement* node = root->FirstChildElement();
	std::multimap <int, CubeDetector> CubeDetectorsMap;
	std::multimap <int, RingDetector> RingDetectorsMap;
    for (node; node; node = node->NextSiblingElement())
    {
		if (strcmp(node->Value(), "StandardDeviation") == 0)
        {
			input->standardDeviation = atof(node->GetText());
		}
        if (strcmp(node->Value(), "NumberOfPhotons") == 0)
        {
            input->numberOfPhotons = atoll(node->GetText());
        }
        else if (strcmp(node->Value(), "MinWeight") == 0)
        {
            input->minWeight = atof(node->GetText());
        }
        else if (strcmp(node->Value(), "TimeStart") == 0)
        {
            input->timeStart = atof(node->GetText());
        }
        else if (strcmp(node->Value(), "TimeFinish") == 0)
        {
            input->timeFinish = atof(node->GetText());
        }
        else if (strcmp(node->Value(), "TimeScaleSize") == 0)
        {
            input->timeScaleSize = atoi(node->GetText());
        }
		else if (strcmp(node->Value(), "TargetRange") == 0)
        {
			TiXmlElement* targetRangeChild = node->FirstChildElement();
			for (targetRangeChild; targetRangeChild; targetRangeChild = targetRangeChild->NextSiblingElement())
			{
				if (strcmp(targetRangeChild->Value(), "Layer") == 0)
				{
					int layer = atoi(targetRangeChild->GetText());
					if (layer >= 0 && layer < MAX_LAYERS)
					{
						input->targetRangeLayers[layer] = 1;
					}
				}
			}
        }
		else if (strcmp(node->Value(), "StartPosition") == 0)
		{
			TiXmlElement* startPositionChild = node->FirstChildElement();
			for (startPositionChild; startPositionChild; startPositionChild = startPositionChild->NextSiblingElement())
			{
				if (strcmp(startPositionChild->Value(), "X") == 0)
                {
					input->startPosition.x = atof(startPositionChild->GetText());
                }
                else if (strcmp(startPositionChild->Value(), "Y") == 0)
                {
                    input->startPosition.y = atof(startPositionChild->GetText());
                }
                else if (strcmp(startPositionChild->Value(), "Z") == 0)
                {
                    input->startPosition.z = atof(startPositionChild->GetText());
                }
			}
		}
		else if (strcmp(node->Value(), "StartDirectionBlock") == 0)
		{
			TiXmlElement* startDirectionBlockChild = node->FirstChildElement();
			for (startDirectionBlockChild; startDirectionBlockChild; startDirectionBlockChild = startDirectionBlockChild->NextSiblingElement())
			{
				if (strcmp(startDirectionBlockChild->Value(), "StartDirection") == 0)
				{
					TiXmlElement* startDirectionChild = node->FirstChildElement();
					for (startDirectionChild; startDirectionChild; startDirectionChild = startDirectionChild->NextSiblingElement())
					{
						if (strcmp(startDirectionChild->Value(), "X") == 0)
						{
							input->startDirectionInfo.startDirection.x = atof(startDirectionChild->GetText());
						}
						else if (strcmp(startDirectionChild->Value(), "Y") == 0)
						{
							input->startDirectionInfo.startDirection.y = atof(startDirectionChild->GetText());
						}
						else if (strcmp(startDirectionChild->Value(), "Z") == 0)
						{
							input->startDirectionInfo.startDirection.z = atof(startDirectionChild->GetText());
						}
					}
				}
				else if (strcmp(startDirectionBlockChild->Value(), "StandardDeviation") == 0)
				{
					input->startDirectionInfo.standardDeviation = atof(startDirectionBlockChild->GetText());
					input->startDirectionInfo.startDirectionMode = 1;
				}
				else if (strcmp(startDirectionBlockChild->Value(), "Distance") == 0)
				{
					input->startDirectionInfo.distance = atof(startDirectionBlockChild->GetText());
					input->startDirectionInfo.startDirectionMode = 1;
				}
			}
		}

        else if (strcmp(node->Value(), "Area") == 0)
        {
            input->area = new Area;
            TiXmlElement* areaChild = node->FirstChildElement();
            for (areaChild; areaChild; areaChild = areaChild->NextSiblingElement())
            {
                if (strcmp(areaChild->Value(), "Corner") == 0)
                {
                    TiXmlElement* cornerChild = areaChild->FirstChildElement();
                    for (cornerChild; cornerChild; cornerChild = cornerChild->NextSiblingElement())
                    {
                        if (strcmp(cornerChild->Value(), "X") == 0)
                        {
                            input->area->corner.x = atof(cornerChild->GetText());
                        }
                        else if (strcmp(cornerChild->Value(), "Y") == 0)
                        {
                            input->area->corner.y = atof(cornerChild->GetText());
                        }
                        else if (strcmp(cornerChild->Value(), "Z") == 0)
                        {
                            input->area->corner.z = atof(cornerChild->GetText());
                        }
                    }
                }
                else if (strcmp(areaChild->Value(), "Length") == 0)
                {
                    TiXmlElement* lengthChild = areaChild->FirstChildElement();
                    for (lengthChild; lengthChild; lengthChild = lengthChild->NextSiblingElement())
                    {
                        if (strcmp(lengthChild->Value(), "X") == 0)
                        {
                            input->area->length.x = atof(lengthChild->GetText());
                        }
                        else if (strcmp(lengthChild->Value(), "Y") == 0)
                        {
                            input->area->length.y = atof(lengthChild->GetText());
                        }
                        else if (strcmp(lengthChild->Value(), "Z") == 0)
                        {
                            input->area->length.z = atof(lengthChild->GetText());
                        }
                    }
                }
                else if (strcmp(areaChild->Value(), "PartitionNumber") == 0)
                {
                    TiXmlElement* partitionNumberChild = areaChild->FirstChildElement();
                    for (partitionNumberChild; partitionNumberChild; partitionNumberChild = partitionNumberChild->NextSiblingElement())
                    {
                        if (strcmp(partitionNumberChild->Value(), "X") == 0)
                        {
                            input->area->partitionNumber.x = atoi(partitionNumberChild->GetText());
                        }
                        else if (strcmp(partitionNumberChild->Value(), "Y") == 0)
                        {
                            input->area->partitionNumber.y = atoi(partitionNumberChild->GetText());
                        }
                        else if (strcmp(partitionNumberChild->Value(), "Z") == 0)
                        {
                            input->area->partitionNumber.z = atoi(partitionNumberChild->GetText());
                        }
                    }
                }
            }
        }
        else if (strcmp(node->Value(), "Biasing") == 0)
        {
            input->useBiasing = 1;
            TiXmlElement* biasingChild = node->FirstChildElement();
            for (biasingChild; biasingChild; biasingChild = biasingChild->NextSiblingElement())
            {           
                if (strcmp(biasingChild->Value(), "WeightIntegral") == 0)
                {
                    TiXmlElement* weightIntegralChild = biasingChild->FirstChildElement();
                    for (weightIntegralChild; weightIntegralChild; weightIntegralChild = weightIntegralChild->NextSiblingElement())
                    {
                        if (strcmp(weightIntegralChild->Value(), "IntegralPrecision") == 0)
                        {
                            input->weightIntegralPrecision = atoi(weightIntegralChild->GetText());
                        }
                        if (strcmp(weightIntegralChild->Value(), "TablePrecision") == 0)
                        {
                            input->weightTablePrecision = atoi(weightIntegralChild->GetText());
                        }
                        if (strcmp(weightIntegralChild->Value(), "AttractiveFactor") == 0)
                        {
                            input->attractiveFactor = atof(weightIntegralChild->GetText());
                        }
                    }
                }
                else if (strcmp(biasingChild->Value(), "TargetPoint") == 0)
                {
                    TiXmlElement* targetPointChild = biasingChild->FirstChildElement();
                    for (targetPointChild; targetPointChild; targetPointChild = targetPointChild->NextSiblingElement())
                    {
                        if (strcmp(targetPointChild->Value(), "X") == 0)
                        {
                            input->targetPoint.x = atoi(targetPointChild->GetText());
                        }
                        else if (strcmp(targetPointChild->Value(), "Y") == 0)
                        {
                            input->targetPoint.y = atoi(targetPointChild->GetText());
                        }
                        else if (strcmp(targetPointChild->Value(), "Z") == 0)
                        {
                            input->targetPoint.z = atoi(targetPointChild->GetText());
                        }
                    }
                }
            }
        }
        else if (strcmp(node->Value(), "NumberOfLayers") == 0)
        {
            input->numberOfLayers = atoi(node->GetText());
        }
        else if (strcmp(node->Value(), "Layers") == 0)
        {
            input->layerInfo = new LayerInfo[input->numberOfLayers];
            TiXmlElement* layersChild = node->FirstChildElement();
            int i = 0;
            for (layersChild; layersChild; layersChild = layersChild->NextSiblingElement(), ++i)
            {
				if (i >= input->numberOfLayers)
				{
					fprintf(stderr, "ERROR: number of layers invalid\n");
					exit(1);
				}
                if (strcmp(layersChild->Value(), "Layer") == 0)
                {
                    TiXmlElement* layerChild = layersChild->FirstChildElement();
                    for (layerChild; layerChild; layerChild = layerChild->NextSiblingElement())
                    {
                        if (strcmp(layerChild->Value(), "RefractiveIndex") == 0)
                        {
                            input->layerInfo[i].refractiveIndex = atof(layerChild->GetText());
                        }
                        else if (strcmp(layerChild->Value(), "AbsorptionCoefficient") == 0)
                        {
                            input->layerInfo[i].absorptionCoefficient = atof(layerChild->GetText());
                        }
                        else if (strcmp(layerChild->Value(), "ScatteringCoefficient") == 0)
                        {
                            input->layerInfo[i].scatteringCoefficient = atof(layerChild->GetText());
                        }
                        else if (strcmp(layerChild->Value(), "Anisotropy") == 0)
                        {
                            input->layerInfo[i].anisotropy = atof(layerChild->GetText());
                        }
                        else if (strcmp(layerChild->Value(), "NumberOfSurfaces") == 0)
                        {
                            input->layerInfo[i].numberOfSurfaces = atoi(layerChild->GetText());
                        }
                        else if (strcmp(layerChild->Value(), "SurfaceIds") == 0)
                        {
                            input->layerInfo[i].surfaceId = new int[input->layerInfo[i].numberOfSurfaces];
                            TiXmlElement* surfaceIdsChild = layerChild->FirstChildElement();
                            int j = 0;
                            for (surfaceIdsChild; surfaceIdsChild; surfaceIdsChild = surfaceIdsChild->NextSiblingElement(), ++j)
                            {
                                if (strcmp(surfaceIdsChild->Value(), "Id") == 0)
                                {
                                    input->layerInfo[i].surfaceId[j] = atoi(surfaceIdsChild->GetText());
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (strcmp(node->Value(), "NumberOfCubeDetectors") == 0)
        {
            input->numberOfCubeDetectors = atoi(node->GetText());
        }
        else if (strcmp(node->Value(), "CubeDetectors") == 0)
        {
			CubeDetector cubeDetector;
            input->cubeDetector = new CubeDetector[input->numberOfCubeDetectors];
            TiXmlElement* detectorsChild = node->FirstChildElement();
            int i = 0;
            for (detectorsChild; detectorsChild; detectorsChild = detectorsChild->NextSiblingElement(), ++i)
            {
				if (i >= input->numberOfCubeDetectors)
				{
					fprintf(stderr, "ERROR: number of cube detectors invalid\n");
					exit(1);
				}
                if (strcmp(detectorsChild->Value(), "CubeDetector") == 0)
                {
					int numberOfFilterLayers = MAX_LAYERS;
					for(int j = 0; j < MAX_LAYERS; j++)
					{
						cubeDetector.filterLayers[j] = true;
					}
					cubeDetector.permissibleAngle = 0.0;
                    TiXmlElement* detectorChild = detectorsChild->FirstChildElement();
                    for (detectorChild; detectorChild; detectorChild = detectorChild->NextSiblingElement())
                    {
						if (strcmp(detectorChild->Value(), "PermissibleAngle") == 0)
						{
							cubeDetector.permissibleAngle = sin(atof(detectorChild->GetText()) * Pi / 180.0);
							if(cubeDetector.permissibleAngle < 0)
							{
								fprintf(stderr, "ERROR: permissible angle invalid\n");
								exit(1);
							}
						}
                        if (strcmp(detectorChild->Value(), "Center") == 0)
                        {
                            TiXmlElement* centerChild = detectorChild->FirstChildElement();
                            for (centerChild; centerChild; centerChild = centerChild->NextSiblingElement())
                            {
                                if (strcmp(centerChild->Value(), "X") == 0)
                                {
                                    cubeDetector.center.x = atof(centerChild->GetText());
                                }
                                else if (strcmp(centerChild->Value(), "Y") == 0)
                                {
                                    cubeDetector.center.y = atof(centerChild->GetText());
                                }
                                else if (strcmp(centerChild->Value(), "Z") == 0)
                                {
                                    cubeDetector.center.z = atof(centerChild->GetText());
                                }
                            }
                        }
                        else if (strcmp(detectorChild->Value(), "Length") == 0)
                        {
                            TiXmlElement* lengthChild = detectorChild->FirstChildElement();
                            for (lengthChild; lengthChild; lengthChild = lengthChild->NextSiblingElement())
                            {
                                if (strcmp(lengthChild->Value(), "X") == 0)
                                {
                                    cubeDetector.length.x = atof(lengthChild->GetText());
                                }
                                else if (strcmp(lengthChild->Value(), "Y") == 0)
                                {
                                    cubeDetector.length.y = atof(lengthChild->GetText());
                                }
                                else if (strcmp(lengthChild->Value(), "Z") == 0)
                                {
                                    cubeDetector.length.z = atof(lengthChild->GetText());
                                }
                            }
                        }
						else if (strcmp(detectorChild->Value(), "Filter") == 0)
						{
							for(int j = 0; j < MAX_LAYERS; j++)
							{
								cubeDetector.filterLayers[j] = false;
							}
							numberOfFilterLayers = 0;
							TiXmlElement* filterChild = detectorChild->FirstChildElement();
							int j = 0, indLayer;
							for (filterChild; filterChild; filterChild = filterChild->NextSiblingElement(), ++j)
							{
								if (strcmp(filterChild->Value(), "Layer") == 0)
								{
									indLayer = atoi(filterChild->GetText());
									if((indLayer < 0) || (indLayer >= MAX_LAYERS)) 
									{
										fprintf(stderr, "ERROR: FilterLayers invalid\n");
										exit(1);
									}
									cubeDetector.filterLayers[indLayer] = true;
									numberOfFilterLayers++;
								}
							}
						}
                    }
					CubeDetectorsMap.insert(std::pair<int, CubeDetector>(numberOfFilterLayers, cubeDetector));
                }
            }
        }
		else if (strcmp(node->Value(), "NumberOfRingDetectors") == 0)
        {
            input->numberOfRingDetectors = atoi(node->GetText());
        }
        else if (strcmp(node->Value(), "RingDetectors") == 0)
        {
			RingDetector ringDetector;
			input->ringDetector = new RingDetector[input->numberOfRingDetectors];
            TiXmlElement* detectorsChild = node->FirstChildElement();
            int i = 0;
            for (detectorsChild; detectorsChild; detectorsChild = detectorsChild->NextSiblingElement(), ++i)
            {
				if (i >= input->numberOfRingDetectors)
				{
					fprintf(stderr, "ERROR: number of ring detectors invalid\n");
					exit(1);
				}
                if (strcmp(detectorsChild->Value(), "RingDetector") == 0)
                {
					for(int j = 0; j < MAX_LAYERS; j++)
					{
						ringDetector.filterLayers[j] = true;
					}
					ringDetector.permissibleAngle = 0.0;
					int numberOfFilterLayers = MAX_LAYERS;
                    TiXmlElement* detectorChild = detectorsChild->FirstChildElement();
                    for (detectorChild; detectorChild; detectorChild = detectorChild->NextSiblingElement())
                    {
						if (strcmp(detectorChild->Value(), "PermissibleAngle") == 0)
						{
							ringDetector.permissibleAngle = sin(atof(detectorChild->GetText()) * Pi / 180.0);
							if(ringDetector.permissibleAngle < 0.0)
							{
								fprintf(stderr, "ERROR: permissible angle invalid\n");
								exit(1);
							}
						}
                        if (strcmp(detectorChild->Value(), "Center") == 0)
                        {
                            TiXmlElement* centerChild = detectorChild->FirstChildElement();
                            for (centerChild; centerChild; centerChild = centerChild->NextSiblingElement())
                            {
                                if (strcmp(centerChild->Value(), "X") == 0)
                                {
                                    ringDetector.center.x = atof(centerChild->GetText());
                                }
                                else if (strcmp(centerChild->Value(), "Y") == 0)
                                {
                                    ringDetector.center.y = atof(centerChild->GetText());
                                }
                                else if (strcmp(centerChild->Value(), "Z") == 0)
                                {
                                    ringDetector.center.z = atof(centerChild->GetText());
                                }
                            }
                        }
                        else if (strcmp(detectorChild->Value(), "SmallRadius") == 0)
                        {
							ringDetector.smallRadius = atof(detectorChild->GetText());
                        }
						else if (strcmp(detectorChild->Value(), "BigRadius") == 0)
                        {
							ringDetector.bigRadius = atof(detectorChild->GetText());
                        }
						
						else if (strcmp(detectorChild->Value(), "Filter") == 0)
						{
							for(int j = 0; j < MAX_LAYERS; j++)
							{
								ringDetector.filterLayers[j] = false;
							}
							numberOfFilterLayers = 0;
							TiXmlElement* filterChild = detectorChild->FirstChildElement();
							int j = 0;
							for (filterChild; filterChild; filterChild = filterChild->NextSiblingElement(), ++j)
							{
								if (j >= MAX_LAYERS)
								{
									fprintf(stderr, "ERROR: number of FilterLayers invalid\n");
									exit(1);
								}
								if (strcmp(filterChild->Value(), "Layer") == 0)
								{
									ringDetector.filterLayers[atoi(filterChild->GetText())] = true;
									numberOfFilterLayers++;
								}
							}
						}
                    }
					RingDetectorsMap.insert(std::pair<int, RingDetector>(numberOfFilterLayers, ringDetector));
                }
            }
        }
        else if (strcmp(node->Value(), "NumberOfSurfaces") == 0)
        {
            input->numberOfSurfaces = atoi(node->GetText());
        }
    }
	
	int i = 0;
	auto cit = CubeDetectorsMap.begin();
	for (cit = CubeDetectorsMap.begin(); cit != CubeDetectorsMap.end(); ++cit)
	{
		input->cubeDetector[i] = cit->second;
		i++;
	}
	i = 0;
	auto rit = RingDetectorsMap.begin();
	for (rit = RingDetectorsMap.begin(); rit != RingDetectorsMap.end(); ++rit)
	{
		input->ringDetector[i] = rit->second;
		i++;
	}

    return true;
}

static bool ReadDouble3(FILE* file, double3* vector)
{
    unsigned long long int read_items;

    read_items = fread(&(vector->x), sizeof(double), 1, file);
    if (read_items < 1)
        return false;

    read_items = fread(&(vector->y), sizeof(double), 1, file);
    if (read_items < 1)
        return false;

    read_items = fread(&(vector->z), sizeof(double), 1, file);
    if (read_items < 1)
        return false;

    return true;
}

static bool ReadInt3(FILE* file, int3* vector)
{
    unsigned long long int read_items;

    read_items = fread(&(vector->x), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    read_items = fread(&(vector->y), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    read_items = fread(&(vector->z), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    return true;
}

static bool ReadSurface(FILE* file, Surface* surface)
{
    unsigned long long int read_items;

    read_items = fread(&(surface->numberOfVertices), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    surface->vertices = new double3[surface->numberOfVertices];
    for (int i = 0; i < surface->numberOfVertices; ++i)
    {
        if (!ReadDouble3(file, &(surface->vertices[i])))
            return false;
    }

    read_items = fread(&(surface->numberOfTriangles), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    surface->triangles = new int3[surface->numberOfTriangles];
    for (int i = 0; i < surface->numberOfTriangles; ++i)
    {
        if (!ReadInt3(file, &(surface->triangles[i])))
            return false;
    }

    return true;
}

bool ParseSurfaceFile(char* fileName, InputInfo* input)
{
    FILE* file = fopen(fileName, "rb");
    if (file == NULL)
        return false;
    
    unsigned long long int read_items;

    unsigned int section_id;
    read_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (read_items < 1 || section_id != MCML_SECTION_SURFACES)
        return false;

    int numberOfSurfaces;
    read_items = fread(&numberOfSurfaces, sizeof(int), 1, file);
    if (read_items < 1 || numberOfSurfaces != input->numberOfSurfaces)
        return false;

    input->surface = new Surface[input->numberOfSurfaces];
    for (int i = 0; i < numberOfSurfaces; ++i)
    {
        if (!ReadSurface(file, &(input->surface[i])))
            return false;
    }

    fclose(file);

    return true;
}

bool ParseBackupFile(char* fileName, OutputInfo* output, MCG59* randomGenerator, 
    int numThreadsPerProcess, int numProcesses)
{
    int error = 0;
    
    int numThreadsPerProcessInBackup, numProcessesInBackup;
    error = ReadThreadsFromBackupFile(fileName, &numThreadsPerProcessInBackup, &numProcessesInBackup);
    if (error || numProcesses != numProcessesInBackup || numThreadsPerProcess != numThreadsPerProcessInBackup)
        return false;

    error = ReadRandomGeneratorFromBackupFile(fileName, randomGenerator);
    if (error) return false;

    error = ReadOutputFormBackupFile(fileName, output);
    if (error) return false;

    return true;
}
