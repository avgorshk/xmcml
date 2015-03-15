#include <string.h>

#include "..\tinyxml\inc\tinyxml.h"
#include "reader.h"
#include "sections.h"

#include "parser.h"

bool ParseInputFile(char* fileName, InputInfo* input)
{
    input->timeScaleSize = 1;
    input->timeStart = 0.0;
    input->timeFinish = 0.0;

    TiXmlDocument document(fileName);
    bool isLoadOk = document.LoadFile();
    if (!isLoadOk)
    {
        return false;
    }
    
    TiXmlElement* root = document.FirstChildElement();
    TiXmlElement* node = root->FirstChildElement();
    for (node; node; node = node->NextSiblingElement())
    {
        if (strcmp(node->Value(), "NumberOfPhotons") == 0)
        {
            input->numberOfPhotons = _atoi64(node->GetText());
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
        else if (strcmp(node->Value(), "NumberOfDetectors") == 0)
        {
            input->numberOfDetectors = atoi(node->GetText());
        }
        else if (strcmp(node->Value(), "Detectors") == 0)
        {
            input->detector = new Detector[input->numberOfDetectors];
            TiXmlElement* detectorsChild = node->FirstChildElement();
            int i = 0;
            for (detectorsChild; detectorsChild; detectorsChild = detectorsChild->NextSiblingElement(), ++i)
            {
                if (strcmp(detectorsChild->Value(), "Detector") == 0)
                {
                    TiXmlElement* detectorChild = detectorsChild->FirstChildElement();
                    for (detectorChild; detectorChild; detectorChild = detectorChild->NextSiblingElement())
                    {
                        if (strcmp(detectorChild->Value(), "Center") == 0)
                        {
                            TiXmlElement* centerChild = detectorChild->FirstChildElement();
                            for (centerChild; centerChild; centerChild = centerChild->NextSiblingElement())
                            {
                                if (strcmp(centerChild->Value(), "X") == 0)
                                {
                                    input->detector[i].center.x = atof(centerChild->GetText());
                                }
                                else if (strcmp(centerChild->Value(), "Y") == 0)
                                {
                                    input->detector[i].center.y = atof(centerChild->GetText());
                                }
                                else if (strcmp(centerChild->Value(), "Z") == 0)
                                {
                                    input->detector[i].center.z = atof(centerChild->GetText());
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
                                    input->detector[i].length.x = atof(lengthChild->GetText());
                                }
                                else if (strcmp(lengthChild->Value(), "Y") == 0)
                                {
                                    input->detector[i].length.y = atof(lengthChild->GetText());
                                }
                                else if (strcmp(lengthChild->Value(), "Z") == 0)
                                {
                                    input->detector[i].length.z = atof(lengthChild->GetText());
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (strcmp(node->Value(), "NumberOfSurfaces") == 0)
        {
            input->numberOfSurfaces = atoi(node->GetText());
        }
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
    FILE* file = NULL;

    fopen_s(&file, fileName, "rb");
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
