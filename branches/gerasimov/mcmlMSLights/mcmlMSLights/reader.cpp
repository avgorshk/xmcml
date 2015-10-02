#include <stdio.h>
#include <stdlib.h>

#include "sections.h"

#include "reader.h"

static int ReadSectionNumberOfPhotons(FILE* file, uint64* numberOfPhotons)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if( section_id != MCML_SECTION_NUMBER_OF_PHOTONS)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

    reading_items = fread(numberOfPhotons, sizeof(uint64), 1, file);
    if (reading_items < 1)
        return -1;

    return 0;
}

static int ReadSectionArea(FILE* file, InputInfo* input)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_AREA)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);
	
    double double_buffer[6];

    reading_items = fread(double_buffer, sizeof(double), 6, file);
    if (reading_items < 6)
        return -1;

	Area* area = new Area;

	area->corner.x = double_buffer[0];
	area->corner.y = double_buffer[1];
	area->corner.z = double_buffer[2];
	area->length.x = double_buffer[3];
	area->length.y = double_buffer[4];
	area->length.z = double_buffer[5];

    int int_buffer[3];

    reading_items = fread(int_buffer, sizeof(int), 3, file);
    if (reading_items < 3)
        return -1;

	area->partitionNumber.x = int_buffer[0];
    area->partitionNumber.y = int_buffer[1];
    area->partitionNumber.z = int_buffer[2];

	input->area = area;

    return 0;
}

static int ReadSectionCubeDetectors(FILE* file, InputInfo* input)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_CUBE_DETECTORS)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int numberOfDetectors;

    reading_items = fread(&numberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	CubeDetector* detector = new CubeDetector[numberOfDetectors];

    double buffer[6];
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        reading_items = fread(buffer, sizeof(double), 6, file);
        if (reading_items < 6)
            return -1;

		detector[i].center.x = buffer[0];
        detector[i].center.y = buffer[1];
        detector[i].center.z = buffer[2];
        detector[i].length.x = buffer[3];
        detector[i].length.y = buffer[4];
        detector[i].length.z = buffer[5];
    }

	input->numberOfCubeDetectors = numberOfDetectors;
	input->cubeDetector = detector;

    return 0;
}

static int ReadSectionRingDetectors(FILE* file, InputInfo* input)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_RING_DETECTORS)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int numberOfDetectors;

    reading_items = fread(&numberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	RingDetector* detector = new RingDetector[numberOfDetectors];

    double buffer[5];
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        reading_items = fread(buffer, sizeof(double), 5, file);
        if (reading_items < 5)
            return -1;

		detector[i].center.x = buffer[0];
        detector[i].center.y = buffer[1];
        detector[i].center.z = buffer[2];
		detector[i].smallRadius = buffer[3];
		detector[i].bigRadius = buffer[4];
    }

	input->numberOfRingDetectors = numberOfDetectors;
	input->ringDetector = detector;

    return 0;
}

static int ReadSectionSpecularReflectance(FILE* file, double* specularReflectance)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_SPECULAR_REFLECTANCE)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

    reading_items = fread(specularReflectance, sizeof(double), 1, file);
    if (reading_items < 1)
        return -1;

    return 0;
}

static int ReadSectionCommonTrajectories(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_COMMON_TRAJECTORIES)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int gridSize;

    reading_items = fread(&gridSize, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	double* absorption = new double[gridSize];

    reading_items = fread(absorption, sizeof(double), gridSize, file);
    if (reading_items < gridSize)
        return -1;

	output->gridSize = gridSize;
	output->absorption = absorption;

    return 0;
}

static int ReadSectionScatteringMap(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_SCATTERING_MAP)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int gridSize;

    reading_items = fread(&gridSize, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	double* scatteringMap = new double[gridSize];

    reading_items = fread(scatteringMap, sizeof(double), gridSize, file);
    if (reading_items < gridSize)
        return -1;

	output->gridSize = gridSize;
	output->scatteringMap = scatteringMap;

    return 0;
}

static int ReadSectionDetectorWeights(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_DETECTOR_WEIGHTS)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int numberOfDetectors;

    reading_items = fread(&numberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	double* detectorWeights = new double[numberOfDetectors];

    reading_items = fread(detectorWeights, sizeof(double), numberOfDetectors, file);
    if (reading_items < numberOfDetectors)
        return -1;

	output->numberOfDetectors = numberOfDetectors;
	output->weightInDetector = detectorWeights;

    return 0;
}

static int ReadSectionGridDetector(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_GRID_DETECTOR)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int gridDetectorSize;

    reading_items = fread(&gridDetectorSize, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	double* weightInGridDetector = new double[gridDetectorSize];

    reading_items = fread(weightInGridDetector, sizeof(double), gridDetectorSize, file);
    if (reading_items < gridDetectorSize)
        return -1;

	output->gridDetectorSize = gridDetectorSize;
	output->weightInGridDetector = weightInGridDetector;

    return 0;
}

static int ReadSectionDetectorTrajectories(FILE* file, DetectorTrajectory* detectorTrajectory,
	int numberOfDetectors)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;
    
	if(section_id != MCML_SECTION_DETECTOR_TRAJECTORIES)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int ReadNumberOfDetectors;

    reading_items = fread(&ReadNumberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	if(numberOfDetectors != ReadNumberOfDetectors)
		return -1;

    for (int i = 0; i < numberOfDetectors; ++i)
    {
        reading_items = fread(&(detectorTrajectory[i].numberOfPhotons), sizeof(uint64), 1, file);
        if (reading_items < 1)
            return -1;

        reading_items = fread(&(detectorTrajectory[i].trajectorySize), sizeof(int), 1, file);
        if (reading_items < 1)
            return -1;

		detectorTrajectory[i].trajectory = new uint64[detectorTrajectory[i].trajectorySize];

        reading_items = fread(detectorTrajectory[i].trajectory, sizeof(uint64), 
            detectorTrajectory[i].trajectorySize, file);
        if (reading_items < detectorTrajectory[i].trajectorySize)
            return -1;
    }

    return 0;
}

static int ReadSectionDetectorTimeScale(FILE* file, DetectorTrajectory* detectorTrajectory,
	int numberOfDetectors)
{
    unsigned long long int reading_items;

    unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_DETECTOR_TIME_SCALE)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

	int ReadNumberOfDetectors;

    reading_items = fread(&ReadNumberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	if(numberOfDetectors != ReadNumberOfDetectors)
		return -1;

    for (int i = 0; i < numberOfDetectors; ++i)
    {
        reading_items = fread(&(detectorTrajectory[i].timeScaleSize), sizeof(int), 1, file);
        if (reading_items < 1)
            return -1;

		detectorTrajectory[i].timeScale = new TimeInfo[detectorTrajectory[i].timeScaleSize];

        for (int j = 0; j < detectorTrajectory[i].timeScaleSize; ++j)
        {
            reading_items = fread(&(detectorTrajectory[i].timeScale[j].timeStart), 
                sizeof(double), 1, file);
            if (reading_items < 1)
                return -1;

            reading_items = fread(&(detectorTrajectory[i].timeScale[j].timeFinish), 
                sizeof(double), 1, file);
            if (reading_items < 1)
                return -1;

            reading_items = fread(&(detectorTrajectory[i].timeScale[j].numberOfPhotons), 
                sizeof(uint64), 1, file);
            if (reading_items < 1)
                return -1;

            reading_items = fread(&(detectorTrajectory[i].timeScale[j].weight), 
                sizeof(double), 1, file);
            if (reading_items < 1)
                return -1;
        }
    }

    return 0;
}

static int ReadSectionDetectorRanges(FILE* file, DetectorTrajectory* detectorTrajectory,
	int numberOfDetectors)
{
    unsigned long long int reading_items;

	unsigned int section_id;
    reading_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

	if(section_id != MCML_SECTION_DETECTOR_RANGES)
		return -1;

    fseek(file, sizeof(unsigned int), SEEK_CUR);

   	int ReadNumberOfDetectors;

    reading_items = fread(&ReadNumberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

	if(numberOfDetectors != ReadNumberOfDetectors)
		return -1;

	for (int i = 0; i < numberOfDetectors; ++i)
	{
		reading_items = fread(&(detectorTrajectory[i].otherRange), sizeof(double), 1, file);
		if (reading_items < 1)
			return -1;

		reading_items = fread(&(detectorTrajectory[i].targetRange), sizeof(double), 1, file);
		if (reading_items < 1)
			return -1;
	}

    return 0;
}

static int ReadSectionTrajectoriesTimeScaleRanges(FILE* file, OutputInfo* output)
{
	DetectorTrajectory* detectorTrajectory = new DetectorTrajectory[output->numberOfDetectors];

    if (ReadSectionDetectorTrajectories(file, detectorTrajectory, output->numberOfDetectors) != 0)
        return -1;
    if (ReadSectionDetectorTimeScale(file, detectorTrajectory, output->numberOfDetectors) != 0)
        return -1;
	if (ReadSectionDetectorRanges(file, detectorTrajectory, output->numberOfDetectors) != 0)
		return -1;

	output->detectorTrajectory = detectorTrajectory;
	return 0;
}

bool ReadOutputToFile(InputInfo* input, OutputInfo* output, char* fileName)
{
    FILE* file = fopen(fileName, "rb");
    if (file == NULL)
        return false;
    
    if (ReadSectionNumberOfPhotons(file, &(input->numberOfPhotons)) != 0)
        return false;
    if (ReadSectionArea(file, input) != 0)
        return false;
    if (ReadSectionCubeDetectors(file, input) != 0)
        return false;
	if (ReadSectionRingDetectors(file, input) != 0)
        return false;
    if (ReadSectionSpecularReflectance(file, &(output->specularReflectance)) != 0)
        return false;
    if (ReadSectionCommonTrajectories(file, output) != 0)
        return false;
	if (ReadSectionScatteringMap(file, output) != 0)
        return false;
    if (ReadSectionDetectorWeights(file, output) != 0)
        return false;
	if (ReadSectionGridDetector(file, output) != 0)
		return false;
	if (ReadSectionTrajectoriesTimeScaleRanges(file, output) != 0)
		return false;

    fclose(file);

    return true;
}