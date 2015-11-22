#include <stdio.h>
#include <stdlib.h>

#include "sections.h"

#include "writer.h"

static char* NumberToString(uint64 number)
{
    char* result = new char[16];

    if (number >= 1000000000)
    {
        sprintf(result, "%.2f%c", number / 1000000000.0f, 'B');
    }
    else if (number >= 1000000) 
    {
        sprintf(result, "%.2f%c", number / 1000000.0f, 'M');
    }
    else if (number >= 1000) 
    {
        sprintf(result, "%.2f%c", number / 1000.0f, 'T');
    }
    else 
    {
        sprintf(result, "%u", (uint)number);
    }

    return result;
}

static int WriteSectionNumberOfPhotons(FILE* file, uint64 numberOfPhotons)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_NUMBER_OF_PHOTONS;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(uint64);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfPhotons, sizeof(uint64), 1, file);
    if (written_items < 1)
        return -1;

    return 0;
}

static int WriteSectionArea(FILE* file, Area* area)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_AREA;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = 3 * sizeof(double) + 3 * sizeof(double) + 3 * sizeof(int);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    double double_buffer[6];
    double_buffer[0] = area->corner.x;
    double_buffer[1] = area->corner.y;
    double_buffer[2] = area->corner.z;
    double_buffer[3] = area->length.x;
    double_buffer[4] = area->length.y;
    double_buffer[5] = area->length.z;
    written_items = fwrite(double_buffer, sizeof(double), 6, file);
    if (written_items < 6)
        return -1;

    int int_buffer[3];
    int_buffer[0] = area->partitionNumber.x;
    int_buffer[1] = area->partitionNumber.y;
    int_buffer[2] = area->partitionNumber.z;
    written_items = fwrite(int_buffer, sizeof(int), 3, file);
    if (written_items < 3)
        return -1;

    return 0;
}

static int WriteSectionCubeDetectors(FILE* file, CubeDetector* detector, int numberOfDetectors)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_CUBE_DETECTORS;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int) + (6 * sizeof(double)) * numberOfDetectors;
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfDetectors, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    double buffer[6];
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        buffer[0] = detector[i].center.x;
        buffer[1] = detector[i].center.y;
        buffer[2] = detector[i].center.z;
        buffer[3] = detector[i].length.x;
        buffer[4] = detector[i].length.y;
        buffer[5] = detector[i].length.z;
        written_items = fwrite(buffer, sizeof(double), 6, file);
        if (written_items < 6)
            return -1;
    }

    return 0;
}

static int WriteSectionRingDetectors(FILE* file, RingDetector* detector, int numberOfDetectors)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_RING_DETECTORS;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int) + (5 * sizeof(double)) * numberOfDetectors;
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfDetectors, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    double buffer[5];
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        buffer[0] = detector[i].center.x;
        buffer[1] = detector[i].center.y;
        buffer[2] = detector[i].center.z;
		buffer[3] = detector[i].smallRadius;
		buffer[4] = detector[i].bigRadius;
        written_items = fwrite(buffer, sizeof(double), 5, file);
        if (written_items < 5)
            return -1;
    }

    return 0;
}

static int WriteSectionSpecularReflectance(FILE* file, double specularReflectance)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_SPECULAR_REFLECTANCE;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&specularReflectance, sizeof(double), 1, file);
    if (written_items < 1)
        return -1;

    return 0;
}

static int WriteSectionCommonTrajectories(FILE* file, double* absorption, int gridSize)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_COMMON_TRAJECTORIES;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int) + gridSize * sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&gridSize, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(absorption, sizeof(double), gridSize, file);
    if (written_items < gridSize)
        return -1;

    return 0;
}

static int WriteSectionScatteringMap(FILE* file, double* scatteringMap, int gridSize)
{
	unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_SCATTERING_MAP;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

	unsigned int section_lenght = sizeof(int) + gridSize * sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

	written_items = fwrite(&gridSize, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

	written_items = fwrite(scatteringMap, sizeof(double), gridSize, file);
    if (written_items < gridSize)
        return -1;

	return 0;
}

static int WriteSectionDepthMap(FILE* file, double* depthMap, int gridSize)
{
	unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_DEPTH_MAP;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

	unsigned int section_lenght = sizeof(int) + gridSize * sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

	written_items = fwrite(&gridSize, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

	written_items = fwrite(depthMap, sizeof(double), gridSize, file);
    if (written_items < gridSize)
        return -1;

	return 0;
}

static int WriteSectionDetectorWeights(FILE* file, double* detectorWeights, int numberOfDetectors)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_DETECTOR_WEIGHTS;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int) + numberOfDetectors * sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfDetectors, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(detectorWeights, sizeof(double), numberOfDetectors, file);
    if (written_items < numberOfDetectors)
        return -1;

    return 0;
}

static int WriteSectionGridDetector(FILE* file, double* weightInGridDetector, int gridDetectorSize)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_GRID_DETECTOR;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int) + gridDetectorSize * sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&gridDetectorSize, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(weightInGridDetector, sizeof(double), gridDetectorSize, file);
    if (written_items < gridDetectorSize)
        return -1;

    return 0;
}

static int WriteSectionDetectorTrajectories(FILE* file, DetectorTrajectory* detectorTrajectory, 
    int numberOfDetectors)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_DETECTOR_TRAJECTORIES;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;
    
    unsigned int section_lenght = sizeof(int);
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        section_lenght += sizeof(uint64) + sizeof(int) + detectorTrajectory[i].trajectorySize * sizeof(uint64);
    }

    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfDetectors, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    for (int i = 0; i < numberOfDetectors; ++i)
    {
        written_items = fwrite(&(detectorTrajectory[i].numberOfPhotons), sizeof(uint64), 1, file);
        if (written_items < 1)
            return -1;

        written_items = fwrite(&(detectorTrajectory[i].trajectorySize), sizeof(int), 1, file);
        if (written_items < 1)
            return -1;

        written_items = fwrite(detectorTrajectory[i].trajectory, sizeof(uint64), 
            detectorTrajectory[i].trajectorySize, file);
        if (written_items < detectorTrajectory[i].trajectorySize)
            return -1;
    }

    return 0;
}

static int WriteSectionDetectorTimeScale(FILE* file, DetectorTrajectory* detectorTrajectory, 
    int numberOfDetectors)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_DETECTOR_TIME_SCALE;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int);
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        section_lenght += sizeof(int) + 
            detectorTrajectory[i].timeScaleSize * (3 * sizeof(double) + sizeof(uint64));
    }
    
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfDetectors, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    for (int i = 0; i < numberOfDetectors; ++i)
    {
        written_items = fwrite(&(detectorTrajectory[i].timeScaleSize), sizeof(int), 1, file);
        if (written_items < 1)
            return -1;

        for (int j = 0; j < detectorTrajectory[i].timeScaleSize; ++j)
        {
            written_items = fwrite(&(detectorTrajectory[i].timeScale[j].timeStart), 
                sizeof(double), 1, file);
            if (written_items < 1)
                return -1;

            written_items = fwrite(&(detectorTrajectory[i].timeScale[j].timeFinish), 
                sizeof(double), 1, file);
            if (written_items < 1)
                return -1;

            written_items = fwrite(&(detectorTrajectory[i].timeScale[j].numberOfPhotons), 
                sizeof(uint64), 1, file);
            if (written_items < 1)
                return -1;

            written_items = fwrite(&(detectorTrajectory[i].timeScale[j].weight), 
                sizeof(double), 1, file);
            if (written_items < 1)
                return -1;
        }
    }

    return 0;
}

static int WriteSectionRandomGenerator(FILE* file, MCG59* randomGenerator, int numThreadsPerProcess, 
    int numProcesses)
{
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_RANDOM_GENERATOR;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;
    
    unsigned int section_lenght = 2 * sizeof(int) + numThreadsPerProcess * numProcesses * 2 * sizeof(uint64);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numThreadsPerProcess, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numProcesses, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    for (int i = 0; i < numThreadsPerProcess * numProcesses; ++i)
    {
        written_items = fwrite(&(randomGenerator[i].value), sizeof(uint64), 1, file);
        if (written_items < 1)
            return -1;

        written_items = fwrite(&(randomGenerator[i].offset), sizeof(uint64), 1, file);
        if (written_items < 1)
            return -1;
    }

    return 0;
}

static int WriteSectionDetectorRanges(FILE* file, DetectorTrajectory* detectorTrajectory, int numberOfDetectors)
{
    unsigned long long int written_items;

	unsigned int section_id = MCML_SECTION_DETECTOR_RANGES;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int) + 2 * numberOfDetectors * sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfDetectors, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

	for (int i = 0; i < numberOfDetectors; ++i)
	{
		written_items = fwrite(&(detectorTrajectory[i].otherRange), sizeof(double), 1, file);
		if (written_items < 1)
			return -1;

		written_items = fwrite(&(detectorTrajectory[i].targetRange), sizeof(double), 1, file);
		if (written_items < 1)
			return -1;
	}

    return 0;
}

bool WriteOutputToFile(InputInfo* input, OutputInfo* output, char* fileName)
{
    FILE* file = fopen(fileName, "wb");
    if (file == NULL)
        return false;
    
    if (WriteSectionNumberOfPhotons(file, input->numberOfPhotons) != 0)
        return false;
    if (WriteSectionArea(file, input->area) != 0)
        return false;
    if (WriteSectionCubeDetectors(file, input->cubeDetector, input->numberOfCubeDetectors) != 0)
        return false;
	if (WriteSectionRingDetectors(file, input->ringDetector, input->numberOfRingDetectors) != 0)
        return false;
    if (WriteSectionSpecularReflectance(file, output->specularReflectance) != 0)
        return false;
    if (WriteSectionCommonTrajectories(file, output->absorption, output->gridSize) != 0)
        return false;
	if (WriteSectionScatteringMap(file, output->scatteringMap, output->gridSize) != 0)
        return false;
	if (WriteSectionDepthMap(file, output->depthMap, output->gridSize) != 0)
        return false;
    if (WriteSectionDetectorWeights(file, output->weightInDetector, output->numberOfDetectors) != 0)
        return false;
	if (WriteSectionGridDetector(file, output->weightInGridDetector, output->gridDetectorSize) != 0)
		return false;
    if (WriteSectionDetectorTrajectories(file, output->detectorTrajectory, output->numberOfDetectors) != 0)
        return false;
    if (WriteSectionDetectorTimeScale(file, output->detectorTrajectory, output->numberOfDetectors) != 0)
        return false;
	if (WriteSectionDetectorRanges(file, output->detectorTrajectory, output->numberOfDetectors) != 0)
		return false;

    fflush(file);
    fclose(file);

    return true;
}

bool WriteBackupToFile(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, 
    int numThreadsPerProcess, int numProcesses)
{
    FILE* file = NULL;
    char fileName[128];

    char* buffer = NumberToString(input->numberOfPhotons);
    sprintf(fileName, "xmcml_%s.mcml.bk", buffer);
    delete[] buffer;

    file = fopen(fileName, "wb");
    if (file == NULL)
        return false;

    if (WriteSectionNumberOfPhotons(file, input->numberOfPhotons) != 0)
        return false;
    if (WriteSectionArea(file, input->area) != 0)
        return false;
    if (WriteSectionCubeDetectors(file, input->cubeDetector, input->numberOfCubeDetectors) != 0)
        return false;
	if (WriteSectionRingDetectors(file, input->ringDetector, input->numberOfRingDetectors) != 0)
        return false;
    if (WriteSectionSpecularReflectance(file, output->specularReflectance) != 0)
        return false;
    if (WriteSectionCommonTrajectories(file, output->absorption, output->gridSize) != 0)
        return false;
	if (WriteSectionScatteringMap(file, output->scatteringMap, output->gridSize) != 0)
        return false;
	if (WriteSectionDepthMap(file, output->depthMap, output->gridSize) != 0)
        return false;
    if (WriteSectionDetectorWeights(file, output->weightInDetector, output->numberOfDetectors) != 0)
        return false;
	if (WriteSectionGridDetector(file, output->weightInGridDetector, output->gridDetectorSize) != 0)
		return false;
    if (WriteSectionDetectorTrajectories(file, output->detectorTrajectory, output->numberOfDetectors) != 0)
        return false;
    if (WriteSectionDetectorTimeScale(file, output->detectorTrajectory, output->numberOfDetectors) != 0)
        return false;
    if (WriteSectionRandomGenerator(file, randomGenerator, numThreadsPerProcess, numProcesses) != 0)
        return false;

    fflush(file);
    fclose(file);

    return true;
}
