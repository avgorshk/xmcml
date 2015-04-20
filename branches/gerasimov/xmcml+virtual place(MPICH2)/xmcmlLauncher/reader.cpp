#include <stdio.h>

#include "sections.h"

#include "reader.h"

int SkipCurrentSection(FILE* file)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    if (fseek(file, section_length, SEEK_CUR) != 0)
        return -1;

    return 0;
}

int ReadThreadsFromBackupFile(char* fileName, int* numThreadsPerProcess, int* numProcesses)
{
    FILE* file = fopen(fileName, "rb");
    if (file == NULL)
        return -1;
    
    SkipCurrentSection(file); //MCML_SECTION_NUMBER_OF_PHOTONS
    SkipCurrentSection(file); //MCML_SECTION_AREA
    SkipCurrentSection(file); //MCML_SECTION_CUBE_DETECTORS
	SkipCurrentSection(file); //MCML_SECTION_RING_DETECTORS
    SkipCurrentSection(file); //MCML_SECTION_SPECULAR_REFLECTANCE
    SkipCurrentSection(file); //MCML_SECTION_COMMON_TRAJECTORIES
	SkipCurrentSection(file); //MCML_SECTION_SCATTERING_MAP
    SkipCurrentSection(file); //MCML_SECTION_DETECTOR_WEIGHTS
    SkipCurrentSection(file); //MCML_SECTION_DETECTOR_TRAJECTORIES
    SkipCurrentSection(file); //MCML_SECTION_DETECTOR_TIME_SCALE

    unsigned long long int reading_items;
    int section;
    unsigned int section_length;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_RANDOM_GENERATOR)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(numThreadsPerProcess, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(numProcesses, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    fclose(file);

    return 0;
}

int ReadRandomGeneratorFromBackupFile(char* fileName, MCG59* randomGenerator)
{
    FILE* file = fopen(fileName, "rb");
    if (file == NULL)
        return -1;
    
    SkipCurrentSection(file); //MCML_SECTION_NUMBER_OF_PHOTONS
    SkipCurrentSection(file); //MCML_SECTION_AREA
    SkipCurrentSection(file); //MCML_SECTION_CUBE_DETECTORS
	SkipCurrentSection(file); //MCML_SECTION_RING_DETECTORS
    SkipCurrentSection(file); //MCML_SECTION_SPECULAR_REFLECTANCE
    SkipCurrentSection(file); //MCML_SECTION_COMMON_TRAJECTORIES
	SkipCurrentSection(file); //MCML_SECTION_SCATTERING_MAP
    SkipCurrentSection(file); //MCML_SECTION_DETECTOR_WEIGHTS
    SkipCurrentSection(file); //MCML_SECTION_DETECTOR_TRAJECTORIES
    SkipCurrentSection(file); //MCML_SECTION_DETECTOR_TIME_SCALE

    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    int numThreadsPerProcess, numProcesses;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_RANDOM_GENERATOR)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&numThreadsPerProcess, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&numProcesses, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    for (int i = 0; i < numProcesses * numThreadsPerProcess; ++i)
    {
        reading_items = fread(&(randomGenerator[i].value), sizeof(uint64), 1, file);
        if (reading_items < 1)
            return -1;

        reading_items = fread(&(randomGenerator[i].offset), sizeof(uint64), 1, file);
        if (reading_items < 1)
            return -1;
    }

    fclose(file);

    return 0;
}

int ReadSectionNumberOfPhotons(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_NUMBER_OF_PHOTONS)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&(output->numberOfPhotons), sizeof(uint64), 1, file);
    if (reading_items < 1)
        return -1;

    return 0;
}

int ReadSectionCommonTrajectories(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    int gridSize;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_COMMON_TRAJECTORIES)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&gridSize, sizeof(int), 1, file);
    if (reading_items < 1 || gridSize != output->gridSize)
        return -1;

    reading_items = fread(output->absorption, sizeof(double), gridSize, file);
    if (reading_items < gridSize)
        return -1;

    return 0;
}

int ReadSectionScatteringMap(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    int gridSize;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_SCATTERING_MAP)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&gridSize, sizeof(int), 1, file);
    if (reading_items < 1 || gridSize != output->gridSize)
        return -1;

    reading_items = fread(output->scatteringMap, sizeof(double), gridSize, file);
    if (reading_items < gridSize)
        return -1;

    return 0;
}

int ReadSectionDetectorWeights(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    int numberOfDetectors;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTOR_WEIGHTS)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&numberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1 || numberOfDetectors != output->numberOfDetectors)
        return -1;

    reading_items = fread(output->weightInDetector, sizeof(double), numberOfDetectors, file);
    if (reading_items < numberOfDetectors)
        return -1;

    return 0;
}

int ReadSectionDetectorTrajectories(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    int numberOfDetectors;
    int trajectorySize;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTOR_TRAJECTORIES)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&numberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1 || numberOfDetectors != output->numberOfDetectors)
        return -1;

    for (int i = 0; i < numberOfDetectors; ++i)
    {
        reading_items = fread(&(output->detectorTrajectory[i].numberOfPhotons), sizeof(uint64), 1, file);
        if (reading_items < 1)
            return -1;

        reading_items = fread(&trajectorySize, sizeof(int), 1, file);
        if (reading_items < 1 || trajectorySize != output->detectorTrajectory[i].trajectorySize)
            return -1;

        reading_items = fread(output->detectorTrajectory[i].trajectory, sizeof(uint64), trajectorySize, file);
        if (reading_items < trajectorySize)
            return -1;
    }

    return 0;
}

int ReadSectionDetectorTimeScale(FILE* file, OutputInfo* output)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    int numberOfDetectors;
    int timeScaleSize;
    double timeStart, timeFinish;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTOR_TIME_SCALE)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&numberOfDetectors, sizeof(int), 1, file);
    if (reading_items < 1 || numberOfDetectors != output->numberOfDetectors)
        return -1;

    for (int i = 0; i < numberOfDetectors; ++i)
    {
        reading_items = fread(&timeScaleSize, sizeof(int), 1, file);
        if (reading_items < 1 || timeScaleSize != output->detectorTrajectory[i].timeScaleSize)
            return -1;

        for (int j = 0; j < timeScaleSize; ++j)
        {
            reading_items = fread(&timeStart, sizeof(double), 1, file);
            if (reading_items < 1 || timeStart != output->detectorTrajectory[i].timeScale[j].timeStart)
                return -1;

            reading_items = fread(&timeFinish, sizeof(double), 1, file);
            if (reading_items < 1 || timeFinish != output->detectorTrajectory[i].timeScale[j].timeFinish)
                return -1;

            reading_items = fread(&(output->detectorTrajectory[i].timeScale[j].numberOfPhotons), 
                sizeof(uint64), 1, file);
            if (reading_items < 1)
                return -1;

            reading_items = fread(&(output->detectorTrajectory[i].timeScale[j].weight), 
                sizeof(double), 1, file);
            if (reading_items < 1)
                return -1;
        }
    }

    return 0;
}

int ReadOutputFormBackupFile(char* fileName, OutputInfo* output)
{
    FILE* file = fopen(fileName, "rb");
    if (file == NULL)
        return -1;

    if (ReadSectionNumberOfPhotons(file, output) != 0)
        return -1;
    
    SkipCurrentSection(file); //MCML_SECTION_AREA
    SkipCurrentSection(file); //MCML_SECTION_CUBE_DETECTORS
	SkipCurrentSection(file); //MCML_SECTION_RING_DETECTORS
    SkipCurrentSection(file); //MCML_SECTION_SPECULAR_REFLECTANCE

    if (ReadSectionCommonTrajectories(file, output) != 0)
        return -1;
	if (ReadSectionScatteringMap(file,output) != 0)
		return -1;
    if (ReadSectionDetectorWeights(file, output) != 0)
        return -1;
    if (ReadSectionDetectorTrajectories(file, output) != 0)
        return -1;
    if (ReadSectionDetectorTimeScale(file, output) != 0)
        return -1;

    fclose(file);

    return 0;
}
