#include "mcml_writer.h"
#include <stdio.h>
#include <stdlib.h>

#define MCML_SECTION_NUMBER_OF_PHOTONS    0x01
#define MCML_SECTION_AREA                 0x02
#define MCML_SECTION_DETECTORS            0x03
#define MCML_SECTION_SPECULAR_REFLECTANCE 0x04
#define MCML_SECTION_TRAJECTORIES         0x05
#define MCML_SECTION_DETECTOR_WEIGHTS     0x06

static int WriteSectionNumberOfPhotons(FILE* file, int numberOfPhotons)
{
    size_t written_items;

    unsigned int section_id = MCML_SECTION_NUMBER_OF_PHOTONS;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfPhotons, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    return 0;
}

static int WriteSectionArea(FILE* file, Area* area)
{
    size_t written_items;

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

static int WriteSectionDetectors(FILE* file, Detector* detector, int numberOfDetectors)
{
    size_t written_items;

    unsigned int section_id = MCML_SECTION_DETECTORS;
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

static int WriteSectionSpecularReflectance(FILE* file, double specularReflectance)
{
    size_t written_items;

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

static int WriteSectionTrajectories(FILE* file, double* absorption, int absorptionSize)
{
    size_t written_items;

    unsigned int section_id = MCML_SECTION_TRAJECTORIES;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    unsigned int section_lenght = sizeof(int) + absorptionSize * sizeof(double);
    written_items = fwrite(&section_lenght, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&absorptionSize, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(absorption, sizeof(double), absorptionSize, file);
    if (written_items < absorptionSize)
        return -1;

    return 0;
}

static int WriteSectionDetectorWeights(FILE* file, double* detectorWeights, int numberOfDetectors)
{
    size_t written_items;

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

int WriteOutputToFile(InputInfo* input, OutputInfo* output, char* fileName)
{
    FILE* file = NULL;

    fopen_s(&file, fileName, "wb");
    if (file == NULL)
        return -1;
    
    WriteSectionNumberOfPhotons(file, input->numberOfPhotons);
    WriteSectionArea(file, input->area);
    WriteSectionDetectors(file, input->detector, input->numberOfDetectors);
    WriteSectionSpecularReflectance(file, output->specularReflectance);
    WriteSectionTrajectories(file, output->absorption, output->absorptionSize);
    WriteSectionDetectorWeights(file, output->weigthInDetector, output->numberOfDetectors);

    fflush(file);
    fclose(file);

    return 0;
}
