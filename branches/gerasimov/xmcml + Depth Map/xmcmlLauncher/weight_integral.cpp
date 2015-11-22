#include "weight_integral.h"

#include <stdio.h>
#include <omp.h>

#include "sections.h"

void ComputeWeightIntegral(InputInfo* input, CmdArguments* args);
int GetDifferentAnisotropies(double* anisotropies, InputInfo* input);
uint GetWeightIntegralPrecision(double anisotropy, uint basePrecision);
bool WriteWeightIntegralTableToFile(InputInfo* input);
uint GetWeightIntegralSectionLength(InputInfo* input);
bool ParseWeightIntegralFile(InputInfo* input, CmdArguments* args);

bool GetWeightIntegralTable(InputInfo* input, CmdArguments* args)
{
    bool isOk = true;
    input->numberOfWeightTables = 0;
    input->weightTable = NULL;

    if (args->weightTableFileName != NULL)
    {
        isOk = ParseWeightIntegralFile(input, args);
    }
    else
    {
        ComputeWeightIntegral(input, args);
        isOk = WriteWeightIntegralTableToFile(input);
    }
    return isOk;
}

void ComputeWeightIntegral(InputInfo* input, CmdArguments* args)
{
    omp_set_num_threads(args->numberOfThreads);

    double* anisotropies = new double[input->numberOfLayers];
    int numberOfDifferentAnisotropies = GetDifferentAnisotropies(anisotropies, input);
    input->numberOfWeightTables = numberOfDifferentAnisotropies;
    input->weightTable = new WeightIntegralTable[input->numberOfWeightTables];

    for (int i = 0; i < numberOfDifferentAnisotropies; ++i)
    {
        input->weightTable[i].anisotropy = anisotropies[i];
        input->weightTable[i].numberOfElements = input->weightTablePrecision + 1;
        input->weightTable[i].elements = new double[input->weightTable[i].numberOfElements];

        double a = -1.0;
        double b = 1.0;
        double step = (b - a) / input->weightTablePrecision;
        uint integralPrecision = GetWeightIntegralPrecision(anisotropies[i], input->weightIntegralPrecision);
        double x = 0;

        #pragma omp parallel for private(x)
        for (int j = 0; j <= input->weightTablePrecision; ++j)
        {
            x = a + j * step;
            input->weightTable[i].elements[j] = ComputeWeightIntegral(x, anisotropies[i], 
                input->attractiveFactor, integralPrecision);
        }
    }

    delete[] anisotropies;
}

int GetDifferentAnisotropies(double* anisotropies, InputInfo* input)
{
    anisotropies[0] = input->layerInfo[1].anisotropy;
    int numberOfDifferentAnisotropies = 1;

    for (int i = 2; i < input->numberOfLayers; ++i)
    {
        double anisotropy = input->layerInfo[i].anisotropy;
        
        bool isHitFound = false;
        for (int j = 0; j < numberOfDifferentAnisotropies; ++j)
        {
            if (anisotropy == anisotropies[j])
            {
                isHitFound = true;
                break;
            }
        }

        if (!isHitFound)
        {
            anisotropies[numberOfDifferentAnisotropies] = anisotropy;
            ++numberOfDifferentAnisotropies;
        }
    }

    return numberOfDifferentAnisotropies;
}

uint GetWeightIntegralPrecision(double anisotropy, uint basePrecision)
{
    uint precision = basePrecision;
    if (anisotropy > 0.98)      precision *= 16;
    else if (anisotropy > 0.97) precision *= 8;
    else if (anisotropy > 0.95) precision *= 4;
    else if (anisotropy > 0.90) precision *= 2;

    return precision;
}

bool WriteWeightIntegralTableToFile(InputInfo* input)
{
    const char* fileName = "xmcml_weight_integral_table.bin";
    
    FILE* file = fopen(fileName, "wb");
    if (file == NULL)
        return false;

    unsigned long long int writtenItems;

    uint sectionId = MCML_SECTION_WEIGHT_INTEGRAL;
    writtenItems = fwrite(&sectionId, sizeof(uint), 1, file);
    if (writtenItems < 1)
        return false;

    uint sectionLength = GetWeightIntegralSectionLength(input);
    writtenItems = fwrite(&sectionLength, sizeof(uint), 1, file);
    if (writtenItems < 1)
        return false;

    writtenItems = fwrite(&(input->attractiveFactor), sizeof(double), 1, file);
    if (writtenItems < 1)
        return false;

    writtenItems = fwrite(&(input->weightIntegralPrecision), sizeof(int), 1, file);
    if (writtenItems < 1)
        return false;

    writtenItems = fwrite(&(input->numberOfWeightTables), sizeof(int), 1, file);
    if (writtenItems < 1)
        return false;

    for (int i = 0; i < input->numberOfWeightTables; ++i)
    {
        writtenItems = fwrite(&(input->weightTable[i].anisotropy), sizeof(double), 1, file);
        if (writtenItems < 1)
            return false;

        writtenItems = fwrite(&(input->weightTable[i].numberOfElements), sizeof(int), 1, file);
        if (writtenItems < 1)
            return false;

        writtenItems = fwrite(input->weightTable[i].elements, 
             sizeof(double), input->weightTable[i].numberOfElements, file);
        if (writtenItems < input->weightTable[i].numberOfElements)
            return false;
    }

    fflush(file);
    fclose(file);

    return true;
}

uint GetWeightIntegralSectionLength(InputInfo* input)
{
    uint sectionLength = sizeof(double) + sizeof(int) + sizeof(int);
    for (int i = 0; i < input->numberOfWeightTables; ++i)
    {
        sectionLength += sizeof(double) + sizeof(int) + 
            input->weightTable[i].numberOfElements * sizeof(double);
    }

    return sectionLength;
}

bool ParseWeightIntegralFile(InputInfo* input, CmdArguments* args)
{
    FILE* file = fopen(args->weightTableFileName, "rb");
    if (file == NULL)
        return false;

    unsigned long long int readingItems;
    int section;

    readingItems = fread(&section, sizeof(int), 1, file);
    if (readingItems < 1 || section != MCML_SECTION_WEIGHT_INTEGRAL)
        return false;

    uint sectionLength;
    readingItems = fread(&sectionLength, sizeof(uint), 1, file);
    if (readingItems < 1)
        return false;

    double* anisotropies = new double[input->numberOfLayers];
    int numberOfDifferentAnisotropies = GetDifferentAnisotropies(anisotropies, input);

    double attracriveFactor = 0.0;
    readingItems = fread(&attracriveFactor, sizeof(double), 1, file);
    if (readingItems < 1 || input->attractiveFactor != attracriveFactor)
        return false;

    int weightIntegralPrecision = 0;
    readingItems = fread(&weightIntegralPrecision, sizeof(int), 1, file);
    if (readingItems < 1 || input->weightIntegralPrecision != weightIntegralPrecision)
        return false;

    readingItems = fread(&(input->numberOfWeightTables), sizeof(int), 1, file);
    if (readingItems < 1 || input->numberOfWeightTables != numberOfDifferentAnisotropies)
        return false;

    input->weightTable = new WeightIntegralTable[input->numberOfWeightTables];
    for (int i = 0; i < input->numberOfWeightTables; ++i)
    {
        readingItems = fread(&(input->weightTable[i].anisotropy), sizeof(double), 1, file);
        if (readingItems < 1 || input->weightTable[i].anisotropy != anisotropies[i])
            return false;

        readingItems = fread(&(input->weightTable[i].numberOfElements), sizeof(int), 1, file);
        if (readingItems < 1 || input->weightTable[i].numberOfElements - 1 != input->weightTablePrecision)
            return false;

        input->weightTable[i].elements = new double[input->weightTable[i].numberOfElements];
        readingItems = fread(input->weightTable[i].elements, sizeof(double), 
            input->weightTable[i].numberOfElements, file);
        if (readingItems < input->weightTable[i].numberOfElements)
            return false;
    }

    fclose(file);
    delete[] anisotropies;

    return true;
}