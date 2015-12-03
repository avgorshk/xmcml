#include <stdio.h>
#include <stdlib.h>

#include "reader.h"
#include "..\..\OCT\xmcmlLauncher\sections.h"

typedef unsigned long long int uint64;

bool ReadSectionTimeScales(FILE* file, InputInfo* input)
{
	unsigned int sectionLength, numberOfDetectors;
	fread(&sectionLength, sizeof(int), 1, file);

    fread(&numberOfDetectors, sizeof(int), 1, file);
	if (numberOfDetectors < input->detectorID + 1)
        return false;

	for(int i = 0; i < input->detectorID; i++)
	{
		int timeScaleSize;
		fread(&timeScaleSize, sizeof(int), 1, file);
		fseek(file, timeScaleSize * (sizeof(uint64) + 3 * sizeof(double)), SEEK_CUR);
	}

	fread(&(input->timeScaleSize), sizeof(int), 1, file);
	input->timeStart = new double[input->timeScaleSize];
	input->TimeFinish = new double[input->timeScaleSize];
	input->weight = new double[input->timeScaleSize];

	for(int i = 0; i < input->timeScaleSize; i++)
	{
		fread(&(input->timeStart[i]), sizeof(double), 1, file);
		fread(&(input->TimeFinish[i]), sizeof(double), 1, file);
		fseek(file, sizeof(uint64), SEEK_CUR);
		fread(&(input->weight[i]), sizeof(double), 1, file);
	}

	for(int i = input->detectorID + 1; i < numberOfDetectors; i++)
	{
		int timeScaleSize;
		fread(&timeScaleSize, sizeof(int), 1, file);
		fseek(file, timeScaleSize * (sizeof(uint64) + 3 * sizeof(double)), SEEK_CUR);
	}
	return true;
}

bool ReedOtherSection(FILE* file)
{
	unsigned int sectionLength;
	fread(&sectionLength, sizeof(int), 1, file);
	fseek(file, sectionLength, SEEK_CUR);
	return true;
}

int ReadSection(FILE* file, InputInfo* input)
{
	unsigned int sectionID;
	fread(&sectionID, sizeof(int), 1, file);
	if(sectionID == MCML_SECTION_DETECTOR_TIME_SCALE)
	{
		if(ReadSectionTimeScales(file, input))
			return 1;
		else 
			return -1;
	}
	else
	{
		if(ReedOtherSection(file))
			return 0;
		else
			return -1;
	}
}

bool ReadFromFile(InputInfo* input, char* fileName)
{
    FILE* file = fopen(fileName, "rb");
	//int f;

	while(true)
	{
		int f;
		f = ReadSection(file, input);
		if(f == -1)
		{
			fflush(file);
			fclose(file);
			return false;
		}
		if(f == 1)
			break;
	}

    fflush(file);
    fclose(file);
	return true;
}
