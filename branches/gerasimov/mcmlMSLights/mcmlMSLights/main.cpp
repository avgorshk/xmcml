#include "matrix_math.h"
#include "reader.h"
#include "writer.h"
#include <stdio.h>
#include <string>

InputInfo* InitializeInput()
{
	InputInfo* input = new InputInfo;
	return input;
}

void FreeInput(InputInfo* input)
{
	if(input != nullptr) 
	{
		delete[] input->area;
		delete[] input->cubeDetector;
		delete[] input->ringDetector;
		delete[] input;
	}
}

OutputInfo* InitializeOutput()
{
	OutputInfo* output = new OutputInfo;
	return output;
}

void FreeOutput(OutputInfo* output)
{
	if(output != nullptr)
	{
		delete[] output->absorption;
		delete[] output->scatteringMap;
		delete[] output->depthMap;
		delete[] output->weightInDetector;
		delete[] output->weightInGridDetector;
	
		for(int i = 0; i < output->numberOfDetectors; i++)
		{
			delete[] output->detectorTrajectory[i].timeScale;
			delete[] output->detectorTrajectory[i].trajectory;
		}

		delete[] output->detectorTrajectory;
		delete[] output;
	}
}

void ParseCommandArgs(int argc, char* argv[], char* fileIO[], InputFuncPar* funcPar)
{
	char* fileInput = nullptr, *fileOutput = nullptr;

    for(int i = 0; i < argc; i++)
	{
		if(strcmp(argv[i], "-i") == 0)
		{
			fileIO[0] = argv[i + 1];
			i++;
		}
		if(strcmp(argv[i], "-o") == 0)
		{
			fileIO[1] = argv[i + 1];
			i++;
		}
		if(strcmp(argv[i], "-A") == 0)
		{
			funcPar->A = atof(argv[i + 1]);
			i++;
		}
		if(strcmp(argv[i], "-a") == 0)
		{
			funcPar->a = atof(argv[i + 1]);
			i++;
		}
		if(strcmp(argv[i], "-f") == 0)
		{
			funcPar->f = atof(argv[i + 1]);
			i++;
		}
	}
}

int main(int argc, char* argv[])
{
	InputFuncPar funcPar;

	funcPar.A = 0.0;
	funcPar.a = 0.0;
	funcPar.f = 0.0;

	char* fileIO[2] = {nullptr, nullptr};

	ParseCommandArgs(argc, argv, fileIO, &funcPar);
	
	char* fileInput = fileIO[0], *fileOutput = fileIO[1];

	if((fileInput == nullptr) || (fileOutput == nullptr))
		return 1;

	InputInfo* input = InitializeInput();

	if(input == nullptr)
		return 1;

	OutputInfo* output = InitializeOutput();

	if(output == nullptr)
		return 1;

	bool ok;

	ok = ReadOutputToFile(input, output, fileInput);
	
	printf("Read file...%s\n",ok ? "OK" : "FALSE");

	if(!ok)
		return 1;

	printf("Calculated absorption map...\n");
	MSL_3D(input->area, output->absorption, &funcPar);
	
	printf("Calculated scattering map...\n");
	MSL_3D(input->area, output->scatteringMap, &funcPar);
	
	printf("Calculated depth map...\n");
	MSL_3D(input->area, output->depthMap, &funcPar);

	printf("Calculated detectors map...\n");
	MSL_2D(input->area, output->weightInGridDetector, &funcPar);
	
	ok = WriteOutputToFile(input, output, fileOutput);

	printf("Write file...%s\n",ok ? "OK" : "FALSE");

	if(!ok)
		return 1;

	FreeInput(input);
	FreeOutput(output);

	return 0;
}