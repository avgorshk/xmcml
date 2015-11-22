#include "matrix_math.h"
#include "reader.h"
#include "writer.h"
#include <stdio.h>

#define FILE_INPUT "result920.mcml.out"
#define FILE_OUTPUT "MSLresult920_16poly.mcml.out"

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

void initializeFuncPar(InputFuncPar* funcPar)
{
	funcPar->A = 1.0;
	funcPar->f = 5.0;
	funcPar->a = 3.14159265358979323846;
}

int main()
{
	InputInfo* input;
	OutputInfo* output;
	
	input = InitializeInput();

	if(input == nullptr)
		return 1;

	output = InitializeOutput();

	if(output == nullptr)
		return 1;

	bool ok;

	ok = ReadOutputToFile(input, output, FILE_INPUT);
	
	printf("Read file...%s\n",ok ? "OK" : "FALSE");

	if(!ok)
		return 1;

	InputFuncPar funcPar;

	initializeFuncPar(&funcPar);

	printf("Calculated absorption map...\n");
	MSL_3D(input->area, output->absorption, &funcPar);
	printf("Calculated scattering map...\n");
	MSL_3D(input->area, output->scatteringMap, &funcPar);
	printf("Calculated detectors map...\n");
	MSL_2D(input->area, output->weightInGridDetector, &funcPar);

	ok = WriteOutputToFile(input, output, FILE_OUTPUT);

	printf("Write file...%s\n",ok ? "OK" : "FALSE");

	if(!ok)
		return 1;

	FreeInput(input);
	FreeOutput(output);

	return 0;
}