#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>

#include "reader.h"

#define MAX_DISTANSE 1000
#define EPSILON 1e-15
#define PI 3.141592

typedef struct __CmdArguments
{
    char* inputFileName;
    char* outputFileName;
	double l;
	double stepSize;
	int scaleSize;
	int maxDistance;
	double timeStart;
} CmdArguments;

double _atod(char* str, int length)
{
	for(int i = 0; i < length; i++)
	{
		if(str[i] == ',')
		{
			str[i] = '.';
		}
	}
	return atof(str);
}

void _dtos(std::string* str)
{
	for(int i = 0; i < str->length(); i++)
	{
		if((*str)[i] == '.')
			(*str)[i] = ',';
	}
}


void writeToFile(InputInfo* input, double* resultMatrix)
{
	int writeMode = std::ofstream::out;
	if(input->writeFileMode == 1)
		writeMode = std::ofstream::app;

	std::ofstream outputFile(input->outputFileName, writeMode);
	std::string tmp;
	if(input->writeFileMode == 0)
	{
		for(int i = 0; i < input->timeScaleSize; i ++)
		{
			tmp = std::to_string((long double)(input->timeStart[i]));
			_dtos(&tmp);
			outputFile << tmp << '	';
		}
	
		outputFile << input->TimeFinish[input->timeScaleSize - 1];
	}
	if(input->writeFileMode == 2)
	{
		for(int i = 0; i < input->timeScaleSize; i ++)
		{
			tmp = std::to_string((long double)resultMatrix[i]);
			outputFile << tmp << '\n';
		}
	}
	else
	{
	outputFile << "\n";
		for(int i = 0; i < input->timeScaleSize; i ++)
		{
			tmp = std::to_string((long double)resultMatrix[i]);
			outputFile << tmp << '	';
		}
	}
	outputFile.close();
}

int Mmin(int a, int b)
{
	if(a > b)
		return b;
	else
		return a;
}

int Mmax(int a, int b)
{
	if(a > b)
		return a;
	else
		return b;
}


void main(int argc, char* argv[])
{
	CmdArguments args;
	double* resultMatrix;
	double* inputMatrix;
	int index = argc - 2;
	InputInfo input;

	args.l = 1;
	args.timeStart = 0;
	args.maxDistance = 10;
	args.outputFileName = "default_out.txt";

	input.lyambda = 0.0;
	
    for (int i = index; i >= 0; i -= 2)
    {
        if (strcmp(argv[i], "-i") == 0)	//input file name
        {
            input.inputFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-o") == 0)	//output file name
        {
            input.outputFileName = argv[i + 1];
        }
		else if (strcmp(argv[i], "-l") == 0)	//convolution setting
        {
			input.convolutionSetting = atof(argv[i + 1]);
        }
		else if (strcmp(argv[i], "-d") == 0)	//detector ID
        {
			input.detectorID = atoi(argv[i + 1]);
        }
		else if (strcmp(argv[i], "-w") == 0)	//write file mode: 0 - w, 1 - app
        {
			input.writeFileMode = atoi(argv[i + 1]);
        }
		else if (strcmp(argv[i], "-c") == 0)
        {
			double lyambda = atof(argv[i + 1]);
			if(lyambda < EPSILON)
			{
				printf("Eror lyambda <= 0");
				return;
			}
			input.lyambda = 1.0 / lyambda;
        }
	}

	bool f = ReadFromFile(&input, input.inputFileName);
	resultMatrix = new double[input.timeScaleSize];
	memset(resultMatrix, 0, 8 * input.timeScaleSize);
	inputMatrix = input.weight;
	double stepSize = input.timeStart[1] - input.timeStart[0];

	double z = 0;
	for(int i = 0; i < input.timeScaleSize; i++)
	{
		for(int j = Mmax(0, i - MAX_DISTANSE); j < Mmin(input.timeScaleSize, i + MAX_DISTANSE); j++)
		{
			resultMatrix[i] += inputMatrix[j] * cos(2 * PI * (i - j) * stepSize * input.lyambda) * exp(-(i - j) * (i - j) * stepSize * stepSize /
				(input.convolutionSetting * input.convolutionSetting));
		}
	}
	writeToFile(&input, resultMatrix);
}