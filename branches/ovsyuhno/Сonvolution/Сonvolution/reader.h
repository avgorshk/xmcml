#ifndef __READER_H
#define __READER_H

typedef struct __InputInfo
{
	int timeScaleSize;
	double* timeStart;
	double* TimeFinish;
	double* weight;
	int detectorID;
	char* inputFileName;
	char* outputFileName;
	double convolutionSetting;
	int writeFileMode;
} InputInfo;

bool ReadFromFile(InputInfo* input, char* fileName);

#endif