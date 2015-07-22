#include <stdio.h>
#include <string.h>
#include <time.h>

#include "logger.h"

void WriteMessageToLog(char* message, char* fileName)
{
    FILE* logFile = fopen(fileName, "a");

    time_t seconds = time(NULL);
    tm* currentTime = localtime(&seconds);
    char* currentTimeString = asctime(currentTime);

	int bufferSize = strlen(currentTimeString);
	char* buffer = new char[bufferSize];
	memcpy(buffer, currentTimeString, bufferSize);
	buffer[bufferSize - 1] = 0;
    fprintf(logFile, "%s: %s\n", buffer, message);

	delete[] buffer;
    fclose(logFile);
}

