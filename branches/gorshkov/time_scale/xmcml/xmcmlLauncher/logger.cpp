#include <stdio.h>
#include <string.h>
#include <time.h>

#include "logger.h"

void WriteMessageToLog(char* message, char* fileName)
{
    FILE* logFile;
    fopen_s(&logFile, fileName, "a");

    char currentTimeString[32];
    time_t seconds = time(NULL);
    tm currentTime;
    localtime_s(&currentTime, &seconds);
    asctime_s(currentTimeString, 32, &currentTime);
    currentTimeString[strlen(currentTimeString) - 1] = 0;
    fprintf(logFile, "%s: %s\n", currentTimeString, message);

    fclose(logFile);
}

