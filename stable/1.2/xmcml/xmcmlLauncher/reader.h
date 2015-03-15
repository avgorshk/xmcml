#ifndef __READER_H
#define __READER_H

#include "../xmcml/mcml_kernel_types.h"
#include "../xmcml/mcml_mcg59.h"

int ReadThreadsFromBackupFile(char* fileName, int* numThreadsPerProcess, int* numProcesses);
int ReadRandomGeneratorFromBackupFile(char* fileName, MCG59* randomGenerator);
int ReadOutputFormBackupFile(char* fileName, OutputInfo* output);

#endif //__READER_H
