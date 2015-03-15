#ifndef __WRITER_H
#define __WRITER_H

#include "..\..\xmcml\xmcml\mcml_kernel.h"
#include "..\..\xmcml\xmcml\mcml_mcg59.h"

int WriteOutputToFile(InputInfo* input, OutputInfo* output, char* fileName);
int WriteBackupToFile(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, 
    int numThreadsPerProcess, int numProcesses);

#endif //__WRITER_H
