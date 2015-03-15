#ifndef __WRITER_H
#define __WRITER_H

#include "../xmcml/mcml_kernel.h"
#include "../xmcml/mcml_mcg59.h"

bool WriteOutputToFile(InputInfo* input, OutputInfo* output, char* fileName);
bool WriteBackupToFile(InputInfo* input, OutputInfo* output, MCG59* randomGenerator, 
    int numThreadsPerProcess, int numProcesses);

#endif //__WRITER_H
