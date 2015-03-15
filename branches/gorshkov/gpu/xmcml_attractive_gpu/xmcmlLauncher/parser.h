#ifndef __PARSER_H
#define __PARSER_H

#include "../xmcml/mcml_kernel.h"

bool ParseInputFile(char* fileName, InputInfo* input);
bool ParseSurfaceFile(char* fileName, InputInfo* input);
bool ParseBackupFile(char* fileName, OutputInfo* output, MCG59* randomGenerator, 
    int numThreadsPerProcess, int numProcesses);

#endif //__PARSER_H
