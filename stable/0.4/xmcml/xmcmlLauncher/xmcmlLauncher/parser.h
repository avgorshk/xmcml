#ifndef __PARSER_H
#define __PARSER_H

#include "..\..\xmcml\xmcml\mcml_kernel.h"

bool ParseInputFile(char* fileName, InputInfo* input);
bool ParseSurfaceFile(char* fileName, InputInfo* input);

#endif //__PARSER_H
