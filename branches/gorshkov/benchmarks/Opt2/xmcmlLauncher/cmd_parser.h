#ifndef __CMD_PARSER_H
#define __CMD_PARSER_H

typedef struct __CmdArguments
{
    char* inputFileName;
    char* surfaceFileName;
    char* outputFileName;
    int numberOfThreads;
    unsigned int seed;
} CmdArguments;

void ParseCmd(int argc, char* argv[], CmdArguments* args);
void PrintHelp();

#endif //__CMD_PARSER_H
