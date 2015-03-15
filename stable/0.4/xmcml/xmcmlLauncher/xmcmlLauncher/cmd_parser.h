#ifndef __CMD_PARSER_H
#define __CMD_PARSER_H

typedef struct __CmdArguments
{
    char* inputFileName;
    char* surfaceFile;
    char* outputFileName;
    int numberOfThreads;
} CmdArguments;

void ParseCmd(int argc, char* argv[], CmdArguments* args);

#endif //__CMD_PARSER_H
