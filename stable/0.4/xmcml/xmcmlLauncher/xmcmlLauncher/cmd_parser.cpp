#include "cmd_parser.h"

#include <string.h>
#include <stdlib.h>

void ParseCmd(int argc, char* argv[], CmdArguments* args)
{
    args->inputFileName = NULL;
    args->outputFileName = NULL;
    args->surfaceFile = NULL;
    args->numberOfThreads = 0;

    int index = argc - 2;
    for (int i = index; i >= 0; i -= 2)
    {
        if (strcmp(argv[i], "-i") == 0)
        {
            args->inputFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-s") == 0)
        {
            args->surfaceFile = argv[i + 1];
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            args->outputFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-nthreads") == 0)
        {
            args->numberOfThreads = atoi(argv[i + 1]);
        }
    }

    if (args->outputFileName == NULL)
        args->outputFileName = "xmcml.out";
}

