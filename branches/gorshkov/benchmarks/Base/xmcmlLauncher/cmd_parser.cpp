#include "cmd_parser.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "launcher_omp.h"

void PrintHelp()
{
    printf("Usage: xmcmlLauncher.exe -i InputFileName.xml -s SurfaceFileName.surface\n");
    printf("\t-i - name of XML file with tissue model description\n");
    printf("\t-s - name of SURFACE file with description of tissue geometry\n");
    printf("\t-o - name of MCML.OUT result file (by default: result.mcml.out)\n");
    printf("\t-nthreads - number of threads for parallel execution (by default: number of logical cores of CPU)\n");
    printf("\t-seed - seed for MCG59 random number generator (by default: 777)\n");
}

void ParseCmd(int argc, char* argv[], CmdArguments* args)
{
    args->inputFileName = NULL;
    args->outputFileName = NULL;
    args->surfaceFileName = NULL;
    args->numberOfThreads = GetMaxThreads();
    args->seed = 777;

    int index = argc - 2;
    for (int i = index; i >= 0; i -= 2)
    {
        if (strcmp(argv[i], "-i") == 0)
        {
            args->inputFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-s") == 0)
        {
            args->surfaceFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            args->outputFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-nthreads") == 0)
        {
            args->numberOfThreads = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-seed") == 0)
        {
            args->seed = atoi(argv[i + 1]);
        }
    }

    if (args->outputFileName == NULL)
        args->outputFileName = "result.mcml.out";
}

