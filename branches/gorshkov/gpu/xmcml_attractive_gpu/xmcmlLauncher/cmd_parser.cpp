#include "cmd_parser.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "launcher_gpu.h"

void PrintHelp()
{
    printf("Usage: xmcmlLauncher.exe -i InputFileName.xml -s SurfaceFileName.surface\n");
    printf("\t-i - name of XML file with tissue model description\n");
    printf("\t-s - name of SURFACE file with description of tissue geometry\n");
    printf("\t-t - name of BIN file with weight integral table (by default: table will compute automatically)\n");
    printf("\t-b - name of MCML.BK backup file\n");
    printf("\t-o - name of MCML.OUT result file (by default: result.mcml.out)\n");
    printf("\t-seed - seed for MCG59 random number generator (by default: 777)\n");
    printf("\t-p - size of backup portion per node (by default: 10 millions photons)\n");
}

void ParseCmd(int argc, char* argv[], CmdArguments* args)
{
    args->inputFileName = NULL;
    args->outputFileName = NULL;
    args->surfaceFileName = NULL;
    args->backupFileName = NULL;
    args->weightTableFileName = NULL;
    args->numberOfThreads = GetMaxThreads();
    args->seed = 777;
    args->backupPortionPerNode = 10000000;

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
        else if (strcmp(argv[i], "-t") == 0)
        {
            args->weightTableFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-b") == 0)
        {
            args->backupFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-o") == 0)
        {
            args->outputFileName = argv[i + 1];
        }
        else if (strcmp(argv[i], "-seed") == 0)
        {
            args->seed = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-p") == 0)
        {
            args->backupPortionPerNode = atoi(argv[i + 1]);
        }
    }

    if (args->outputFileName == NULL)
        args->outputFileName = "result.mcml.out";
}

