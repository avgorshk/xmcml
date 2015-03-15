#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

#include "cmd_parser.h"
#include "parser.h"
#include "launcher_omp.h"
#include "portable_time.h"
#include "writer.h"

void FreeInput(InputInfo* input)
{
    if (input != NULL)
    {
        if (input->area != NULL)
        {
            delete input->area;
        }
        if (input->cubeDetector != NULL)
        {
            delete[] input->cubeDetector;
        }
        if (input->layerInfo != NULL)
        {
            for (int i = 0; i < input->numberOfLayers; ++i)
            {
                if (input->layerInfo[i].surfaceId != NULL)
                {
                    delete[] input->layerInfo[i].surfaceId;
                }
            }
            delete[] input->layerInfo;
        }
        if (input->surface != NULL)
        {
            for (int i = 0; i < input->numberOfSurfaces; ++i)
            {
                if (input->surface[i].triangles != NULL)
                {
                    delete[] input->surface[i].triangles;
                }
                if (input->surface[i].vertices != NULL)
                {
                    delete[] input->surface[i].vertices;
                }
            }
            delete[] input->surface;
        }
    }
}

int main(int argc, char* argv[])
{
    InputInfo input;
    OutputInfo output;
    CmdArguments args;
    bool isOk = true;
    double start, finish;
    MCG59* randomGenerator = NULL;

    ParseCmd(argc, argv, &args);
    if (args.inputFileName == NULL || args.surfaceFileName == NULL)
    {
        PrintHelp();
        return 0;
    }
 
    isOk = ParseInputFile(args.inputFileName, &input);
    printf("Parsing input file...%s\n", isOk ? "OK" : "FALSE");
    if (!isOk) return 0;

    isOk = ParseSurfaceFile(args.surfaceFileName, &input);
    printf("Parsing surface file...%s\n", isOk ? "OK" : "FALSE");
    if (!isOk)
    {
        return 0;
    }

    InitOutput(&input, &output);

    randomGenerator = new MCG59[args.numberOfThreads];
    for (int i = 0; i < args.numberOfThreads; ++i)
    {
        InitMCG59(&(randomGenerator[i]), args.seed, i, args.numberOfThreads);
    }

    printf("Processing...");
    start = PortableGetTime();
    
    LaunchOMP(&input, &output, randomGenerator, args.numberOfThreads);
    
    finish = PortableGetTime();
    printf("OK\n");

    isOk = WriteOutputToFile(&input, &output, args.outputFileName);
    printf("Writing output file...%s\n", isOk ? "OK" : "FALSE");

    printf("Time: %.2f\n", finish - start);

    FreeInput(&input);
    FreeOutput(&output);

    return 0;
}