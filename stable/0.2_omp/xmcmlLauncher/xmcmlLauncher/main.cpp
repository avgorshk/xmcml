#include <stdlib.h>
#include <stdio.h>

#include "cmd_parser.h"
#include "parser.h"
#include "launcher.h"
#include "portable_time.h"
#include "writer.h"

void FreeInput(InputInfo* input)
{
    delete input->area;
    delete[] input->detector;
    for (int i = 0; i < input->numberOfLayers; ++i)
    {
        delete[] input->layerInfo[i].surfaceId;
    }
    delete[] input->layerInfo;
    for (int i = 0; i < input->numberOfSurfaces; ++i)
    {
        delete[] input->surface[i].triangles;
        delete[] input->surface[i].vertices;
    }
    delete[] input->surface;
}

void FreeOutput(OutputInfo* output)
{
    delete[] output->absorption;
    delete[] output->weigthInDetector;
}

int main(int argc, char* argv[])
{
    InputInfo input;
    OutputInfo output;
    bool isOk = true;
    double start, finish;

    CmdArguments args;
    ParseCmd(argc, argv, &args);
    if (args.inputFileName == NULL || args.surfaceFile == NULL)
    {
        printf("Input or surface file is not specified\n");
        return 0;
    }

    isOk = ParseInputFile(args.inputFileName, &input);
    printf("Parsing input file...%s\n", isOk ? "OK" : "FALSE");
    if (!isOk) return 0;

    isOk = ParseSurfaceFile(args.surfaceFile, &input);
    printf("Parsing surface file...%s\n", isOk ? "OK" : "FALSE");
    if (!isOk) return 0;

    printf("Processing...");
    start = PortableGetTime();
    Launch(&input, &output, args.numberOfThreads);
    finish = PortableGetTime();
    printf("OK\n");
    printf("Time: %.2f\n", finish - start);

    WriteOutputToFile(&input, &output, args.outputFileName);

    FreeInput(&input);
    FreeOutput(&output);

    return 0;
}