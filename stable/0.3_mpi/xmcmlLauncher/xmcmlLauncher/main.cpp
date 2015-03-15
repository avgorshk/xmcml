#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "cmd_parser.h"
#include "parser.h"
#include "launcher_mpi.h"
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
        if (input->detector != NULL)
        {
            delete[] input->detector;
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

void FreeOutput(OutputInfo* output)
{
    if (output != NULL)
    {
        if (output->absorption != NULL)
        {
            delete[] output->absorption;
        }
        if (output->weightInDetector != NULL)
        {
            delete[] output->weightInDetector;
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
    int mpi_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    ParseCmd(argc, argv, &args);
    if (args.inputFileName == NULL || args.surfaceFile == NULL)
    {
        printf("Input or surface file is not specified\n");
        return 0;
    }

    if (mpi_rank == 0)
    {       
        isOk = ParseInputFile(args.inputFileName, &input);
        printf("Parsing input file...%s\n", isOk ? "OK" : "FALSE");
        if (!isOk) return 0;

        isOk = ParseSurfaceFile(args.surfaceFile, &input);
        printf("Parsing surface file...%s\n", isOk ? "OK" : "FALSE");
        if (!isOk) return 0;

        printf("Processing...");
        start = PortableGetTime();
    }

    LaunchMPI(&input, &output, args.numberOfThreads);

    if (mpi_rank == 0)
    {
        finish = PortableGetTime();
        printf("OK\n");
        printf("Time: %.2f\n", finish - start);

        WriteOutputToFile(&input, &output, args.outputFileName); 
    }

    FreeInput(&input);
    FreeOutput(&output);

    MPI_Finalize();

    return 0;
}