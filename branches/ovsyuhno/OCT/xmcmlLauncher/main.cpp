#include <mpi.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

#include "cmd_parser.h"
#include "parser.h"
#include "launcher_mpi.h"
#include "launcher_omp.h"
#include "portable_time.h"
#include "writer.h"
#include "weight_integral.h"

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
		if (input->ringDetector != NULL)
		{
			delete[] input->ringDetector;
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
        if (input->bvhTree != NULL)
        {
            delete input->bvhTree;
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
    int mpi_rank, mpi_size;
    MCG59* randomGenerator = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    ParseCmd(argc, argv, &args);
    if (args.inputFileName == NULL || args.surfaceFileName == NULL)
    {
        PrintHelp();
		MPI_Finalize();
        return 0;
    }

	/*input.startPosition.x = args.launchPointX;
	input.startPosition.y = args.launchPointY;*/

    if (mpi_rank == 0)
    {       
        isOk = ParseInputFile(args.inputFileName, &input);
		input.startPosition.x = args.launchPointX;
		input.startPosition.y = args.launchPointY;
        printf("Parsing input file...%s\n", isOk ? "OK" : "FALSE");
        if (!isOk) 
        {
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
            return 0;
        }

        isOk = ParseSurfaceFile(args.surfaceFileName, &input);
        printf("Parsing surface file...%s\n", isOk ? "OK" : "FALSE");
        if (!isOk)
        {
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
            return 0;
        }

        InitOutput(&input, &output);

        randomGenerator = new MCG59[mpi_size * args.numberOfThreads];
        if (args.backupFileName == NULL)
        {
            for (int i = 0; i < mpi_size * args.numberOfThreads; ++i)
            {
                InitMCG59(&(randomGenerator[i]), args.seed, i, mpi_size * args.numberOfThreads);
            }
        }
        else
        {
            isOk = ParseBackupFile(args.backupFileName, &output, randomGenerator, args.numberOfThreads, mpi_size);
            printf("Parsing backup file...%s\n", isOk ? "OK" : "FALSE");
            if (!isOk)
            {
                MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
                return 0;
            }

            if (input.numberOfPhotons <= output.numberOfPhotons)
            {
                printf("Number of photons in backup more then in input: nothing to do\n");
                MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
                return 0;
            }

            input.numberOfPhotons -= output.numberOfPhotons;
        }

        if (input.useBiasing)
        {
            start = PortableGetTime();
            isOk = GetWeightIntegralTable(&input, &args);
            finish = PortableGetTime();
            printf("Getting weight integral table...%s\n", isOk ? "OK" : "FALSE");
            printf("Getting weight integral table time: %.2f s\n", finish - start);
            if (!isOk)
            {
                MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
                return 0;
            }
        }
        else
        {
            input.numberOfWeightTables = 1;
            input.weightTable = new WeightIntegralTable[1];
            input.weightTable[0].numberOfElements = 1;
            input.weightTable[0].elements = new double[1];
        }

        printf("Processing...");
        start = PortableGetTime();
    }

    LaunchMPI(&input, &output, randomGenerator, args.numberOfThreads, args.backupPortionPerNode);

    if (mpi_rank == 0)
    {
        finish = PortableGetTime();
        printf("OK\n");

        input.numberOfPhotons = output.numberOfPhotons;
        isOk = WriteOutputToFile(&input, &output, args.outputFileName);
        printf("Writing output file...%s\n", isOk ? "OK" : "FALSE");

        printf("Time: %.2f\n", finish - start);
    }

    FreeInput(&input);
    FreeOutput(&output);

    MPI_Finalize();

    return 0;
}