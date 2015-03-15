#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <mpi.h>

#include "mcml.h"

#define GPU_DEVICES 2

FILE *GetFile(char *Fname);
void CheckParm(FILE* File_Ptr, InputStruct * In_Ptr);
short ReadNumRuns(FILE* File_Ptr);
void ReadParm(FILE* File_Ptr, InputStruct * In_Ptr);
void InitOutputData(InputStruct In_Parm, OutStruct * Out_Ptr);
void SumScaleResult(InputStruct In_Parm, OutStruct * Out_Ptr);
void WriteResult(InputStruct In_Parm, OutStruct Out_Parm, char * TimeReport);
void FreeData(InputStruct In_Parm, OutStruct * Out_Ptr);

void GetFnameFromArgv(int argc,
                      char * argv[],
                      char * input_filename)
{
    if(argc>=2) { /* filename in command line */
        strcpy(input_filename, argv[argc - 1]);
    }
    else
        input_filename[0] = '\0';
}

void FreeOutStruct(InputStruct* in, OutStruct* out)
{
	for (int i = 0; i < in->nr; ++i)
	{
		free(out->A_rz[i]);
	}
	free(out->A_rz);

	for (int i = 0; i < in->nr; ++i)
	{
		free(out->Rd_ra[i]);
		free(out->Tt_ra[i]);
	}
	free(out->Rd_ra);
	free(out->Tt_ra);
}

void SetControlStruct(ControlStruct* control, int num_photons, int mpi_size, int mpi_id)
{
	int photons = num_photons / mpi_size;
	int gpu_devices = GPU_DEVICES;
	int gpu_threads = GetNumberOfGPUThreads();
	int gpu_photons = photons;
	control->mcg59_step = gpu_devices * gpu_threads * mpi_size;
	control->mcg59_first_id = mpi_id * gpu_devices * gpu_threads;
	control->mcg59_seed = 777;
	control->omp_threads = 0;
	control->omp_photons = 0;
	control->gpu_devices = gpu_devices;
	control->gpu_photons = gpu_photons;
}

int main(int argc, char *argv[])
{
    char input_filename[STRLEN];
    FILE *input_file_ptr;
    short num_runs;
    InputStruct in_parm;

	int id;
	int size;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if (id == 0)
	{
		printf("GeteroMCML 1.0\n");
		GetFnameFromArgv(argc, argv, input_filename);
		input_file_ptr = GetFile(input_filename);
		CheckParm(input_file_ptr, &in_parm);
		num_runs = ReadNumRuns(input_file_ptr);
		printf("OK\n");
		ReadParm(input_file_ptr, &in_parm);
	    
		clock_t start, finish;
		char* time_report = new char[256];

		start = clock();

		MPI_Bcast(&in_parm, sizeof(InputStruct), MPI_BYTE, 0, MPI_COMM_WORLD);
		MPI_Bcast(in_parm.layerspecs, sizeof(LayerStruct) * (in_parm.num_layers + 2), 
			MPI_BYTE, 0, MPI_COMM_WORLD);

		ControlStruct control;
		SetControlStruct(&control, in_parm.num_photons, size, id);

		OutStruct out_parm;
		InitOutputData(in_parm, &out_parm);
		
		CountMcml(&in_parm, &out_parm, &control);

		double** A_rz = (double**)malloc(in_parm.nr * sizeof(double*));
		for (int i = 0; i < in_parm.nr; ++i)
		{
			A_rz[i] = (double*)malloc(in_parm.nz * sizeof(double));
			MPI_Reduce(out_parm.A_rz[i], A_rz[i], in_parm.nz, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}

		double** Rd_ra = (double**)malloc(in_parm.nr * sizeof(double*));
		for (int i = 0; i < in_parm.nr; ++i)
		{
			Rd_ra[i] = (double*)malloc(in_parm.na * sizeof(double));
			MPI_Reduce(out_parm.Rd_ra[i], Rd_ra[i], in_parm.na, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}

		double** Tt_ra = (double**)malloc(in_parm.nr * sizeof(double*));
		for (int i = 0; i < in_parm.nr; ++i)
		{
			Tt_ra[i] = (double*)malloc(in_parm.na * sizeof(double));
			MPI_Reduce(out_parm.Tt_ra[i], Tt_ra[i], in_parm.na, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}

		FreeOutStruct(&in_parm, &out_parm);

		out_parm.A_rz = A_rz;
		out_parm.Rd_ra = Rd_ra;
		out_parm.Tt_ra = Tt_ra;

		finish = clock();

		sprintf(time_report, "Time of run: %.2f ms", (float)(finish - start) / CLOCKS_PER_SEC); 
		printf("%d: %s\n", num_runs, time_report);
        
        sprintf(in_parm.out_fname, "mcml_result_%u.out", size);

		SumScaleResult(in_parm, &out_parm);
		WriteResult(in_parm, out_parm, time_report);
		FreeData(in_parm, &out_parm);

		fclose(input_file_ptr);
	}
	else
	{
		InputStruct in_parm;
		MPI_Bcast(&in_parm, sizeof(InputStruct), MPI_BYTE, 0, MPI_COMM_WORLD);

		in_parm.layerspecs = new LayerStruct[in_parm.num_layers + 2];
		MPI_Bcast(in_parm.layerspecs, sizeof(LayerStruct) * (in_parm.num_layers + 2), 
			MPI_BYTE, 0, MPI_COMM_WORLD);

		ControlStruct control;
		SetControlStruct(&control, in_parm.num_photons, size, id);

		OutStruct out_parm;
		InitOutputData(in_parm, &out_parm);
		
		CountMcml(&in_parm, &out_parm, &control);

		for (int i = 0; i < in_parm.nr; ++i)
		{
			MPI_Reduce(out_parm.A_rz[i], NULL, in_parm.nz, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}

		for (int i = 0; i < in_parm.nr; ++i)
		{
			MPI_Reduce(out_parm.Rd_ra[i], NULL, in_parm.na, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}
		
		for (int i = 0; i < in_parm.nr; ++i)
		{
			MPI_Reduce(out_parm.Tt_ra[i], NULL, in_parm.na, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();

    return 0;
}