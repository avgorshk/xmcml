#ifndef __MCML_GETERO_H
#define __MCML_GETERO_H

#define STRLEN 256

typedef struct __LayerStruct
{
	double z0, z1;
	double n;
	double mua;
	double mus;
	double g;
	double cos_crit0, cos_crit1;
} LayerStruct;

typedef struct __InputStruct 
{
	char out_fname[STRLEN];
	char out_fformat;
	long num_photons;
	double Wth;
	double dz;
	double dr;
	double da;
	short nz;
	short nr;
	short na;
	short num_layers;
	LayerStruct * layerspecs;
} InputStruct;

typedef struct __OutStruct
{
	double Rsp;
	double** Rd_ra;
	double* Rd_r;
	double* Rd_a;
	double Rd;
	double** A_rz;
	double* A_z;
	double* A_l;
	double A;
	double** Tt_ra;
	double* Tt_r;
	double* Tt_a;
	double Tt;
} OutStruct;

typedef struct __ControlStruct
{
	unsigned int mcg59_step;
	unsigned int mcg59_first_id;
	unsigned int mcg59_seed;
	int omp_threads;
	unsigned int omp_photons;
	int gpu_devices;
	unsigned int gpu_photons;
} ControlStruct;

void CountMcml(InputStruct* in, OutStruct* out, ControlStruct* control);
int GetNumberOfCPU();
int GetNumberOfGPU();
int GetNumberOfGPUThreads();

#endif //__MCML_GETERO_H
