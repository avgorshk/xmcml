#include "matrix_math.h"
#include <memory>
#include <math.h>

#define M_PI       3.14159265358979323846

double calculateScaledCoefficient(InputFuncPar* funcPar, double x)
{
//	double result = funcPar->A * cos(2.0 * M_PI * funcPar->f * x + funcPar->a) + funcPar->A;
	
	double result = funcPar->A * cos(funcPar->f * x + funcPar->a) + funcPar->A;

	if(result > funcPar->A)
		result = 2.0 * funcPar->A;
	else
		result = 0.0;

	return result;
}

void weightNormalization2D(Area* area, double* weight, double scaled)
{
	int size = area->partitionNumber.x * area->partitionNumber.y;

	if(scaled != 0.0)
		scaled *= size;
	else
		scaled = size;

	for(int i = 0; i < size; i++)
		weight[i] /= scaled;
}

void weightNormalization3D(Area* area, double* weight, double scaled)
{
	int size = area->partitionNumber.x * area->partitionNumber.y;

	if(scaled != 0.0)
		scaled *= size;
	else
		scaled = size;

	size *= area->partitionNumber.z;

	for(int i = 0; i < size; i++)
		weight[i] /= scaled;
}

void sumMatrix2D(Area* area, double* resMatrix, double* originMatrix, int offsetRows,
	int offsetColumns, double scaledCoefficient)
{
	int i, j;
	int m, n;
	int rowsBegin, rowsEnd, columnsBegin, columnsEnd;

	m = area->partitionNumber.x;
	n = area->partitionNumber.y;

	if(offsetRows > 0)
	{
		rowsBegin = 0;
		rowsEnd = m - abs(offsetRows);
	}
	else
	{
		rowsBegin = abs(offsetRows);
		rowsEnd = m;
	}

	if(offsetColumns > 0)
	{
		columnsBegin = 0;
		columnsEnd = n - abs(offsetColumns);
	}
	else
	{
		columnsBegin = abs(offsetColumns);
		columnsEnd = n;
	}

	for(i = rowsBegin; i < rowsEnd; i++)
		for(j = columnsBegin; j < columnsEnd; j++)
			resMatrix[(i + offsetRows) * n + j + offsetColumns] += scaledCoefficient * originMatrix[i * n + j];
}

void sumMatrix3D(int mode, Area* area, double* resMatrix, double* originMatrix, int offsetRows,
	int offsetColumns, double scaledCoefficient)
{
	int i, j, k;
	int rowsBegin, rowsEnd, columnsBegin, columnsEnd;
	int tmpOne, tmpTwo;

	int m, n, p;

	m = area->partitionNumber.x;
	n = area->partitionNumber.y;
	p = area->partitionNumber.z;

	switch(mode)
	{
		case MATRIX_3D_MODE_XY:
		{
			tmpOne = m;
			tmpTwo = n;
			break;
		}
		case MATRIX_3D_MODE_XZ:
		{
			tmpOne = m;
			tmpTwo = p;
			break;
		}
		case MATRIX_3D_MODE_YZ:
		{
			tmpOne = n;
			tmpTwo = p;
			break;
		}
	}
	
	if(offsetRows > 0)
	{
		rowsBegin = 0;
		rowsEnd = tmpOne - abs(offsetRows);
	}
	else
	{
		rowsBegin = abs(offsetRows);
		rowsEnd = tmpOne;
	}

	if(offsetColumns > 0)
	{
		columnsBegin = 0;
		columnsEnd = tmpTwo - abs(offsetColumns);
	}
	else
	{
		columnsBegin = abs(offsetColumns);
		columnsEnd = tmpTwo;
	}

	switch(mode)
	{
		case MATRIX_3D_MODE_XY: 
		{
			for(i = rowsBegin; i < rowsEnd; i++)
				for(j = columnsBegin; j < columnsEnd; j++)
					for(k = 0; k < p; k++)
						resMatrix[(i + offsetRows) * n * p + (j + offsetColumns) * p + k] += scaledCoefficient * originMatrix[i * n * p + j * p + k];

			break;
		}
		case MATRIX_3D_MODE_XZ:
		{
			for(i = rowsBegin; i < rowsEnd; i++)
				for(j = 0; j < n; j++)
					for(k = columnsBegin; k < columnsEnd; k++)
						resMatrix[(i + offsetRows) * n * p + j * p + k + offsetColumns] += scaledCoefficient * originMatrix[i * n * p + j * p + k];
			break;
		}
		case MATRIX_3D_MODE_YZ:
		{
			for(i = 0; i < m; i++)
				for(j = rowsBegin; j < rowsEnd; j++)
					for(k = columnsBegin; k < columnsEnd; k++)
						resMatrix[i * n * p + (j + offsetRows) * p + k + offsetColumns] += scaledCoefficient * originMatrix[i * n * p + j * p + k];
			break;
		}
	}
}

void sumMatrix3D(int mode, Area* area, uint64* resMatrix, uint64* originMatrix, int offsetRows,
	int offsetColumns, double scaledCoefficient)
{
	int i, j, k;
	int rowsBegin, rowsEnd, columnsBegin, columnsEnd;
	int tmpOne, tmpTwo;

	int m, n, p;

	m = area->partitionNumber.x;
	n = area->partitionNumber.y;
	p = area->partitionNumber.z;

	switch(mode)
	{
		case MATRIX_3D_MODE_XY:
		{
			tmpOne = m;
			tmpTwo = n;
			break;
		}
		case MATRIX_3D_MODE_XZ:
		{
			tmpOne = m;
			tmpTwo = p;
			break;
		}
		case MATRIX_3D_MODE_YZ:
		{
			tmpOne = n;
			tmpTwo = p;
			break;
		}
	}
	
	if(offsetRows > 0)
	{
		rowsBegin = 0;
		rowsEnd = tmpOne - abs(offsetRows);
	}
	else
	{
		rowsBegin = abs(offsetRows);
		rowsEnd = tmpOne;
	}

	if(offsetColumns > 0)
	{
		columnsBegin = 0;
		columnsEnd = tmpTwo - abs(offsetColumns);
	}
	else
	{
		columnsBegin = abs(offsetColumns);
		columnsEnd = tmpTwo;
	}

	switch(mode)
	{
		case MATRIX_3D_MODE_XY: 
		{
			for(i = rowsBegin; i < rowsEnd; i++)
				for(j = columnsBegin; j < columnsEnd; j++)
					for(k = 0; k < p; k++)
						resMatrix[(i + offsetRows) * n * p + (j + offsetColumns) * p + k] += (uint64)(scaledCoefficient * originMatrix[i * n * p + j * p + k]);

			break;
		}
		case MATRIX_3D_MODE_XZ:
		{
			for(i = rowsBegin; i < rowsEnd; i++)
				for(j = 0; j < n; j++)
					for(k = columnsBegin; k < columnsEnd; k++)
						resMatrix[(i + offsetRows) * n * p + j * p + k + offsetColumns] += (uint64)(scaledCoefficient * originMatrix[i * n * p + j * p + k]);
			break;
		}
		case MATRIX_3D_MODE_YZ:
		{
			for(i = 0; i < m; i++)
				for(j = rowsBegin; j < rowsEnd; j++)
					for(k = columnsBegin; k < columnsEnd; k++)
						resMatrix[i * n * p + (j + offsetRows) * p + k + offsetColumns] += (uint64)(scaledCoefficient * originMatrix[i * n * p + j * p + k]);
			break;
		}
	}
}

void MSL_2D(Area* area, double* resWeight, InputFuncPar* funcPar)
{
	int i, j;

	int m = area->partitionNumber.x;
	int n = area->partitionNumber.y;

	double lengthOnePartX = area->length.x / area->partitionNumber.x;
	double centerFirstX = area->corner.x + lengthOnePartX / 2.0;

	double* originWeight = new double[m * n];

	memcpy(originWeight, resWeight, m * n * sizeof(double));
	memset(resWeight, 0, m * n * sizeof(double));
		
	double x, scaledCoefficient;

	for(i = -m / 2; i <= m / 2; i++)
	{
		x = centerFirstX + lengthOnePartX * i;
		scaledCoefficient = calculateScaledCoefficient(funcPar, x);

		for(j = -n / 2; j <= n / 2; j++)
			sumMatrix2D(area, resWeight, originWeight, i, j, scaledCoefficient);
	}

	weightNormalization2D(area, resWeight, 2.0 * funcPar->A);

	delete[] originWeight;
}

void MSL_3D(Area* area, double* resWeight, InputFuncPar* funcPar)
{
	int i, j;

	int m = area->partitionNumber.x;
	int n = area->partitionNumber.y;
	int p = area->partitionNumber.z;

	double lengthOnePartX = area->length.x / area->partitionNumber.x;
	double centerFirstX = area->corner.x + lengthOnePartX / 2.0;

	double* originWeight = new double[m * n * p];

	memcpy(originWeight, resWeight, m * n * p * sizeof(double));
	memset(resWeight, 0, m * n * p * sizeof(double));
		
	double x, scaledCoefficient;

	for(i = -m / 2; i <= m / 2; i++)
	{
		x = centerFirstX + lengthOnePartX * i;
		scaledCoefficient = calculateScaledCoefficient(funcPar, x);

		for(j = -n / 2; j <= n / 2; j++)
			sumMatrix3D(MATRIX_3D_MODE_XY, area, resWeight, originWeight, i, j, scaledCoefficient);
	}

	weightNormalization3D(area, resWeight, 2.0 * funcPar->A);

	delete[] originWeight;
}