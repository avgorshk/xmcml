#include "function.h"

#include <math.h>

void GenerateSurfaceFtomFunction(double3 center, double lengthX, double lengthY, int scaleSizeX, int scaleSizeY, double (*inputFunction)(double , double), Surface* fSurface)
{
	fSurface->numberOfVertices = (scaleSizeX + 1) * (scaleSizeY + 1);
	fSurface->vertices = new double3[fSurface->numberOfVertices];
	double3 corner;
	double stepX = lengthX / scaleSizeX, stepY = lengthY / scaleSizeY;
	corner.x = center.x - lengthX / 2.0;
	corner.y = center.y - lengthY / 2.0;
	corner.z = center.z;

	for(int i = 0; i < scaleSizeX + 1; i++)
	{
		for(int j = 0; j < scaleSizeY + 1; j++)
		{
			fSurface->vertices[i * (scaleSizeX + 1) + j].x = corner.x + i * stepX;
			fSurface->vertices[i * (scaleSizeX + 1) + j].y = corner.y + j * stepY;
			fSurface->vertices[i * (scaleSizeX + 1) + j].z = corner.z + inputFunction(fSurface->vertices[i * (scaleSizeX + 1) + j].x, fSurface->vertices[i * (scaleSizeX + 1) + j].y);
		}
	}
	fSurface->numberOfTriangles = scaleSizeX * scaleSizeY * 2;
	fSurface->triangles = new int3[fSurface->numberOfTriangles];

	for(int i = 0; i < scaleSizeX; i++)
	{
		for(int j = 0; j < scaleSizeY; j++)
		{
			fSurface->triangles[(i * scaleSizeX + j) * 2].x = i * (scaleSizeX + 1) + j;
			fSurface->triangles[(i * scaleSizeX + j) * 2].y = i * (scaleSizeX + 1) + j + 1;
			fSurface->triangles[(i * scaleSizeX + j) * 2].z = (i + 1) * (scaleSizeX + 1) + j;

			fSurface->triangles[(i * scaleSizeX + j) * 2 + 1].x = i * (scaleSizeX + 1) + j + 1;
			fSurface->triangles[(i * scaleSizeX + j) * 2 + 1].y = (i + 1) * (scaleSizeX + 1) + j + 1;
			fSurface->triangles[(i * scaleSizeX + j) * 2 + 1].z = (i + 1) * (scaleSizeX + 1) + j;
		}
	}
}