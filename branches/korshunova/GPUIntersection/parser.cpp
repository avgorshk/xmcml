#include "parser.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MCML_SECTION_SURFACES 0x0100

static bool ReadFloat3(FILE* file, floatVec3* vector)
{
    unsigned long long int read_items;
	double item;

    read_items = fread(&item, sizeof(double), 1, file);
    if (read_items < 1)
        return false;
	vector->x = (float)item;

    read_items = fread(&item, sizeof(double), 1, file);
    if (read_items < 1)
        return false;
	vector->y = (float)item;

    read_items = fread(&item, sizeof(double), 1, file);
    if (read_items < 1)
        return false;
	vector->z = (float)item;

    return true;
}

static bool ReadInt3(FILE* file, int3* vector)
{
    unsigned long long int read_items;

    read_items = fread(&(vector->x), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    read_items = fread(&(vector->y), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    read_items = fread(&(vector->z), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    return true;
}

static bool ReadSurface(FILE* file, Surface* surface)
{
    unsigned long long int read_items;

    read_items = fread(&(surface->numberOfVertices), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    surface->vertices = new floatVec3[surface->numberOfVertices];
    for (int i = 0; i < surface->numberOfVertices; ++i)
    {
        if (!ReadFloat3(file, &(surface->vertices[i])))
            return false;
    }

    read_items = fread(&(surface->numberOfTriangles), sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    surface->triangles = new int3[surface->numberOfTriangles];
    for (int i = 0; i < surface->numberOfTriangles; ++i)
    {
        if (!ReadInt3(file, &(surface->triangles[i])))
            return false;
    }

    return true;
}

bool ParseSurfaceFile(char* fileName, Surface* &surface, int &numberOfSurfaces)
{
    FILE* file = NULL;

    fopen_s(&file, fileName, "rb");
    if (file == NULL)
        return false;
    
    unsigned long long int read_items;

    unsigned int section_id;
    read_items = fread(&section_id, sizeof(unsigned int), 1, file);
    if (read_items < 1 || section_id != MCML_SECTION_SURFACES)
        return false;

    read_items = fread(&numberOfSurfaces, sizeof(int), 1, file);
    if (read_items < 1)
        return false;

    surface = new Surface[numberOfSurfaces];
    for (int i = 0; i < numberOfSurfaces; ++i)
    {
        if (!ReadSurface(file, &(surface[i])))
            return false;
    }

    fclose(file);

    return true;
}
