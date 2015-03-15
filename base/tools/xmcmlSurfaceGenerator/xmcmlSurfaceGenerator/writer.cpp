#include <stdio.h>
#include <stdlib.h>

#include "writer.h"

#define MCML_SECTION_SURFACES    0x0100

static int WriteDouble3(FILE* file, double3 vector)
{
    unsigned long long int written_items;

    written_items = fwrite(&(vector.x), sizeof(double), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&(vector.y), sizeof(double), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&(vector.z), sizeof(double), 1, file);
    if (written_items < 1)
        return -1;

    return 0;
}

static int WriteInt3(FILE* file, int3 vector)
{
    unsigned long long int written_items;

    written_items = fwrite(&(vector.x), sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&(vector.y), sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&(vector.z), sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    return 0;
}

static int WriteSurface(FILE* file, Surface* surface)
{
    unsigned long long int written_items;

    written_items = fwrite(&(surface->numberOfVertices), sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    for (int i = 0; i < surface->numberOfVertices; ++i)
    {
        if (WriteDouble3(file, surface->vertices[i]) == -1)
            return -1;
    }

    written_items = fwrite(&(surface->numberOfTriangles), sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    for (int i = 0; i < surface->numberOfTriangles; ++i)
    {
        if (WriteInt3(file, surface->triangles[i]) == -1)
            return -1;
    }

    return 0;
}

int WriteOutputToFile(Surface* surface, int numberOfSurfaces, char* fileName)
{
    FILE* file = NULL;

    fopen_s(&file, fileName, "wb");
    if (file == NULL)
        return -1;
    
    unsigned long long int written_items;

    unsigned int section_id = MCML_SECTION_SURFACES;
    written_items = fwrite(&section_id, sizeof(unsigned int), 1, file);
    if (written_items < 1)
        return -1;

    written_items = fwrite(&numberOfSurfaces, sizeof(int), 1, file);
    if (written_items < 1)
        return -1;

    for (int i = 0; i < numberOfSurfaces; ++i)
    {
        if (WriteSurface(file, &(surface[i])) == -1)
            return -1;
    }

    fflush(file);
    fclose(file);

    return 0;
}
