#include "cylinder.h"

#include <math.h>

#define PI  3.14159265358979323846
#define EPS 1.0E-6

int GetVertexIndex(int circleId, int dividingId, int numberOfSubdividings)
{
    return circleId*numberOfSubdividings + dividingId;
}

void GenerateCylinderYZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder)
{
    if (numberOfSubdividings < 3)
        numberOfSubdividings = 3;

    cylinder->numberOfVertices = numberOfSubdividings*(numberOfSubdividings + 1);
    cylinder->vertices = new double3[cylinder->numberOfVertices];
    cylinder->numberOfTriangles = 2*numberOfSubdividings*numberOfSubdividings;
    cylinder->triangles = new int3[cylinder->numberOfTriangles];

    int vertexId = 0;
    double currentLength = 0;
    double lengthStep = length/numberOfSubdividings;
    double circleStep = 2*PI/numberOfSubdividings;
    for (int i = 0; i < numberOfSubdividings + 1; ++i)
    {
        double phi = -PI/2;
        while (phi < (3*PI/2 - EPS))
        {
            cylinder->vertices[vertexId].x = center.x - length/2 + currentLength;
            cylinder->vertices[vertexId].y = center.y + radius*sin(phi);
            cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
            phi += circleStep;
            ++vertexId;
        }

        currentLength += lengthStep;
    }

    int triagleId = 0;
    for (int circleId = 0; circleId < numberOfSubdividings; ++circleId)
    {
        for (int dividingId = 0; dividingId < numberOfSubdividings - 1; ++dividingId)
        {
            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, dividingId + 0, numberOfSubdividings);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings);
            ++triagleId;

            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 1, numberOfSubdividings);
            ++triagleId;
        }

        cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, 0, numberOfSubdividings);
        cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, numberOfSubdividings - 1, numberOfSubdividings);
        cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, numberOfSubdividings - 1, numberOfSubdividings);
        ++triagleId;

        cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, 0, numberOfSubdividings);
        cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, numberOfSubdividings - 1, numberOfSubdividings);
        cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, 0, numberOfSubdividings);
        ++triagleId;
    }
}

void GenerateCylinderXZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder)
{
    if (numberOfSubdividings < 3)
        numberOfSubdividings = 3;

    cylinder->numberOfVertices = numberOfSubdividings*(numberOfSubdividings + 1);
    cylinder->vertices = new double3[cylinder->numberOfVertices];
    cylinder->numberOfTriangles = 2*numberOfSubdividings*numberOfSubdividings;
    cylinder->triangles = new int3[cylinder->numberOfTriangles];

    int vertexId = 0;
    double currentLength = 0;
    double lengthStep = length/numberOfSubdividings;
    double circleStep = 2*PI/numberOfSubdividings;
    for (int i = 0; i < numberOfSubdividings + 1; ++i)
    {
        double phi = -PI/2;
        while (phi < (3*PI/2 - EPS))
        {
            cylinder->vertices[vertexId].x = center.x + radius*sin(phi);
            cylinder->vertices[vertexId].y = center.y - length/2 + currentLength;
            cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
            phi += circleStep;
            ++vertexId;
        }

        currentLength += lengthStep;
    }

    int triagleId = 0;
    for (int circleId = 0; circleId < numberOfSubdividings; ++circleId)
    {
        for (int dividingId = 0; dividingId < numberOfSubdividings - 1; ++dividingId)
        {
            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, dividingId + 0, numberOfSubdividings);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings);
            ++triagleId;

            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 1, numberOfSubdividings);
            ++triagleId;
        }

        cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, 0, numberOfSubdividings);
        cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, numberOfSubdividings - 1, numberOfSubdividings);
        cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, numberOfSubdividings - 1, numberOfSubdividings);
        ++triagleId;

        cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, 0, numberOfSubdividings);
        cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, numberOfSubdividings - 1, numberOfSubdividings);
        cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, 0, numberOfSubdividings);
        ++triagleId;
    }
}

void GenerateTopHalfCylinderYZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder)
{
    if (numberOfSubdividings < 1)
        numberOfSubdividings = 1;

    cylinder->numberOfVertices = (numberOfSubdividings + 2)*(numberOfSubdividings + 1);
    cylinder->vertices = new double3[cylinder->numberOfVertices];
    cylinder->numberOfTriangles = 2*(numberOfSubdividings + 1)*numberOfSubdividings;
    cylinder->triangles = new int3[cylinder->numberOfTriangles];

    int vertexId = 0;
    double currentLength = 0;
    double lengthStep = length/numberOfSubdividings;
    double circleStep = PI/(numberOfSubdividings + 1);
    for (int i = 0; i < numberOfSubdividings + 1; ++i)
    {
        double phi = -PI/2;
        while (phi < (PI/2 - EPS))
        {
            cylinder->vertices[vertexId].x = center.x - length/2 + currentLength;
            cylinder->vertices[vertexId].y = center.y + radius*sin(phi);
            cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
            phi += circleStep;
            ++vertexId;
        }

        phi = PI/2;
        cylinder->vertices[vertexId].x = center.x - length/2 + currentLength;
        cylinder->vertices[vertexId].y = center.y + radius*sin(phi);
        cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
        ++vertexId;

        currentLength += lengthStep;
    }

    int triagleId = 0;
    for (int circleId = 0; circleId < numberOfSubdividings; ++circleId)
    {
        for (int dividingId = 0; dividingId < numberOfSubdividings + 1; ++dividingId)
        {
            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            ++triagleId;

            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 1, numberOfSubdividings + 2);
            ++triagleId;
        }
    }
}

void GenerateTopHalfCylinderXZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder)
{
    if (numberOfSubdividings < 1)
        numberOfSubdividings = 1;

    cylinder->numberOfVertices = (numberOfSubdividings + 2)*(numberOfSubdividings + 1);
    cylinder->vertices = new double3[cylinder->numberOfVertices];
    cylinder->numberOfTriangles = 2*(numberOfSubdividings + 1)*numberOfSubdividings;
    cylinder->triangles = new int3[cylinder->numberOfTriangles];

    int vertexId = 0;
    double currentLength = 0;
    double lengthStep = length/numberOfSubdividings;
    double circleStep = PI/(numberOfSubdividings + 1);
    for (int i = 0; i < numberOfSubdividings + 1; ++i)
    {
        double phi = -PI/2;
        while (phi < (PI/2 - EPS))
        {
            cylinder->vertices[vertexId].x = center.x + radius*sin(phi);
            cylinder->vertices[vertexId].y = center.y - length/2 + currentLength;
            cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
            phi += circleStep;
            ++vertexId;
        }

        phi = PI/2;
        cylinder->vertices[vertexId].x = center.x + radius*sin(phi);
        cylinder->vertices[vertexId].y = center.y - length/2 + currentLength;
        cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
        ++vertexId;

        currentLength += lengthStep;
    }

    int triagleId = 0;
    for (int circleId = 0; circleId < numberOfSubdividings; ++circleId)
    {
        for (int dividingId = 0; dividingId < numberOfSubdividings + 1; ++dividingId)
        {
            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            ++triagleId;

            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 1, numberOfSubdividings + 2);
            ++triagleId;
        }
    }
}

void GenerateBottomHalfCylinderYZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder)
{
    if (numberOfSubdividings < 1)
        numberOfSubdividings = 1;

    cylinder->numberOfVertices = (numberOfSubdividings + 2)*(numberOfSubdividings + 1);
    cylinder->vertices = new double3[cylinder->numberOfVertices];
    cylinder->numberOfTriangles = 2*(numberOfSubdividings + 1)*numberOfSubdividings;
    cylinder->triangles = new int3[cylinder->numberOfTriangles];

    int vertexId = 0;
    double currentLength = 0;
    double lengthStep = length/numberOfSubdividings;
    double circleStep = PI/(numberOfSubdividings + 1);
    for (int i = 0; i < numberOfSubdividings + 1; ++i)
    {
        double phi = PI/2;
        while (phi < (3*PI/2 - EPS))
        {
            cylinder->vertices[vertexId].x = center.x - length/2 + currentLength;
            cylinder->vertices[vertexId].y = center.y + radius*sin(phi);
            cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
            phi += circleStep;
            ++vertexId;
        }

        phi = 3*PI/2;
        cylinder->vertices[vertexId].x = center.x - length/2 + currentLength;
        cylinder->vertices[vertexId].y = center.y + radius*sin(phi);
        cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
        ++vertexId;

        currentLength += lengthStep;
    }

    int triagleId = 0;
    for (int circleId = 0; circleId < numberOfSubdividings; ++circleId)
    {
        for (int dividingId = 0; dividingId < numberOfSubdividings + 1; ++dividingId)
        {
            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            ++triagleId;

            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 1, numberOfSubdividings + 2);
            ++triagleId;
        }
    }
}

void GenerateBottomHalfCylinderXZ(double3 center, double radius, double length, int numberOfSubdividings, Surface* cylinder)
{
    if (numberOfSubdividings < 1)
        numberOfSubdividings = 1;

    cylinder->numberOfVertices = (numberOfSubdividings + 2)*(numberOfSubdividings + 1);
    cylinder->vertices = new double3[cylinder->numberOfVertices];
    cylinder->numberOfTriangles = 2*(numberOfSubdividings + 1)*numberOfSubdividings;
    cylinder->triangles = new int3[cylinder->numberOfTriangles];

    int vertexId = 0;
    double currentLength = 0;
    double lengthStep = length/numberOfSubdividings;
    double circleStep = PI/(numberOfSubdividings + 1);
    for (int i = 0; i < numberOfSubdividings + 1; ++i)
    {
        double phi = PI/2;
        while (phi < (3*PI/2 - EPS))
        {
            cylinder->vertices[vertexId].x = center.x + radius*sin(phi);
            cylinder->vertices[vertexId].y = center.y - length/2 + currentLength;
            cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
            phi += circleStep;
            ++vertexId;
        }

        phi = 3*PI/2;
        cylinder->vertices[vertexId].x = center.x + radius*sin(phi);
        cylinder->vertices[vertexId].y = center.y - length/2 + currentLength;
        cylinder->vertices[vertexId].z = center.z + radius*cos(phi);
        ++vertexId;

        currentLength += lengthStep;
    }

    int triagleId = 0;
    for (int circleId = 0; circleId < numberOfSubdividings; ++circleId)
    {
        for (int dividingId = 0; dividingId < numberOfSubdividings + 1; ++dividingId)
        {
            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 0, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            ++triagleId;

            cylinder->triangles[triagleId].x = GetVertexIndex(circleId + 0, dividingId + 1, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].y = GetVertexIndex(circleId + 1, dividingId + 0, numberOfSubdividings + 2);
            cylinder->triangles[triagleId].z = GetVertexIndex(circleId + 1, dividingId + 1, numberOfSubdividings + 2);
            ++triagleId;
        }
    }
}
