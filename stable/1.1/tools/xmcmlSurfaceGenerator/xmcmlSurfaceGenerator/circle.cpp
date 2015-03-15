#include "circle.h"

#include <math.h>

#define PI  3.14159265358979323846
#define EPS 1.0E-6

void GenerateCircleYZ(double3 center, double radius, int numberOfSubdividings, Surface* circle)
{
    if (numberOfSubdividings < 3)
        numberOfSubdividings = 3;
    
    circle->numberOfVertices = numberOfSubdividings + 1;
    circle->vertices = new double3[circle->numberOfVertices];
    circle->numberOfTriangles = numberOfSubdividings;
    circle->triangles = new int3[circle->numberOfTriangles];

    double step = 2*PI/numberOfSubdividings;
    double phi = -PI/2;
    int vertexId = 0;
    
    circle->vertices[vertexId] = center;
    while (phi < (3*PI/2 - EPS))
    {
        ++vertexId;
        circle->vertices[vertexId].x = center.x;
        circle->vertices[vertexId].y = center.y + radius*sin(phi);
        circle->vertices[vertexId].z = center.z + radius*cos(phi);
        phi += step;
    }

    for (int i = 0; i < circle->numberOfTriangles - 1; ++i)
    {
        circle->triangles[i].x = i + 1;
        circle->triangles[i].y = 0;
        circle->triangles[i].z = i + 2;
    }
    circle->triangles[circle->numberOfTriangles - 1].x = circle->numberOfVertices - 1;
    circle->triangles[circle->numberOfTriangles - 1].y = 0;
    circle->triangles[circle->numberOfTriangles - 1].z = 1;
}

void GenerateCircleXZ(double3 center, double radius, int numberOfSubdividings, Surface* circle)
{
    if (numberOfSubdividings < 3)
        numberOfSubdividings = 3;
    
    circle->numberOfVertices = numberOfSubdividings + 1;
    circle->vertices = new double3[circle->numberOfVertices];
    circle->numberOfTriangles = numberOfSubdividings;
    circle->triangles = new int3[circle->numberOfTriangles];

    double step = 2*PI/numberOfSubdividings;
    double phi = -PI/2;
    int vertexId = 0;
    
    circle->vertices[vertexId] = center;
    while (phi < (3*PI/2 - EPS))
    {
        ++vertexId;
        circle->vertices[vertexId].x = center.x + radius*sin(phi);
        circle->vertices[vertexId].y = center.y;
        circle->vertices[vertexId].z = center.z + radius*cos(phi);
        phi += step;
    }

    for (int i = 0; i < circle->numberOfTriangles - 1; ++i)
    {
        circle->triangles[i].x = i + 1;
        circle->triangles[i].y = 0;
        circle->triangles[i].z = i + 2;
    }
    circle->triangles[circle->numberOfTriangles - 1].x = circle->numberOfVertices - 1;
    circle->triangles[circle->numberOfTriangles - 1].y = 0;
    circle->triangles[circle->numberOfTriangles - 1].z = 1;
}