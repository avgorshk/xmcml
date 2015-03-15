#include "plane.h"

void GeneratePlane(double3 center, double lengthX, double lengthY, Surface* plane)
{
    plane->numberOfVertices = 4;
    plane->vertices = new double3[plane->numberOfVertices];
    plane->numberOfTriangles = 2;
    plane->triangles = new int3[plane->numberOfTriangles];

    double3 vertice1 = { center.x - lengthX / 2.0, center.y - lengthY / 2.0, center.z};
    double3 vertice2 = { center.x - lengthX / 2.0, center.y + lengthY / 2.0, center.z};
    double3 vertice3 = { center.x + lengthX / 2.0, center.y - lengthY / 2.0, center.z};
    double3 vertice4 = { center.x + lengthX / 2.0, center.y + lengthY / 2.0, center.z};

    plane->vertices[0] = vertice1;
    plane->vertices[1] = vertice2;
    plane->vertices[2] = vertice3;
    plane->vertices[3] = vertice4;

    plane->numberOfTriangles = 2;
    
    int3 triangle1 = {0, 1, 2};
    int3 triangle2 = {1, 2, 3};

    plane->triangles[0] = triangle1;
    plane->triangles[1] = triangle2;
}