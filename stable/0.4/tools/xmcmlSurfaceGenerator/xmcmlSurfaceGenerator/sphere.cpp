#include "sphere.h"

#include <vector>

#define PI       3.14159265358979323846
#define EPSILON  1.0E-6

void GenerateVerticeLine(double3 center, double radius, double phi, double thetaStep,
    std::vector<double3>* vertices)
{
    double theta;
    double sinPhi, cosPhi, sinTheta, cosTheta;

    sinPhi = sin(phi);
    cosPhi = cos(phi);

    theta = thetaStep;
    while (theta < (PI - EPSILON))
    {
        sinTheta = sin(theta);
        cosTheta = cos(theta);

        double3 v = {
            center.x + radius * sinTheta * cosPhi, 
            center.y + radius * sinTheta * sinPhi,
            center.z + radius * cosTheta
            };
        vertices->push_back(v);

        theta += thetaStep;
    }
}

void GenerateTriangleLine(int verticeIndex, int verticeIndexNext, int verticeLineSize,
    std::vector<int3>* triangles)
{
    int3 _t = {0, verticeIndex, verticeIndexNext};
    triangles->push_back(_t);
    
    for (int i = 0; i < verticeLineSize - 1; ++i)
    {
        int3 t1 = {verticeIndex + i, verticeIndexNext + i, verticeIndex + i + 1};
        triangles->push_back(t1);
        int3 t2 = {verticeIndexNext + i, verticeIndex + i + 1, verticeIndexNext + i + 1};
        triangles->push_back(t2);
    }

    int3 t_ = {verticeIndex + verticeLineSize - 1, verticeIndexNext + verticeLineSize - 1, 1};
    triangles->push_back(t_);
}

void GenerateSphere(double3 center, double radius, int numberOfSubdividings, Surface* sphere)
{
    double thetaStep;
    double phi, phiStep;
    int verticeIndex, verticeLineSize;

    if (numberOfSubdividings < 3)
        numberOfSubdividings = 3;

    std::vector<double3> vertices;
    std::vector<int3> triangles;
    
    double3 vTop = {center.x, center.y, center.z + radius};
    vertices.push_back(vTop);
    double3 vBottom = {center.x, center.y, center.z - radius};
    vertices.push_back(vBottom);

    thetaStep = PI / numberOfSubdividings;
    phiStep = 2.0 * PI / numberOfSubdividings;
    verticeLineSize = numberOfSubdividings - 1;

    phi = 0;
    GenerateVerticeLine(center, radius, phi, thetaStep, &vertices);

    verticeIndex = 2;
    while (phi < (2.0 * PI - phiStep - EPSILON))
    {
        phi += phiStep;
        GenerateVerticeLine(center, radius, phi, thetaStep, &vertices);
        GenerateTriangleLine(verticeIndex, verticeIndex + verticeLineSize, verticeLineSize, &triangles);
        verticeIndex += verticeLineSize;
    }
    GenerateTriangleLine(verticeIndex, 2, verticeLineSize, &triangles);

    sphere->numberOfVertices = (int)vertices.size();
    sphere->vertices = new double3[sphere->numberOfVertices];
    for (int i = 0; i < sphere->numberOfVertices; ++i)
    {
        sphere->vertices[i] = vertices[i];
    }

    sphere->numberOfTriangles = (int)triangles.size();
    sphere->triangles = new int3[sphere->numberOfTriangles];
    for (int i = 0; i < sphere->numberOfTriangles; ++i)
    {
        sphere->triangles[i] = triangles[i];
    }
}

void GenerateTopHalfVerticeLine(double3 center, double radius, double phi, double thetaStep,
    std::vector<double3>* vertices)
{
    double theta;
    double sinPhi, cosPhi, sinTheta, cosTheta;

    sinPhi = sin(phi);
    cosPhi = cos(phi);

    theta = thetaStep;
    while (theta < (PI / 2.0 - EPSILON))
    {
        sinTheta = sin(theta);
        cosTheta = cos(theta);

        double3 v = {
            center.x + radius * sinTheta * cosPhi, 
            center.y + radius * sinTheta * sinPhi,
            center.z + radius * cosTheta
            };
        vertices->push_back(v);

        theta += thetaStep;
    }

    sinTheta = sin(theta);
    cosTheta = cos(theta);

    double3 v_ = {
        center.x + radius * sinTheta * cosPhi, 
        center.y + radius * sinTheta * sinPhi,
        center.z + radius * cosTheta
        };
    vertices->push_back(v_);
}

void GenerateTopHalfTriangleLine(int verticeIndex, int verticeIndexNext, int verticeLineSize,
    std::vector<int3>* triangles)
{
    int3 _t = {0, verticeIndex, verticeIndexNext};
    triangles->push_back(_t);
    
    for (int i = 0; i < verticeLineSize - 1; ++i)
    {
        int3 t1 = {verticeIndex + i, verticeIndexNext + i, verticeIndex + i + 1};
        triangles->push_back(t1);
        int3 t2 = {verticeIndexNext + i, verticeIndex + i + 1, verticeIndexNext + i + 1};
        triangles->push_back(t2);
    }
}

void GenerateTopHalfSphere(double3 center, double radius, int numberOfSubdividings, Surface* sphere)
{
    double thetaStep;
    double phi, phiStep;
    int verticeIndex, verticeLineSize;

    if (numberOfSubdividings < 4)
        numberOfSubdividings = 4;
    if ((numberOfSubdividings & 1) == 1)
        numberOfSubdividings <<= 1;

    std::vector<double3> vertices;
    std::vector<int3> triangles;
    
    double3 vTop = {center.x, center.y, center.z + radius};
    vertices.push_back(vTop);

    thetaStep = PI / numberOfSubdividings;
    phiStep = 2.0 * PI / numberOfSubdividings;
    verticeLineSize = numberOfSubdividings >> 1;

    phi = 0;
    GenerateTopHalfVerticeLine(center, radius, phi, thetaStep, &vertices);

    verticeIndex = 1;
    while (phi < (2.0 * PI - phiStep - EPSILON))
    {
        phi += phiStep;
        GenerateTopHalfVerticeLine(center, radius, phi, thetaStep, &vertices);
        GenerateTopHalfTriangleLine(verticeIndex, verticeIndex + verticeLineSize, verticeLineSize, &triangles);
        verticeIndex += verticeLineSize;
    }
    GenerateTopHalfTriangleLine(verticeIndex, 1, verticeLineSize, &triangles);

    sphere->numberOfVertices = (int)vertices.size();
    sphere->vertices = new double3[sphere->numberOfVertices];
    for (int i = 0; i < sphere->numberOfVertices; ++i)
    {
        sphere->vertices[i] = vertices[i];
    }

    sphere->numberOfTriangles = (int)triangles.size();
    sphere->triangles = new int3[sphere->numberOfTriangles];
    for (int i = 0; i < sphere->numberOfTriangles; ++i)
    {
        sphere->triangles[i] = triangles[i];
    }
}

void GenerateBottomHalfVerticeLine(double3 center, double radius, double phi, double thetaStep,
    std::vector<double3>* vertices)
{
    double theta;
    double sinPhi, cosPhi, sinTheta, cosTheta;

    sinPhi = sin(phi);
    cosPhi = cos(phi);

    theta = PI / 2.0;
    sinTheta = sin(theta);
    cosTheta = cos(theta);

    double3 _v = {
        center.x + radius * sinTheta * cosPhi, 
        center.y + radius * sinTheta * sinPhi,
        center.z + radius * cosTheta
        };
    vertices->push_back(_v);

    while (theta < (PI - thetaStep - EPSILON))
    {
        theta += thetaStep;

        sinTheta = sin(theta);
        cosTheta = cos(theta);

        double3 v = {
            center.x + radius * sinTheta * cosPhi, 
            center.y + radius * sinTheta * sinPhi,
            center.z + radius * cosTheta
            };
        vertices->push_back(v);
    }
}

void GenerateBottomHalfTriangleLine(int verticeIndex, int verticeIndexNext, int verticeLineSize,
    std::vector<int3>* triangles)
{
    for (int i = 0; i < verticeLineSize - 1; ++i)
    {
        int3 t1 = {verticeIndex + i, verticeIndexNext + i, verticeIndex + i + 1};
        triangles->push_back(t1);
        int3 t2 = {verticeIndexNext + i, verticeIndex + i + 1, verticeIndexNext + i + 1};
        triangles->push_back(t2);
    }

    int3 t_ = {verticeIndex + verticeLineSize - 1, verticeIndexNext + verticeLineSize - 1, 0};
    triangles->push_back(t_);
}

void GenerateBottomHalfSphere(double3 center, double radius, int numberOfSubdividings, Surface* sphere)
{
    double thetaStep;
    double phi, phiStep;
    int verticeIndex, verticeLineSize;

    if (numberOfSubdividings < 4)
        numberOfSubdividings = 4;
    if ((numberOfSubdividings & 1) == 1)
        numberOfSubdividings <<= 1;

    std::vector<double3> vertices;
    std::vector<int3> triangles;
    
    double3 vBottom = {center.x, center.y, center.z - radius};
    vertices.push_back(vBottom);

    thetaStep = PI / numberOfSubdividings;
    phiStep = 2.0 * PI / numberOfSubdividings;
    verticeLineSize = numberOfSubdividings >> 1;

    phi = 0;
    GenerateBottomHalfVerticeLine(center, radius, phi, thetaStep, &vertices);

    verticeIndex = 1;
    while (phi < (2.0 * PI - phiStep - EPSILON))
    {
        phi += phiStep;
        GenerateBottomHalfVerticeLine(center, radius, phi, thetaStep, &vertices);
        GenerateBottomHalfTriangleLine(verticeIndex, verticeIndex + verticeLineSize, verticeLineSize, &triangles);
        verticeIndex += verticeLineSize;
    }
    GenerateBottomHalfTriangleLine(verticeIndex, 1, verticeLineSize, &triangles);

    sphere->numberOfVertices = (int)vertices.size();
    sphere->vertices = new double3[sphere->numberOfVertices];
    for (int i = 0; i < sphere->numberOfVertices; ++i)
    {
        sphere->vertices[i] = vertices[i];
    }

    sphere->numberOfTriangles = (int)triangles.size();
    sphere->triangles = new int3[sphere->numberOfTriangles];
    for (int i = 0; i < sphere->numberOfTriangles; ++i)
    {
        sphere->triangles[i] = triangles[i];
    }
}
