#include "quasisphere.h"
#include "sphere.h"

#include <stdlib.h>

double randomDelta(double delta)
{
    double result = (double)rand() / RAND_MAX;
    result = 2.0 * delta * result - delta;
    return result;
}

void GenerateBottomHalfQuasiSphere(double3 center, double radius, int numberOfSubdividings, 
    double delta, Surface* quasiSphere)
{
    GenerateBottomHalfSphere(center, radius,  numberOfSubdividings, quasiSphere);

    for (int i = 0; i < quasiSphere->numberOfVertices; ++i)
    {
        if (rand() % 5 == 0)
        {
            quasiSphere->vertices[i].z += randomDelta(delta);
        }
    }
}
