#include <stdio.h>

#include "plane.h"
#include "sphere.h"

#include "writer.h"

int main(int argc, char* argv[])
{
    Surface surface[6];
    double3 center = {0.0, 0.0, 39.0};
    double radius;

    radius = 39.0;
    GenerateBottomHalfSphere(center, radius, 32, &(surface[0]));
    radius = 36.0;
    GenerateBottomHalfSphere(center, radius, 32, &(surface[1]));
    radius = 26.0;
    GenerateBottomHalfSphere(center, radius, 32, &(surface[2]));
    radius = 24.0;
    GenerateBottomHalfSphere(center, radius, 32, &(surface[3]));
    radius = 20.0;
    GenerateBottomHalfSphere(center, radius, 32, &(surface[4]));

    double length = 100.0;
    GeneratePlane(center, length, length, &(surface[5]));

    int result;
    result = WriteOutputToFile(surface, 6, "brain_sphere_992.surface");
    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");

    return 0;
}