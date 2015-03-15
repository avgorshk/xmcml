#include <stdio.h>

#include "plane.h"
#include "sphere.h"

#include "writer.h"

#define STR_LENGTH 255

int main(int argc, char* argv[])
{
    Surface surface[6];
    double3 center = {0.0, 0.0, 80.0};
    double radius;
    int numberOfSubdividings = 32;
    
    radius = 80.0;
    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[0]));
    radius = 77.0;
    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[1]));
    radius = 67.0;
    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[2]));
    radius = 65.0;
    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[3]));
    radius = 61.0;
    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[4]));

    double length = 200.0;
    GeneratePlane(center, length, length, &(surface[5]));

    int result;
    char fileName[STR_LENGTH];
    sprintf(fileName, "brain_sphere_%d.surface", numberOfSubdividings * (numberOfSubdividings - 1));
    result = WriteOutputToFile(surface, 6, fileName);
    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");

    return 0;
}

//int main(int argc, char* argv[])
//{
//    Surface surface[6];
//    double3 center = {0.0, 0.0, 0.0};
//    double length = 200.0;
//    
//    center.z = 0.0;
//    GeneratePlane(center, length, length, &(surface[0]));
//    center.z = 3.0;
//    GeneratePlane(center, length, length, &(surface[1]));
//    center.z = 13.0;
//    GeneratePlane(center, length, length, &(surface[2]));
//    center.z = 15.0;
//    GeneratePlane(center, length, length, &(surface[3]));
//    center.z = 19.0;
//    GeneratePlane(center, length, length, &(surface[4]));
//    center.z = 80.0;
//    GeneratePlane(center, length, length, &(surface[5]));
//
//    int result;
//    result = WriteOutputToFile(surface, 6, "brain_plane.surface");
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}