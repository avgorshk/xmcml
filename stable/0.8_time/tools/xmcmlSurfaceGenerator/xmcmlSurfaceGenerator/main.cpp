#include <stdio.h>

#include "plane.h"
#include "sphere.h"
#include "quasisphere.h"

#include "writer.h"

#define STR_LENGTH 255

//int main(int argc, char* argv[])
//{
//    Surface surface[6];
//    double3 center = {0.0, 0.0, 80.0};
//    double radius;
//    int numberOfSubdividings = 32;
//    
//    radius = 80.0;
//    GenerateBottomHalfQuasiSphere(center, radius,numberOfSubdividings,  2.5, &(surface[0]));
//    radius = 77.0;
//    GenerateBottomHalfQuasiSphere(center, radius, numberOfSubdividings, 2.5, &(surface[1]));
//    radius = 67.0;
//    GenerateBottomHalfQuasiSphere(center, radius, numberOfSubdividings, 1.5, &(surface[2]));
//    radius = 65.0;
//    GenerateBottomHalfQuasiSphere(center, radius, numberOfSubdividings, 1.5, &(surface[3]));
//    radius = 61.0;
//    GenerateBottomHalfQuasiSphere(center, radius, numberOfSubdividings, 3.5, &(surface[4]));
//
//    double length = 200.0;
//    GeneratePlane(center, length, length, &(surface[5]));
//
//    int result;
//    char fileName[STR_LENGTH];
//    sprintf(fileName, "brain_quasisphere_%d.surface", numberOfSubdividings * (numberOfSubdividings - 1));
//    result = WriteOutputToFile(surface, 6, fileName);
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//int main(int argc, char* argv[])
//{
//    Surface surfaces[2];
//    double3 center = {0.0, 0.0, 0.0};
//    double length = 400.0;
//    
//    center.z = 0.0;
//    GeneratePlane(center, length, length, &(surfaces[0]));
//    center.z = 200.0;
//    GeneratePlane(center, length, length, &(surfaces[1]));
//
//    int result;
//    result = WriteOutputToFile(surfaces, 2, "r_diff.surface");
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//int main(int argc, char* argv[])
//{
//    Surface surface[7];
//    double3 center = {0.0, 0.0, 80.5};
//    double radius;
//    int numberOfSubdividings = 32;
//    
//    radius = 80.5; //skin
//    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[0]));
//    radius = 78.4; //fat
//    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[1]));
//    radius = 75.2; //skull
//    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[2]));
//    radius = 68.3; //CSF
//    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[3]));
//    radius = 65.8; //grey matter
//    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[4]));
//    radius = 60.0; //white matter 
//    GenerateBottomHalfSphere(center, radius, numberOfSubdividings, &(surface[5]));
//
//    double length = 161.0;
//    GeneratePlane(center, length, length, &(surface[6]));
//
//    int result;
//    char fileName[STR_LENGTH];
//    sprintf(fileName, "brain_sphere_%dl_%d.surface", 7, numberOfSubdividings * (numberOfSubdividings - 1));
//    result = WriteOutputToFile(surface, 7, fileName);
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

int main(int argc, char* argv[])
{
    Surface surface[7];
    double3 center = {0.0, 0.0, 0.0};
    double length = 200.0;
    
    center.z = 0.0; //outer medium
    GeneratePlane(center, length, length, &(surface[0]));
    center.z = 2.1; //skin
    GeneratePlane(center, length, length, &(surface[1]));
    center.z = 5.3; //fat
    GeneratePlane(center, length, length, &(surface[2]));
    center.z = 12.2; //skull
    GeneratePlane(center, length, length, &(surface[3]));
    center.z = 14.7; //CSF
    GeneratePlane(center, length, length, &(surface[4]));
    center.z = 20.5; //grey matter
    GeneratePlane(center, length, length, &(surface[5]));
    center.z = 80.5; //white matter
    GeneratePlane(center, length, length, &(surface[6]));

    int result;
    result = WriteOutputToFile(surface, 7, "brain_plane_7l.surface");
    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");

    return 0;
}