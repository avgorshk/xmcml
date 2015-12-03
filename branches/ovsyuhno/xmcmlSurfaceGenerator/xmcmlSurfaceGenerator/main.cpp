#include <stdio.h>
#include <string.h>
#include <math.h>

#include "plane.h"
#include "sphere.h"
#include "quasisphere.h"
#include "circle.h"
#include "cylinder.h"
#include "function.h"

#include "writer.h"

#define STR_LENGTH 255
#define PI  3.14159265358979323846

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

//int main(int argc, char* argv[])
//{
//    Surface surface[7];
//    double3 center = {0.0, 0.0, 0.0};
//    double length = 200.0;
//    
//    center.z = 0.0; //outer medium
//    GeneratePlane(center, length, length, &(surface[0]));
//    center.z = 2.1; //skin
//    GeneratePlane(center, length, length, &(surface[1]));
//    center.z = 5.3; //fat
//    GeneratePlane(center, length, length, &(surface[2]));
//    center.z = 12.2; //skull
//    GeneratePlane(center, length, length, &(surface[3]));
//    center.z = 14.7; //CSF
//    GeneratePlane(center, length, length, &(surface[4]));
//    center.z = 20.5; //grey matter
//    GeneratePlane(center, length, length, &(surface[5]));
//    center.z = 80.5; //white matter
//    GeneratePlane(center, length, length, &(surface[6]));
//
//    int result;
//    result = WriteOutputToFile(surface, 7, "brain_plane_7l.surface");
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//Intralipid with air
//int main(int argc, char* argv[])
//{
//    Surface surface[15];
//    
//    double length = 60;
//    double radius = 2;
//    double intralipid_length = 4;
//    double shift = 0;
//
//    double3 center = {0.0, 0.0, 0.0};
//    GeneratePlane(center, length, length, &(surface[0]));
//    
//    center.z += 3;
//    GeneratePlane(center, length, length, &(surface[1]));
//
//    center.z += 10;
//    GeneratePlane(center, length, length, &(surface[2]));
//
//    center.z += 2;
//    GeneratePlane(center, length, length, &(surface[3]));
//
//    center.z += 4;
//    
//    double3 center_left = center;
//    center_left.x = (-length/2 - radius)/2;
//    GeneratePlane(center_left, length/2 - radius, length, &(surface[4]));
//
//    double3 center_right = center;
//    center_right.x = (length/2 + radius)/2;
//    GeneratePlane(center_right, length/2 - radius, length, &(surface[5]));
//
//    double3 center_6 = center;
//    center_6.y = (-intralipid_length/2 - length/2 + shift)/2;
//    GenerateBottomHalfCylinderXZ(center_6, radius, length/2 - intralipid_length/2 + shift, 10, &(surface[6]));
//    GenerateTopHalfCylinderXZ(center_6, radius, length/2 - intralipid_length/2 + shift, 10, &(surface[7]));
//
//    double3 center_8 = center;
//    center_8.y += shift;
//    GenerateBottomHalfCylinderXZ(center_8, radius, intralipid_length, 10, &(surface[8]));
//    GenerateTopHalfCylinderXZ(center_8, radius, intralipid_length, 10, &(surface[9]));
//
//    double3 center_10 = center;
//    center_10.y = (intralipid_length/2 + length/2 + shift)/2;
//    GenerateBottomHalfCylinderXZ(center_10, radius, length/2 - intralipid_length/2 - shift, 10, &(surface[10]));
//    GenerateTopHalfCylinderXZ(center_10, radius, length/2 - intralipid_length/2 - shift, 10, &(surface[11]));
//
//    double3 center_12 = center;
//    center_12.y = -intralipid_length/2 + shift;
//    GenerateCircleXZ(center_12, radius, 22, &(surface[12]));
//
//    double3 center_13 = center;
//    center_13.y = intralipid_length/2 + shift;
//    GenerateCircleXZ(center_13, radius, 22, &(surface[13]));
//
//    center.z += 20;
//    GeneratePlane(center, length, length, &(surface[14]));
//
//    int result;
//    char fileName[256];
//    sprintf(fileName, "intralipid_%d.surface", (int)shift);
//    result = WriteOutputToFile(surface, 15, fileName);
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//Intralipid without air (19 mm)
//int main(int argc, char* argv[])
//{
//    Surface surface[9];
//    
//    double length = 60;
//    double radius = 2;
//
//    double3 center = {0.0, 0.0, 0.0};
//    GeneratePlane(center, length, length, &(surface[0]));
//    
//    center.z += 3;
//    GeneratePlane(center, length, length, &(surface[1]));
//
//    center.z += 10;
//    GeneratePlane(center, length, length, &(surface[2]));
//
//    center.z += 2;
//    GeneratePlane(center, length, length, &(surface[3]));
//
//    center.z += 4;
//    
//    double3 center_left = center;
//    center_left.x = (-length/2 - radius)/2;
//    GeneratePlane(center_left, length/2 - radius, length, &(surface[4]));
//
//    double3 center_right = center;
//    center_right.x = (length/2 + radius)/2;
//    GeneratePlane(center_right, length/2 - radius, length, &(surface[5]));
//
//    GenerateBottomHalfCylinderXZ(center, radius, length, 10, &(surface[6]));
//    GenerateTopHalfCylinderXZ(center, radius, length, 10, &(surface[7]));
//
//    center.z += 20;
//    GeneratePlane(center, length, length, &(surface[8]));
//
//    int result;
//    result = WriteOutputToFile(surface, 9, "../simple_tube.surface");
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//Intralipid without air (6 mm)
//int main(int argc, char* argv[])
//{
//    const int numberOfSurfaces = 8;
//    Surface surface[numberOfSurfaces];
//    
//    double length = 60;
//    double radius = 2;
//
//    double3 center = {0.0, 0.0, 0.0};
//    GeneratePlane(center, length, length, &(surface[0]));
//    
//    center.z += 3;
//    GeneratePlane(center, length, length, &(surface[1]));
//
//    center.z += 10;
//    GeneratePlane(center, length, length, &(surface[2]));
//
//    center.z += 2;
//    GeneratePlane(center, length, length, &(surface[3]));
//
//    center.z += 4;
//    GeneratePlane(center, length, length, &(surface[4]));
//
//    center.z += 20;
//    GeneratePlane(center, length, length, &(surface[5]));
//
//    center.z = 6;
//    GenerateBottomHalfCylinderXZ(center, radius, length, 10, &(surface[6]));
//    GenerateTopHalfCylinderXZ(center, radius, length, 10, &(surface[7]));
//
//    int result;
//    result = WriteOutputToFile(surface, numberOfSurfaces, "../simple_tube_6mm.surface");
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//Intralipid without air (10 mm)
//int main(int argc, char* argv[])
//{
//    const int numberOfSurfaces = 8;
//    Surface surface[numberOfSurfaces];
//    
//    double length = 60;
//    double radius = 2;
//
//    double3 center = {0.0, 0.0, 0.0};
//    GeneratePlane(center, length, length, &(surface[0]));
//    
//    center.z += 3;
//    GeneratePlane(center, length, length, &(surface[1]));
//
//    center.z += 10;
//    GeneratePlane(center, length, length, &(surface[2]));
//
//    center.z += 2;
//    GeneratePlane(center, length, length, &(surface[3]));
//
//    center.z += 4;
//    GeneratePlane(center, length, length, &(surface[4]));
//
//    center.z += 20;
//    GeneratePlane(center, length, length, &(surface[5]));
//
//    center.z = 10;
//    GenerateBottomHalfCylinderXZ(center, radius, length, 10, &(surface[6]));
//    GenerateTopHalfCylinderXZ(center, radius, length, 10, &(surface[7]));
//
//    int result;
//    result = WriteOutputToFile(surface, numberOfSurfaces, "../simple_tube_10mm.surface");
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//Intralipid without air (13 mm)
//int main(int argc, char* argv[])
//{
//    Surface surface[9];
//    
//    double length = 60;
//    double radius = 1.9;
//
//    double3 center = {0.0, 0.0, 0.0};
//    GeneratePlane(center, length, length, &(surface[0]));
//    
//    center.z += 3;
//    GeneratePlane(center, length, length, &(surface[1]));
//
//    center.z += 10;
//    
//    double3 center_left = center;
//    center_left.x = (-length/2 - radius)/2;
//    GeneratePlane(center_left, length/2 - radius, length, &(surface[2]));
//
//    double3 center_right = center;
//    center_right.x = (length/2 + radius)/2;
//    GeneratePlane(center_right, length/2 - radius, length, &(surface[3]));
//
//    GenerateBottomHalfCylinderXZ(center, radius, length, 10, &(surface[4]));
//    GenerateTopHalfCylinderXZ(center, radius, length, 10, &(surface[5]));
//
//    center.z += 2;
//    GeneratePlane(center, length, length, &(surface[6]));
//
//    center.z += 4;
//    GeneratePlane(center, length, length, &(surface[7]));
//
//    center.z += 20;
//    GeneratePlane(center, length, length, &(surface[8]));
//
//    int result;
//    result = WriteOutputToFile(surface, 9, "../simple_tube_13mm.surface");
//    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
//
//    return 0;
//}

//Intralipid without air (15 mm)

double Function1(double x, double y)
{
	/*double tmp;
	x += 5.25;
	x = x * 2.0;
	x = modf(x, &tmp);
	x = x / 2.0 - 0.25;

	y += 5.25;
	y = y * 2.0;
	y = modf(y, &tmp);
	y = y / 2.0 - 0.25;
	if(x * x + y * y > 0.0625)
		return 0.0;
	return (-0.021 / (cos((sqrt((x * x) + (y * y)) - 0.25) * (4 * PI)) + 1.1) + 0.01);*/
	return sin(x) * sin(y);
}

int main(int argc, char* argv[])
{
    Surface surface[3];
    
    double length = 50;
    double radius = 1.9;
	int size = 200;

    double3 center = {0, 0, 0};
    GeneratePlane(center, length, length, &(surface[0]));
    
    center.z += 0.2;
    GeneratePlane(center, length, length, &(surface[1]));

    center.z += 0.8;
    GeneratePlane(center, length, length, &(surface[2]));

	/*double (*pFunc)(double , double) = Function1;
	GenerateSurfaceFtomFunction(center, length, length, size, size, pFunc, &(surface[0]));*/
	
	/*center.z += 0.5;
    GeneratePlane(center, length, length, &(surface[1]));

    center.z += 40;
    GeneratePlane(center, length, length, &(surface[2]));*/

    int result;
    result = WriteOutputToFile(surface, 3, "../skin14.surface");
    printf("Writing data to file...%s\n", (result == -1) ? "FALSE" : "OK");
	
    return 0;
}