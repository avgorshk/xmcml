#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "test_cuda.h"
#include "mcml_intersection.h"
#include "parser.h"
#include "portable_time.h"
#include <iostream>
#include <omp.h>

using namespace std;

void Test(char* fileName, int n)
{
    double start_time, finish_time;

    Surface* surface = NULL;
    int numberOfSurfaces = 0;

    floatVec3* origin = new floatVec3[n];
    floatVec3* direction = new floatVec3[n];
	float* step = new float[n];
    
    bool isOk = true;
    isOk = ParseSurfaceFile(fileName, surface, numberOfSurfaces);
    if (!isOk)
    {
        printf("Test %s FAILED: file not found\n", fileName);
        return;
    }

    printf("Test %s results:\n", fileName);

    for (int i = 0; i < n; ++i)
    {
        origin[i].x = ((float)rand() / RAND_MAX) * 160.f - 80.f;
        origin[i].y = ((float)rand() / RAND_MAX) * 160.f - 80.f;
        origin[i].z = ((float)rand() / RAND_MAX) * 80.f;

        direction[i].x = ((float)rand() / RAND_MAX) - 0.5f;
        direction[i].y = ((float)rand() / RAND_MAX) - 0.5f;
        direction[i].z = ((float)rand() / RAND_MAX) - 0.5f;

		step[i] = ((float)rand() / RAND_MAX) * 0.5f;
    }

    IntersectionInfo* info_cpu = new IntersectionInfo[n];
    IntersectionInfo* info_gpu = new IntersectionInfo[n];

    BVHTree* tree = 0;
	start_time = PortableGetTime();
	tree = GenerateBVHTree(surface, numberOfSurfaces);
	finish_time = PortableGetTime();
	printf("GenerateBVHTree: %f sec\n", finish_time - start_time);
    printf("Triangles count: %d \n", tree->numOfTriangles);

    start_time = PortableGetTime();
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        info_cpu[i] = ComputeBVHIntersection(origin[i], direction[i], step[i], tree, surface);
    }
    finish_time = PortableGetTime();
    printf("CPU BVH (with step): %f sec\n", finish_time - start_time);	

    float computeTime = 0;
    float fullTime = 0;
	GpuBVHIntersections(origin, direction, step, n, tree, surface, numberOfSurfaces, info_gpu, computeTime, fullTime);
    printf("GPU BVH (with step): %f sec\n", fullTime / 1000.f);
    printf("GPU compute time: %f sec\n", computeTime / 1000.f);

	delete tree;

    int passed = 0, failed = 0;
	floatVec3 &v1 = info_cpu[0].normal, &v2 = info_gpu[0].normal;
    for (int i = 0; i < n; ++i)
    {
        bool isMethodsEqual = true;
        if (info_cpu[i].isFindIntersection != info_gpu[i].isFindIntersection)
		{
            isMethodsEqual = false;
		}
		if (info_cpu[i].isFindIntersection) 
		{            
			if (info_cpu[i].surfaceId != info_gpu[i].surfaceId)
            {
				isMethodsEqual = false;
            }
			if (fabs(info_cpu[i].distance - info_gpu[i].distance) > EPSILON)
            {
				isMethodsEqual = false;
            }
			v1 = info_cpu[i].normal;
			v2 = info_gpu[i].normal;
			if ((fabs(v1.x-v2.x)>EPSILON || fabs(v1.y-v2.y)>EPSILON || fabs(v1.z-v2.z)>EPSILON) &&
				(fabs(v1.x+v2.x)>EPSILON || fabs(v1.y+v2.y)>EPSILON || fabs(v1.z+v2.z)>EPSILON))
            {
				isMethodsEqual = false;
            }
		}      

        if (isMethodsEqual)
            ++passed;
        else
            ++failed;
    }
    printf("%d all tests; %d passed; %d failed\n", n, passed, failed);

    delete[] surface;
    delete[] step;
    delete[] origin;
    delete[] direction;

    delete[] info_cpu;
    delete[] info_gpu;
}

void main()
{
    srand((unsigned int)time(0));
    
    Test("surfaces\\human_head_scaled.surface", 10000000);
    printf("\n");
    Test("surfaces\\sphere_992.surface", 10000000);
    printf("\n");
    Test("surfaces\\sphere_9900.surface", 10000000);
    printf("\n");
    Test("surfaces\\sphere_999000.surface", 10000000);
    printf("\n");
}