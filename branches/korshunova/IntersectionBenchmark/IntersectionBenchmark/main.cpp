#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mcml_intersection.h"
#include "parser.h"
#include "portable_time.h"

void Test(char* fileName, int n)
{
    double start_time, finish_time;

    Surface* surface = NULL;
    int numberOfSurfaces = 0;

    double3* origin = new double3[n];
    double3* direction = new double3[n];
	double* step = new double[n];
    
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
        origin[i].x = ((double)rand() / RAND_MAX) * 160.0 - 80.0;
        origin[i].y = ((double)rand() / RAND_MAX) * 160.0 - 80.0;
        origin[i].z = ((double)rand() / RAND_MAX) * 80.0;

        direction[i].x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        direction[i].y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        direction[i].z = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

		step[i] = ((double)rand() / RAND_MAX);
		//step[i] = MAX_DISTANCE;
    }

    IntersectionInfo* info_simple = new IntersectionInfo[n];
    IntersectionInfo* info_bvh = new IntersectionInfo[n];
	IntersectionInfo* info_kd = new IntersectionInfo[n];

    start_time = PortableGetTime();
    for (int i = 0; i < n; ++i)
    {
        //Простейший метод, не использует информацию о длине вектора
        info_simple[i] = ComputeIntersection(origin[i], direction[i],
            surface, numberOfSurfaces); 
		//if (info_simple[i].distance > step[i])
			//info_simple[i].isFindIntersection = 0;
    }
    finish_time = PortableGetTime();
    printf("Simple method: %f sec\n", finish_time - start_time);

	BVHTree* tree = 0;
	start_time = PortableGetTime();
	tree = GenerateBVHTree(surface, numberOfSurfaces);
	finish_time = PortableGetTime();
	printf("GenerateBVHTree: %f sec\n", finish_time - start_time);

    /*start_time = PortableGetTime();	
    for (int i = 0; i < n; ++i)
    {
		info_bvh[i] = ComputeBVHIntersection(origin[i], direction[i], step[i], tree, surface); 
    }
    finish_time = PortableGetTime();
    printf("BVH method (step): %f sec\n", finish_time - start_time);*/

	start_time = PortableGetTime();	
    for (int i = 0; i < n; ++i)
    {
		info_bvh[i] = ComputeBVHIntersectionWithoutStep(origin[i], direction[i], tree, surface); 
    }
    finish_time = PortableGetTime();
    printf("BVH method (without step): %f sec\n", finish_time - start_time);

	delete tree;

	/*KdTree* kdtree = 0;
	start_time = PortableGetTime();
	kdtree = GenerateKdTree(surface, numberOfSurfaces);
	finish_time = PortableGetTime();
	printf("GenerateKDTree: %f sec\n", finish_time - start_time);

	start_time = PortableGetTime();	
    for (int i = 0; i < n; ++i)
    {
		info_kd[i] = ComputeKDIntersectionWithoutStep(origin[i], direction[i], kdtree, surface); 
    }
    finish_time = PortableGetTime();
    printf("Kd method (without step): %f sec\n", finish_time - start_time);

	start_time = PortableGetTime();	
    for (int i = 0; i < n; ++i)
    {
		info_kd[i] = ComputeKDIntersection(origin[i], direction[i], step[i], kdtree, surface); 
    }
    finish_time = PortableGetTime();
    printf("Kd method (step): %f sec\n", finish_time - start_time);

	delete kdtree;*/

    int passed = 0, failed = 0;
	double3 &v1 = info_simple[0].normal, &v2 = info_bvh[0].normal;
    for (int i = 0; i < n; ++i)
    {
        bool isMethodsEqual = true;
        if (info_simple[i].isFindIntersection != info_bvh[i].isFindIntersection)
		{
            isMethodsEqual = false;
		}
		if (info_simple[i].isFindIntersection) 
		{
			if (info_simple[i].surfaceId != info_bvh[i].surfaceId)
				isMethodsEqual = false;
			if (fabs(info_simple[i].distance - info_bvh[i].distance) > EPSILON)
				isMethodsEqual = false;
			v1 = info_simple[i].normal;
			v2 = info_bvh[i].normal;
			if ((abs(v1.x-v2.x)>EPSILON || abs(v1.y-v2.y)>EPSILON || abs(v1.z-v2.z)>EPSILON) &&
				(abs(v1.x+v2.x)>EPSILON || abs(v1.y+v2.y)>EPSILON || abs(v1.z+v2.z)>EPSILON))
				isMethodsEqual = false;
		}      
        //TODO: compare normals

        if (isMethodsEqual)
            ++passed;
        else
            ++failed;
    }
    printf("%d all tests; %d passed; %d failed\n", n, passed, failed);

	/*int passed = 0, failed = 0;
	double3 &v1 = info_bvh[0].normal, &v2 = info_kd[0].normal;
	for (int i = 0; i < n; ++i)
    {
        bool isMethodsEqual = true;
        if (info_bvh[i].isFindIntersection != info_kd[i].isFindIntersection)
		{
            isMethodsEqual = false;
		}
		if (info_kd[i].isFindIntersection) 
		{
			if (info_bvh[i].surfaceId != info_kd[i].surfaceId)
				isMethodsEqual = false;
			if (fabs(info_bvh[i].distance - info_kd[i].distance) > EPSILON)
				isMethodsEqual = false;
			v1 = info_bvh[i].normal;
			v2 = info_kd[i].normal;
			if ((abs(v1.x-v2.x)>EPSILON || abs(v1.y-v2.y)>EPSILON || abs(v1.z-v2.z)>EPSILON) &&
				(abs(v1.x+v2.x)>EPSILON || abs(v1.y+v2.y)>EPSILON || abs(v1.z+v2.z)>EPSILON))
				isMethodsEqual = false;
		}      
        //TODO: compare normals

        if (isMethodsEqual)
            ++passed;
        else
            ++failed;
    }
    printf("%d all tests; %d passed; %d failed\n", n, passed, failed);*/

    delete[] origin;
    delete[] direction;

    delete[] info_simple;
    delete[] info_bvh;
	delete[] info_kd;
}

void main()
{
    srand((unsigned int)time(0));
    Test("..\\surfaces\\plane.surface", 100);
    printf("\n");
    Test("..\\surfaces\\sphere_992.surface", 100);
    printf("\n");
    Test("..\\surfaces\\sphere_9900.surface", 100);
    printf("\n");
    Test("..\\surfaces\\sphere_999000.surface", 1);
    printf("\n");
}