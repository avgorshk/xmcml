#include "McmlIntersectionTestCase.h"

CPPUNIT_TEST_SUITE_REGISTRATION(McmlIntersectionTestCase);

#define MAX_DISTANCE 1.0E+256
#define MIN_DISTANCE 1.0E-12

const double McmlIntersectionTestCase::delta = 1.0E-7;

void McmlIntersectionTestCase::setUp()
{
}

void McmlIntersectionTestCase::tearDown()
{
}

void McmlIntersectionTestCase::assertVectorEqual(double3 expected, double3 actual, double delta)
{
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, actual.x, delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, actual.y, delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, actual.z, delta);
}

//GetTriangleIntersectionDistance==============================================

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance1()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 a = {2.0, -1.0, -1.0};
	double3 b = {2.0, 1.0, -1.0};
	double3 c = {2.0, 0.0, 2.0};

	double distance = GetTriangleIntersectionDistance(origin, direction, a, b, c);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance2()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 a = {0.0, -1.0, -1.0};
	double3 b = {0.0, 1.0, -1.0};
	double3 c = {2.25, 0.0, 1.0};

	double distance = GetTriangleIntersectionDistance(origin, direction, a, b, c);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.125, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance3()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {-1.0, 0.0, 0.0};

	double3 a = {2.0, -1.0, -1.0};
	double3 b = {2.0, 1.0, -1.0};
	double3 c = {2.0, 0.0, 2.0};

	double distance = GetTriangleIntersectionDistance(origin, direction, a, b, c);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(-2.0, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance4()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 a = {0.0, -1.0, -1.0};
	double3 b = {0.0, 1.0, -1.0};
	double3 c = {0.0, 0.0, 1.0};

	double distance = GetTriangleIntersectionDistance(origin, direction, a, b, c);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance5()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 a = {1.0, 0.0, 5.0};
	double3 b = {0.0, 1.0, 5.0};
	double3 c = {0.0, 0.0, 5.0};

	double distance = GetTriangleIntersectionDistance(origin, direction, a, b, c);

	CPPUNIT_ASSERT(distance >= MAX_DISTANCE);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance6()
{
	double3 origin = {0.0, 3.0, -1.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 a = {2.0, -1.0, -1.0};
	double3 b = {2.0, 1.0, -1.0};
	double3 c = {2.0, 0.0, 2.0};

	double distance = GetTriangleIntersectionDistance(origin, direction, a, b, c);

	CPPUNIT_ASSERT(distance >= MAX_DISTANCE);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance7()
{
	double3 origin = {0.0, 1.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 a = {2.0, -1.0, -1.0};
	double3 b = {2.0, 1.0, -1.0};
	double3 c = {2.0, 0.0, 2.0};

	double distance = GetTriangleIntersectionDistance(origin, direction, a, b, c);

	CPPUNIT_ASSERT(distance >= MAX_DISTANCE);
}

//END GetTriangleIntersectionDistance==========================================

//ComputeSurfaceIntersection===================================================

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection1()
{
    Surface surface;
    surface.numberOfVertices = 3;
    surface.vertices = new double3[surface.numberOfVertices];
    surface.numberOfTriangles = 1;
    surface.triangles = new int3[surface.numberOfTriangles];

    double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {0.0, -1.0, -1.0};
	surface.vertices[0] = temp1;
	double3 temp2 = {0.0, 1.0, -1.0};
	surface.vertices[1] = temp2;
	double3 temp3 = {2.0, 0.0, 1.0};
	surface.vertices[2] = temp3;

    int3 triangle = {0, 1, 2};
    surface.triangles[0] = triangle;

    double3 normal = {4.0, 0.0, -4.0};
    normal = NormalizeVector(normal);

    BVHTree* bvhTree = GenerateBVHTree(&surface, 1);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, &surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
    delete[] surface.triangles;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection2()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];
    surface.numberOfTriangles = 4;
    surface.triangles = new int3[surface.numberOfTriangles];

    double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {4.0, 1.0, -1.0};
	surface.vertices[0] = temp1;
	double3 temp2 = {4.0, -1.0, -1.0};
	surface.vertices[4] = temp2;
	double3 temp3 = {4.0, 0.0, 1.0};
	surface.vertices[2] = temp3;
    double3 temp4 = {2.0, -1.0, -1.0};
    surface.vertices[3] = temp4;
    double3 temp5 = {2.0, 0.0, 1.0};
    surface.vertices[1] = temp5;
    double3 temp6 = {2.0, 1.0, -1.0};
    surface.vertices[5] = temp6;

    int3 triangle1 = {0, 4, 2};
    surface.triangles[0] = triangle1;
    int3 triangle2 = {4, 2, 3};
    surface.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 1};
    surface.triangles[2] = triangle3;
    int3 triangle4 = {3, 1, 5};
    surface.triangles[3] = triangle4;

    double3 normal = {-4.0, 0.0, 0.0};
    normal = NormalizeVector(normal);

    BVHTree* bvhTree = GenerateBVHTree(&surface, 1);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, &surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
    delete[] surface.triangles;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection3()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];
    surface.numberOfTriangles = 4;
    surface.triangles = new int3[surface.numberOfTriangles];

    double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {-1.0, 0.0, 0.0};

    double3 temp1 = {4.0, 1.0, -1.0};
    surface.vertices[0] = temp1;
    double3 temp2 = {4.0, -1.0, -1.0};
    surface.vertices[3] = temp2;    
    double3 temp3 = {4.0, 0.0, 1.0};
    surface.vertices[2] = temp3;
    double3 temp4 = {2.0, -1.0, -1.0};
    surface.vertices[1] = temp4;
    double3 temp5 = {2.0, 0.0, 1.0};
    surface.vertices[4] = temp5;
    double3 temp6 = {2.0, 1.0, -1.0};
    surface.vertices[5] = temp6;

    int3 triangle1 = {0, 3, 2};
    surface.triangles[0] = triangle1;
    int3 triangle2 = {3, 2, 1};
    surface.triangles[1] = triangle2;
    int3 triangle3 = {2, 1, 4};
    surface.triangles[2] = triangle3;
    int3 triangle4 = {1, 4, 5};
    surface.triangles[3] = triangle4;

    BVHTree* bvhTree = GenerateBVHTree(&surface, 1);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, &surface);

    CPPUNIT_ASSERT_EQUAL(0, result.isFindIntersection);

    delete[] surface.vertices;
    delete[] surface.triangles;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection4()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];
    surface.numberOfTriangles = 4;
    surface.triangles = new int3[surface.numberOfTriangles];

    double3 origin = {2.0 + MIN_DISTANCE, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {4.0, 1.0, -1.0};
	surface.vertices[0] = temp1;
	double3 temp2 = {4.0, -1.0, -1.0};
	surface.vertices[3] = temp2;
	double3 temp3 = {4.0, 0.0, 1.0};
	surface.vertices[2] = temp3;
    double3 temp4 = {2.0, -1.0, -1.0};
    surface.vertices[1] = temp4;
    double3 temp5 = {2.0, 0.0, 1.0};
    surface.vertices[4] = temp5;
    double3 temp6 = {2.0, 1.0, -1.0};
    surface.vertices[5] = temp6;

    int3 triangle1 = {0, 3, 2};
    surface.triangles[0] = triangle1;
    int3 triangle2 = {3, 2, 1};
    surface.triangles[1] = triangle2;
    int3 triangle3 = {2, 1, 4};
    surface.triangles[2] = triangle3;
    int3 triangle4 = {1, 4, 5};
    surface.triangles[3] = triangle4;

    double3 normal = {-4.0, 0.0, 0.0};
    normal = NormalizeVector(normal);

    BVHTree* bvhTree = GenerateBVHTree(&surface, 1);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, &surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
    delete[] surface.triangles;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection5()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];
    surface.numberOfTriangles = 4;
    surface.triangles = new int3[surface.numberOfTriangles];

    double3 origin = {2.0 - MIN_DISTANCE, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {4.0, 1.0, -1.0};
	surface.vertices[0] = temp1;
	double3 temp2 = {4.0, -1.0, -1.0};
	surface.vertices[1] = temp2;
	double3 temp3 = {4.0, 0.0, 1.0};
	surface.vertices[2] = temp3;
    double3 temp4 = {2.0, -1.0, -1.0};
    surface.vertices[3] = temp4;
    double3 temp5 = {2.0, 0.0, 1.0};
    surface.vertices[4] = temp5;
    double3 temp6 = {2.0, 1.0, -1.0};
    surface.vertices[5] = temp6;

    int3 triangle1 = {0, 1, 2};
    surface.triangles[0] = triangle1;
    int3 triangle2 = {1, 2, 3};
    surface.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 4};
    surface.triangles[2] = triangle3;
    int3 triangle4 = {3, 4, 5};
    surface.triangles[3] = triangle4;

    double3 normal = {-4.0, 0.0, 0.0};
    normal = NormalizeVector(normal);

    BVHTree* bvhTree = GenerateBVHTree(&surface, 1);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, &surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
    delete[] surface.triangles;
}

//END ComputeSurfaceIntersection===============================================

//ComputeIntersection==========================================================

void McmlIntersectionTestCase::IsCorrectComputeIntersection1()
{
    double3 origin = {0.0, 0.0, 0.0};
    double3 direction = {1.0, 0.0, 0.0};

    Surface surface1;
    surface1.numberOfVertices = 6;
    surface1.vertices = new double3[surface1.numberOfVertices];
    surface1.numberOfTriangles = 4;
    surface1.triangles = new int3[surface1.numberOfTriangles];

    double3 temp11 = {4.0, 1.0, -1.0};
    surface1.vertices[0] = temp11;
    double3 temp12 = {4.0, -1.0, -1.0};
    surface1.vertices[1] = temp12;
    double3 temp13 = {4.0, 0.0, 1.0};
    surface1.vertices[2] = temp13;
    double3 temp14 = {2.0, -1.0, -1.0};
    surface1.vertices[3] = temp14;
    double3 temp15 = {2.0, 0.0, 1.0};
    surface1.vertices[4] = temp15;
    double3 temp16 = {2.0, 1.0, -1.0};
    surface1.vertices[5] = temp16;

    int3 triangle1 = {0, 1, 2};
    surface1.triangles[0] = triangle1;
    int3 triangle2 = {1, 2, 3};
    surface1.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 4};
    surface1.triangles[2] = triangle3;
    int3 triangle4 = {3, 4, 5};
    surface1.triangles[3] = triangle4;

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface1.numberOfVertices];
    surface2.numberOfTriangles = 4;
    surface2.triangles = surface1.triangles;

    double3 temp21 = {8.0, 1.0, -1.0};
    surface2.vertices[0] = temp21;
    double3 temp22 = {8.0, -1.0, -1.0};
    surface2.vertices[1] = temp22;
    double3 temp23 = {8.0, 0.0, 1.0};
    surface2.vertices[2] = temp23;
    double3 temp24 = {6.0, -1.0, -1.0};
    surface2.vertices[3] = temp24;
    double3 temp25 = {6.0, 0.0, 1.0};
    surface2.vertices[4] = temp25;
    double3 temp26 = {6.0, 1.0, -1.0};
    surface2.vertices[5] = temp26;

    double3 normal = {-1.0, 0.0, 0.0};

    Surface* surfaces = new Surface[2];
    surfaces[0] = surface1;
    surfaces[1] = surface2;

    BVHTree* bvhTree = GenerateBVHTree(surfaces, 2);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, surfaces);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface1.vertices;
    delete[] surface2.vertices;
    delete[] surface1.triangles;
    delete[] surfaces;
}

void McmlIntersectionTestCase::IsCorrectComputeIntersection2()
{
    double3 origin = {11.0, 0.0, 0.0};
    double3 direction = {-1.0, 0.0, 0.0};

    Surface surface1;
    surface1.numberOfVertices = 6;
    surface1.vertices = new double3[surface1.numberOfVertices];
    surface1.numberOfTriangles = 4;
    surface1.triangles = new int3[surface1.numberOfTriangles];

    double3 temp11 = {4.0, 1.0, -1.0};
    surface1.vertices[0] = temp11;
    double3 temp12 = {4.0, -1.0, -1.0};
    surface1.vertices[1] = temp12;
    double3 temp13 = {4.0, 0.0, 1.0};
    surface1.vertices[2] = temp13;
    double3 temp14 = {2.0, -1.0, -1.0};
    surface1.vertices[3] = temp14;
    double3 temp15 = {2.0, 0.0, 1.0};
    surface1.vertices[4] = temp15;
    double3 temp16 = {2.0, 1.0, -1.0};
    surface1.vertices[5] = temp16;

    int3 triangle1 = {0, 1, 2};
    surface1.triangles[0] = triangle1;
    int3 triangle2 = {1, 2, 3};
    surface1.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 4};
    surface1.triangles[2] = triangle3;
    int3 triangle4 = {3, 4, 5};
    surface1.triangles[3] = triangle4;

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface1.numberOfVertices];
    surface2.numberOfTriangles = 4;
    surface2.triangles = surface1.triangles;

    double3 temp21 = {8.0, 1.0, -1.0};
    surface2.vertices[0] = temp21;
    double3 temp22 = {8.0, -1.0, -1.0};
    surface2.vertices[1] = temp22;
    double3 temp23 = {8.0, 0.0, 1.0};
    surface2.vertices[2] = temp23;
    double3 temp24 = {6.0, -1.0, -1.0};
    surface2.vertices[3] = temp24;
    double3 temp25 = {6.0, 0.0, 1.0};
    surface2.vertices[4] = temp25;
    double3 temp26 = {6.0, 1.0, -1.0};
    surface2.vertices[5] = temp26;

    double3 normal = {-1.0, 0.0, 0.0};

    Surface* surfaces = new Surface[2];
    surfaces[0] = surface1;
    surfaces[1] = surface2;

    BVHTree* bvhTree = GenerateBVHTree(surfaces, 2);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, surfaces);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface1.vertices;
    delete[] surface2.vertices;
    delete[] surface1.triangles;
    delete[] surfaces;
}

void McmlIntersectionTestCase::IsCorrectComputeIntersection3()
{
    double3 origin = {5.0, 0.0, 0.0};
    double3 direction = {0.0, 1.0, 0.0};

    Surface surface1;
    surface1.numberOfVertices = 6;
    surface1.vertices = new double3[surface1.numberOfVertices];
    surface1.numberOfTriangles = 4;
    surface1.triangles = new int3[surface1.numberOfTriangles];

    double3 temp11 = {4.0, 1.0, -1.0};
    surface1.vertices[0] = temp11;
    double3 temp12 = {4.0, -1.0, -1.0};
    surface1.vertices[1] = temp12;
    double3 temp13 = {4.0, 0.0, 1.0};
    surface1.vertices[2] = temp13;
    double3 temp14 = {2.0, -1.0, -1.0};
    surface1.vertices[3] = temp14;
    double3 temp15 = {2.0, 0.0, 1.0};
    surface1.vertices[4] = temp15;
    double3 temp16 = {2.0, 1.0, -1.0};
    surface1.vertices[5] = temp16;

    int3 triangle1 = {0, 1, 2};
    surface1.triangles[0] = triangle1;
    int3 triangle2 = {1, 2, 3};
    surface1.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 4};
    surface1.triangles[2] = triangle3;
    int3 triangle4 = {3, 4, 5};
    surface1.triangles[3] = triangle4;

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface1.numberOfVertices];
    surface2.numberOfTriangles = 4;
    surface2.triangles = surface1.triangles;

    double3 temp21 = {8.0, 1.0, -1.0};
    surface2.vertices[0] = temp21;
    double3 temp22 = {8.0, -1.0, -1.0};
    surface2.vertices[1] = temp22;
    double3 temp23 = {8.0, 0.0, 1.0};
    surface2.vertices[2] = temp23;
    double3 temp24 = {6.0, -1.0, -1.0};
    surface2.vertices[3] = temp24;
    double3 temp25 = {6.0, 0.0, 1.0};
    surface2.vertices[4] = temp25;
    double3 temp26 = {6.0, 1.0, -1.0};
    surface2.vertices[5] = temp26;

    Surface* surfaces = new Surface[2];
    surfaces[0] = surface1;
    surfaces[1] = surface2;

    BVHTree* bvhTree = GenerateBVHTree(surfaces, 2);
    IntersectionInfo result = ComputeBVHIntersectionWithoutStep(origin, direction,
        bvhTree, surfaces);

    CPPUNIT_ASSERT_EQUAL(0, result.isFindIntersection);

    delete[] surface1.vertices;
    delete[] surface2.vertices;
    delete[] surface1.triangles;
    delete[] surfaces;
}

//END ComputeIntersection======================================================