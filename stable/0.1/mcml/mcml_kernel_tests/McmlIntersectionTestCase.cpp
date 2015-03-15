#include "McmlIntersectionTestCase.h"

CPPUNIT_TEST_SUITE_REGISTRATION(McmlIntersectionTestCase);

#define MAX_DISTANCE 1.0E+256
#define MIN_DISTANCE 1.0E-12

const double McmlIntersectionTestCase::delta = 1.0E-7;

void McmlIntersectionTestCase::setUp()
{
    vertices = new double3[3];
}

void McmlIntersectionTestCase::tearDown()
{
    delete[] vertices;
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

	double3 temp1 = {2.0, -1.0, -1.0};
	vertices[0] = temp1;
	double3 temp2 = {2.0, 1.0, -1.0};
	vertices[1] = temp2;
	double3 temp3 = {2.0, 0.0, 2.0};
	vertices[2] = temp3;

	double distance = GetTriangleIntersectionDistance(origin, direction, vertices);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance2()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {0.0, -1.0, -1.0};
	vertices[0] = temp1;
	double3 temp2 = {0.0, 1.0, -1.0};
	vertices[1] = temp2;
	double3 temp3 = {2.25, 0.0, 1.0};
	vertices[2] = temp3;

	double distance = GetTriangleIntersectionDistance(origin, direction, vertices);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.125, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance3()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {-1.0, 0.0, 0.0};

	double3 temp1 = {2.0, -1.0, -1.0};
	vertices[0] = temp1;
	double3 temp2 = {2.0, 1.0, -1.0};
	vertices[1] = temp2;
	double3 temp3 = {2.0, 0.0, 2.0};
	vertices[2] = temp3;

	double distance = GetTriangleIntersectionDistance(origin, direction, vertices);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(-2.0, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance4()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {0.0, -1.0, -1.0};
	vertices[0] = temp1;
	double3 temp2 = {0.0, 1.0, -1.0};
	vertices[1] = temp2;
	double3 temp3 = {0.0, 0.0, 1.0};
	vertices[2] = temp3;

	double distance = GetTriangleIntersectionDistance(origin, direction, vertices);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, distance, delta);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance5()
{
	double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {1.0, 0.0, 5.0};
	vertices[0] = temp1;
	double3 temp2 = {0.0, 1.0, 5.0};
	vertices[1] = temp2;
	double3 temp3 = {0.0, 0.0, 5.0};
	vertices[2] = temp3;

	double distance = GetTriangleIntersectionDistance(origin, direction, vertices);

	CPPUNIT_ASSERT(distance >= MAX_DISTANCE);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance6()
{
	double3 origin = {0.0, 3.0, -1.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {2.0, -1.0, -1.0};
	vertices[0] = temp1;
	double3 temp2 = {2.0, 1.0, -1.0};
	vertices[1] = temp2;
	double3 temp3 = {2.0, 0.0, 2.0};
	vertices[2] = temp3;

	double distance = GetTriangleIntersectionDistance(origin, direction, vertices);

	CPPUNIT_ASSERT(distance >= MAX_DISTANCE);
}

void McmlIntersectionTestCase::IsCorrectTriangleIntersectionDistance7()
{
	double3 origin = {0.0, 1.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {2.0, -1.0, -1.0};
	vertices[0] = temp1;
	double3 temp2 = {2.0, 1.0, -1.0};
	vertices[1] = temp2;
	double3 temp3 = {2.0, 0.0, 2.0};
	vertices[2] = temp3;

	double distance = GetTriangleIntersectionDistance(origin, direction, vertices);

	CPPUNIT_ASSERT(distance >= MAX_DISTANCE);
}

//END GetTriangleIntersectionDistance==========================================

//ComputeSurfaceIntersection===================================================

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection1()
{
    Surface surface;
    surface.numberOfVertices = 3;
    surface.vertices = new double3[surface.numberOfVertices];

    double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

	double3 temp1 = {0.0, -1.0, -1.0};
	surface.vertices[0] = temp1;
	double3 temp2 = {0.0, 1.0, -1.0};
	surface.vertices[1] = temp2;
	double3 temp3 = {2.0, 0.0, 1.0};
	surface.vertices[2] = temp3;

    double3 normal = {4.0, 0.0, -4.0};

    IntersectionInfo result = ComputeSurfaceIntersection(origin, direction, surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection2()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];

    double3 origin = {0.0, 0.0, 0.0};
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

    double3 normal = {-4.0, 0.0, 0.0};

    IntersectionInfo result = ComputeSurfaceIntersection(origin, direction, surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection3()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];

    double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {-1.0, 0.0, 0.0};

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

    IntersectionInfo result = ComputeSurfaceIntersection(origin, direction, surface);

    CPPUNIT_ASSERT_EQUAL(0, result.isFindIntersection);

    delete[] surface.vertices;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection4()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];

    double3 origin = {2.0 + MIN_DISTANCE, 0.0, 0.0};
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

    double3 normal = {-4.0, 0.0, 0.0};

    IntersectionInfo result = ComputeSurfaceIntersection(origin, direction, surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
}

void McmlIntersectionTestCase::IsCorrectComputeSurfaceIntersection5()
{
    Surface surface;
    surface.numberOfVertices = 6;
    surface.vertices = new double3[surface.numberOfVertices];

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

    double3 normal = {-4.0, 0.0, 0.0};

    IntersectionInfo result = ComputeSurfaceIntersection(origin, direction, surface);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface.vertices;
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

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface1.numberOfVertices];

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

    IntersectionInfo result = ComputeIntersection(origin, direction, surfaces, 2);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface1.vertices;
    delete[] surface2.vertices;
    delete[] surfaces;
}

void McmlIntersectionTestCase::IsCorrectComputeIntersection2()
{
    double3 origin = {11.0, 0.0, 0.0};
	double3 direction = {-1.0, 0.0, 0.0};

    Surface surface1;
    surface1.numberOfVertices = 6;
    surface1.vertices = new double3[surface1.numberOfVertices];

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

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface1.numberOfVertices];

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

    IntersectionInfo result = ComputeIntersection(origin, direction, surfaces, 2);

    CPPUNIT_ASSERT_EQUAL(1, result.isFindIntersection);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, result.distance, delta);
    assertVectorEqual(normal, result.normal, delta);

    delete[] surface1.vertices;
    delete[] surface2.vertices;
    delete[] surfaces;
}

void McmlIntersectionTestCase::IsCorrectComputeIntersection3()
{
    double3 origin = {5.0, 0.0, 0.0};
	double3 direction = {0.0, 1.0, 0.0};

    Surface surface1;
    surface1.numberOfVertices = 6;
    surface1.vertices = new double3[surface1.numberOfVertices];

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

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface1.numberOfVertices];

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

    IntersectionInfo result = ComputeIntersection(origin, direction, surfaces, 2);

    CPPUNIT_ASSERT_EQUAL(0, result.isFindIntersection);

    delete[] surface1.vertices;
    delete[] surface2.vertices;
    delete[] surfaces;
}

//END ComputeIntersection======================================================