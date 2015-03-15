#include <math.h>
#include "../xmcml/mcml_math.h"
#include "McmlKernelTestCase.h"

#define PI 3.14159265358979323846

CPPUNIT_TEST_SUITE_REGISTRATION(McmlKernelTestCase);

const double McmlKernelTestCase::delta = 1.0E-7;

void McmlKernelTestCase::setUp()
{
    randomGenerator = new MCG59();
    InitMCG59(randomGenerator, 777, 0, 1); // randomValue1 = 0.81700408 randomValue2 = 0.41785875
}

void McmlKernelTestCase::tearDown()
{
    delete randomGenerator;
}

void McmlKernelTestCase::assertVectorEqual(double3 expected, double3 actual, double delta)
{
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, actual.x, delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, actual.y, delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, actual.z, delta);
}

double3 McmlKernelTestCase::getVector(double x, double y, double z)
{
    double3 result = {x, y, z};
    return result;
}

//ComputeSpecularReflectance===================================================

void McmlKernelTestCase::IsCorrectSpecularReflectance1()
{
    int n = 3;
    LayerInfo* layer = new LayerInfo[n];

	layer[0].refractiveIndex = 1.0;

	layer[1].refractiveIndex = 1.4;
	layer[1].absorptionCoefficient = 1.0;
	layer[1].scatteringCoefficient = 1.0;

	layer[2].refractiveIndex = 1.0;

	double specularReflectance = ComputeSpecularReflectance(layer);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.027777778, specularReflectance, delta);

    delete[] layer;
}

void McmlKernelTestCase::IsCorrectSpecularReflectance2()
{
    int n = 3;
    LayerInfo* layer = new LayerInfo[n];

	layer[0].refractiveIndex = 1.0;

	layer[1].refractiveIndex = 2.0;
	layer[1].absorptionCoefficient = 1.0;
	layer[1].scatteringCoefficient = 1.0;

	layer[2].refractiveIndex = 1.0;

	double specularReflectance = ComputeSpecularReflectance(layer);

	CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0 / 9.0, specularReflectance, delta);

    delete[] layer;
}

void McmlKernelTestCase::IsCorrectSpecularReflectanceInGlass1()
{
    int n = 3;
    LayerInfo* layer = new LayerInfo[n];

	layer[0].refractiveIndex = 1.0;

	layer[1].refractiveIndex = 2.0;
	layer[1].absorptionCoefficient = 0.0;
	layer[1].scatteringCoefficient = 0.0;

	layer[2].refractiveIndex = 3.0;

	double specularReflectance = ComputeSpecularReflectance(layer);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / 7.0, specularReflectance, delta);

    delete[] layer;
}

void McmlKernelTestCase::IsCorrectSpecularReflectanceInGlass2()
{
    int n = 3;
    LayerInfo* layer = new LayerInfo[n];

	layer[0].refractiveIndex = 1.0;

	layer[1].refractiveIndex = 1.0;
	layer[1].absorptionCoefficient = 0.0;
	layer[1].scatteringCoefficient = 0.0;

	layer[2].refractiveIndex = 3.0;

	double specularReflectance = ComputeSpecularReflectance(layer);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.25, specularReflectance, delta);

    delete[] layer;
}

//END ComputeSpecularReflectance===============================================

//GetIntersectionInfo==========================================================

void McmlKernelTestCase::IsCorrectGettingIntersectionInfo1()
{
    int m = 3;
    Surface* surfaces = new Surface[m];

    PhotonState* photon = new PhotonState();

    Surface surface1;
    surface1.numberOfVertices = 3;
    surface1.vertices = new double3[surface1.numberOfVertices];
    surface1.numberOfTriangles = 1;
    surface1.triangles = new int3[surface1.numberOfTriangles];
    double3 temp11 = {2.0, -1.0, -1.0};
	surface1.vertices[0] = temp11;
	double3 temp12 = {2.0, 1.0, -1.0};
	surface1.vertices[1] = temp12;
	double3 temp13 = {2.0, 0.0, 2.0};
	surface1.vertices[2] = temp13;
    int3 triangle = {0, 1, 2};
    surface1.triangles[0] = triangle;

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface2.numberOfVertices];
    surface2.numberOfTriangles = 4;
    surface2.triangles = new int3[surface2.numberOfTriangles];
    double3 temp21 = {4.0, 1.0, -1.0};
	surface2.vertices[0] = temp21;
	double3 temp22 = {4.0, -1.0, -1.0};
	surface2.vertices[1] = temp22;
	double3 temp23 = {4.0, 0.0, 1.0};
	surface2.vertices[2] = temp23;
    double3 temp24 = {2.0, -1.0, -1.0};
    surface2.vertices[3] = temp24;
    double3 temp25 = {2.0, 0.0, 1.0};
    surface2.vertices[4] = temp25;
    double3 temp26 = {2.0, 1.0, -1.0};
    surface2.vertices[5] = temp26;
    int3 triangle1 = {0, 1, 2};
    surface2.triangles[0] = triangle1;
    int3 triangle2 = {1, 2, 3};
    surface2.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 4};
    surface2.triangles[2] = triangle3;
    int3 triangle4 = {3, 4, 5};
    surface2.triangles[3] = triangle4;

    Surface surface3;
    surface3.numberOfVertices = 6;
    surface3.vertices = new double3[surface3.numberOfVertices];
    surface3.numberOfTriangles = 4;
    surface3.triangles = surface2.triangles;
    double3 temp31 = {8.0, 1.0, -1.0};
	surface3.vertices[0] = temp31;
	double3 temp32 = {8.0, -1.0, -1.0};
	surface3.vertices[1] = temp32;
	double3 temp33 = {8.0, 0.0, 1.0};
	surface3.vertices[2] = temp33;
    double3 temp34 = {6.0, -1.0, -1.0};
    surface3.vertices[3] = temp34;
    double3 temp35 = {6.0, 0.0, 1.0};
    surface3.vertices[4] = temp35;
    double3 temp36 = {6.0, 1.0, -1.0};
    surface3.vertices[5] = temp36;

    surfaces[0] = surface1;
    surfaces[1] = surface2;
    surfaces[2] = surface3;

    double3 origin = {0.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

    photon->position = origin;
    photon->direction = direction;
    photon->layerId = 1;

    InputInfo input;
    input.surface = surfaces;
    input.layerInfo = new LayerInfo[3];
    input.layerInfo[1].numberOfSurfaces = 2;
    input.layerInfo[1].surfaceId = new int[input.layerInfo[1].numberOfSurfaces];
    input.layerInfo[1].surfaceId[0] = 0;
    input.layerInfo[1].surfaceId[1] = 2;

    IntersectionInfo intersection = GetIntersectionInfo(photon, &input);

    CPPUNIT_ASSERT_EQUAL(1, intersection.isFindIntersection);
    CPPUNIT_ASSERT_EQUAL(2.0, intersection.distance);
    CPPUNIT_ASSERT_EQUAL(0, intersection.surfaceId);

    delete[] surface1.vertices;
    delete[] surface1.triangles;
    delete[] surface2.vertices;
    delete[] surface2.triangles;
    delete[] surface3.vertices;
    delete[] input.layerInfo[1].surfaceId;
    delete[] input.layerInfo;
    delete[] surfaces;
    delete photon;
}

void McmlKernelTestCase::IsCorrectGettingIntersectionInfo2()
{
    int m = 3;
    Surface* surfaces = new Surface[m];

    PhotonState* photon = new PhotonState();

    Surface surface1;
    surface1.numberOfVertices = 3;
    surface1.vertices = new double3[surface1.numberOfVertices];
    surface1.numberOfTriangles = 1;
    surface1.triangles = new int3[surface1.numberOfTriangles];
    double3 temp11 = {2.0, -1.0, -1.0};
	surface1.vertices[0] = temp11;
	double3 temp12 = {2.0, 1.0, -1.0};
	surface1.vertices[1] = temp12;
	double3 temp13 = {2.0, 0.0, 2.0};
	surface1.vertices[2] = temp13;
    int3 triangle = {0, 1, 2};
    surface1.triangles[0] = triangle;

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface2.numberOfVertices];
    surface2.numberOfTriangles = 4;
    surface2.triangles = new int3[surface2.numberOfTriangles];
    double3 temp21 = {4.0, 1.0, -1.0};
	surface2.vertices[0] = temp21;
	double3 temp22 = {4.0, -1.0, -1.0};
	surface2.vertices[1] = temp22;
	double3 temp23 = {4.0, 0.0, 1.0};
	surface2.vertices[2] = temp23;
    double3 temp24 = {2.0, -1.0, -1.0};
    surface2.vertices[3] = temp24;
    double3 temp25 = {2.0, 0.0, 1.0};
    surface2.vertices[4] = temp25;
    double3 temp26 = {2.0, 1.0, -1.0};
    surface2.vertices[5] = temp26;
    int3 triangle1 = {0, 1, 2};
    surface2.triangles[0] = triangle1;
    int3 triangle2 = {1, 2, 3};
    surface2.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 4};
    surface2.triangles[2] = triangle3;
    int3 triangle4 = {3, 4, 5};
    surface2.triangles[3] = triangle4;

    Surface surface3;
    surface3.numberOfVertices = 6;
    surface3.vertices = new double3[surface3.numberOfVertices];
    surface3.numberOfTriangles = 4;
    surface3.triangles = surface2.triangles;
    double3 temp31 = {8.0, 1.0, -1.0};
	surface3.vertices[0] = temp31;
	double3 temp32 = {8.0, -1.0, -1.0};
	surface3.vertices[1] = temp32;
	double3 temp33 = {8.0, 0.0, 1.0};
	surface3.vertices[2] = temp33;
    double3 temp34 = {6.0, -1.0, -1.0};
    surface3.vertices[3] = temp34;
    double3 temp35 = {6.0, 0.0, 1.0};
    surface3.vertices[4] = temp35;
    double3 temp36 = {6.0, 1.0, -1.0};
    surface3.vertices[5] = temp36;

    surfaces[0] = surface1;
    surfaces[1] = surface2;
    surfaces[2] = surface3;

    double3 origin = {7.0, 0.0, 0.0};
	double3 direction = {-1.0, 0.0, 0.0};

    photon->position = origin;
    photon->direction = direction;
    photon->layerId = 1;

    InputInfo input;
    input.surface = surfaces;
    input.layerInfo = new LayerInfo[3];
    input.layerInfo[1].numberOfSurfaces = 2;
    input.layerInfo[1].surfaceId = new int[input.layerInfo[1].numberOfSurfaces];
    input.layerInfo[1].surfaceId[0] = 0;
    input.layerInfo[1].surfaceId[1] = 2;

    IntersectionInfo intersection = GetIntersectionInfo(photon, &input);

    CPPUNIT_ASSERT_EQUAL(1, intersection.isFindIntersection);
    CPPUNIT_ASSERT_EQUAL(1.0, intersection.distance);
    CPPUNIT_ASSERT_EQUAL(2, intersection.surfaceId);

    delete[] surface1.vertices;
    delete[] surface1.triangles;
    delete[] surface2.vertices;
    delete[] surface2.triangles;
    delete[] surface3.vertices;
    delete[] input.layerInfo[1].surfaceId;
    delete[] input.layerInfo;
    delete[] surfaces;
    delete photon;
}

void McmlKernelTestCase::IsCorrectGettingIntersectionInfo3()
{
    int m = 3;
    Surface* surfaces = new Surface[m];

    PhotonState* photon = new PhotonState();

    Surface surface1;
    surface1.numberOfVertices = 3;
    surface1.vertices = new double3[surface1.numberOfVertices];
    surface1.numberOfTriangles = 1;
    surface1.triangles = new int3[surface1.numberOfTriangles];
    double3 temp11 = {2.0, -1.0, -1.0};
	surface1.vertices[0] = temp11;
	double3 temp12 = {2.0, 1.0, -1.0};
	surface1.vertices[1] = temp12;
	double3 temp13 = {2.0, 0.0, 2.0};
	surface1.vertices[2] = temp13;
    int3 triangle = {0, 1, 2};
    surface1.triangles[0] = triangle;

    Surface surface2;
    surface2.numberOfVertices = 6;
    surface2.vertices = new double3[surface2.numberOfVertices];
    surface2.numberOfTriangles = 4;
    surface2.triangles = new int3[surface2.numberOfTriangles];
    double3 temp21 = {4.0, 1.0, -1.0};
	surface2.vertices[0] = temp21;
	double3 temp22 = {4.0, -1.0, -1.0};
	surface2.vertices[1] = temp22;
	double3 temp23 = {4.0, 0.0, 1.0};
	surface2.vertices[2] = temp23;
    double3 temp24 = {2.0, -1.0, -1.0};
    surface2.vertices[3] = temp24;
    double3 temp25 = {2.0, 0.0, 1.0};
    surface2.vertices[4] = temp25;
    double3 temp26 = {2.0, 1.0, -1.0};
    surface2.vertices[5] = temp26;
    int3 triangle1 = {0, 1, 2};
    surface2.triangles[0] = triangle1;
    int3 triangle2 = {1, 2, 3};
    surface2.triangles[1] = triangle2;
    int3 triangle3 = {2, 3, 4};
    surface2.triangles[2] = triangle3;
    int3 triangle4 = {3, 4, 5};
    surface2.triangles[3] = triangle4;

    Surface surface3;
    surface3.numberOfVertices = 6;
    surface3.vertices = new double3[surface3.numberOfVertices];
    surface3.numberOfTriangles = 4;
    surface3.triangles = surface2.triangles;
    double3 temp31 = {8.0, 1.0, -1.0};
	surface3.vertices[0] = temp31;
	double3 temp32 = {8.0, -1.0, -1.0};
	surface3.vertices[1] = temp32;
	double3 temp33 = {8.0, 0.0, 1.0};
	surface3.vertices[2] = temp33;
    double3 temp34 = {6.0, -1.0, -1.0};
    surface3.vertices[3] = temp34;
    double3 temp35 = {6.0, 0.0, 1.0};
    surface3.vertices[4] = temp35;
    double3 temp36 = {6.0, 1.0, -1.0};
    surface3.vertices[5] = temp36;

    surfaces[0] = surface1;
    surfaces[1] = surface2;
    surfaces[2] = surface3;

    double3 origin = {9.0, 0.0, 0.0};
	double3 direction = {1.0, 0.0, 0.0};

    photon->position = origin;
    photon->direction = direction;
    photon->layerId = 1;

    InputInfo input;
    input.surface = surfaces;
    input.layerInfo = new LayerInfo[3];
    input.layerInfo[1].numberOfSurfaces = 2;
    input.layerInfo[1].surfaceId = new int[input.layerInfo[1].numberOfSurfaces];
    input.layerInfo[1].surfaceId[0] = 0;
    input.layerInfo[1].surfaceId[1] = 2;

    IntersectionInfo intersection = GetIntersectionInfo(photon, &input);

    CPPUNIT_ASSERT_EQUAL(0, intersection.isFindIntersection);

    delete[] surface1.vertices;
    delete[] surface1.triangles;
    delete[] surface2.vertices;
    delete[] surface2.triangles;
    delete[] surface3.vertices;
    delete[] input.layerInfo[1].surfaceId;
    delete[] input.layerInfo;
    delete[] surfaces;
    delete photon;
}

//END GetIntersectionInfo======================================================

//MovePhoton===================================================================

void McmlKernelTestCase::IsCorrectPhotonMoving1()
{
    PhotonState photon;
    photon.position = getVector(0.0, 0.5, 1.0);
    photon.direction = getVector(1.0, 1.0, 1.0);
	photon.step = 1.0;
    
	MovePhoton(&photon);

    assertVectorEqual(getVector(1.0, 1.5, 2.0), photon.position, delta);
}

void McmlKernelTestCase::IsCorrectPhotonMoving2()
{
    PhotonState photon;
    photon.position = getVector(0.1, 0.2, 0.3);
    photon.direction = getVector(0.3, 0.2, 0.1);
	photon.step = 0.5;
    
	MovePhoton(&photon);

    assertVectorEqual(getVector(0.25, 0.3, 0.35), photon.position, delta);
}

//END MovePhoton===============================================================

//GetAreaIndex=================================================================

void McmlKernelTestCase::IsGetCorrectAreaIndex1()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

	area->corner.x = 0.0;
	area->corner.y = 0.0;
	area->corner.z = 0.0;
	area->length.x = 1.0;
	area->length.y = 1.0;
	area->length.z = 1.0;
	area->partitionNumber.x = 100;
	area->partitionNumber.y = 100;
	area->partitionNumber.z = 100;

	photon->position.x = 0.5;
	photon->position.y = 0.5;
	photon->position.z = 0.5;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(505050, index);

    delete area;
    delete photon;
}

void McmlKernelTestCase::IsGetCorrectAreaIndex2()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

	area->corner.x = 0.0;
	area->corner.y = 0.0;
	area->corner.z = 0.0;
	area->length.x = 1.0;
	area->length.y = 1.0;
	area->length.z = 1.0;
	area->partitionNumber.x = 99;
	area->partitionNumber.y = 99;
	area->partitionNumber.z = 99;

	photon->position.x = 0.5;
	photon->position.y = 0.5;
	photon->position.z = 0.5;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(485149, index);

    delete area;
    delete photon;
}

void McmlKernelTestCase::IsGetCorrectAreaIndex3()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

	area->corner.x = 0.0;
	area->corner.y = 0.0;
	area->corner.z = 0.0;
	area->length.x = 1.0;
	area->length.y = 1.0;
	area->length.z = 1.0;
	area->partitionNumber.x = 100;
	area->partitionNumber.y = 100;
	area->partitionNumber.z = 100;

	photon->position.x = 0.9999;
	photon->position.y = 0.9999;
	photon->position.z = 0.9999;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(999999, index);

    delete area;
    delete photon;
}

void McmlKernelTestCase::IsGetCorrectAreaIndex4()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

	area->corner.x = 1.0;
	area->corner.y = 1.0;
	area->corner.z = 1.0;
	area->length.x = 1.0;
	area->length.y = 1.0;
	area->length.z = 1.0;
	area->partitionNumber.x = 100;
	area->partitionNumber.y = 100;
	area->partitionNumber.z = 100;

	photon->position.x = 1.5;
	photon->position.y = 1.5;
	photon->position.z = 1.5;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(505050, index);

    delete area;
    delete photon;
}

void McmlKernelTestCase::IsGetCorrectAreaIndex5()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

    area->corner.x = -1.5;
	area->corner.y = -1.4;
	area->corner.z = 0.3;
	area->length.x = 2.0;
	area->length.y = 2.0;
	area->length.z = 2.0;
	area->partitionNumber.x = 10;
	area->partitionNumber.y = 20;
	area->partitionNumber.z = 30;

	photon->position.x = -0.9;
	photon->position.y = 0.0;
	photon->position.z = 0.5;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(2223, index);

    delete area;
    delete photon;
}

void McmlKernelTestCase::IsGetCorrectAreaIndex6()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

    area->corner.x = 0.0;
	area->corner.y = 0.0;
	area->corner.z = 0.0;
	area->length.x = 1.0;
	area->length.y = 1.0;
	area->length.z = 1.0;
	area->partitionNumber.x = 3;
	area->partitionNumber.y = 3;
	area->partitionNumber.z = 3;

	photon->position.x = 0.5;
	photon->position.y = 0.5;
	photon->position.z = 0.5;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(13, index);

    delete area;
    delete photon;
}

void McmlKernelTestCase::IsGetCorrectAreaIndexIfIndexOutOfRange1()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

    area->corner.x = 1.0;
	area->corner.y = 1.0;
	area->corner.z = 1.0;
	area->length.x = 1.0;
	area->length.y = 1.0;
	area->length.z = 1.0;
	area->partitionNumber.x = 100;
	area->partitionNumber.y = 100;
	area->partitionNumber.z = 100;

	photon->position.x = 1.5;
	photon->position.y = 1.5;
	photon->position.z = 2.5;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(-1, index);

    delete area;
    delete photon;
}

void McmlKernelTestCase::IsGetCorrectAreaIndexIfIndexOutOfRange2()
{
    Area* area = new Area;
    PhotonState* photon = new PhotonState();

	area->corner.x = 1.0;
	area->corner.y = 1.0;
	area->corner.z = 1.0;
	area->length.x = 1.0;
	area->length.y = 1.0;
	area->length.z = 1.0;
	area->partitionNumber.x = 100;
	area->partitionNumber.y = 100;
	area->partitionNumber.z = 100;

	photon->position.x = -1.5;
	photon->position.y = 1.5;
	photon->position.z = 1.5;

	int index = GetAreaIndex(photon, area);

	CPPUNIT_ASSERT_EQUAL(-1, index);

    delete area;
    delete photon;
}

//END GetAreaIndex=============================================================

//ComputeCosineTheta===========================================================

void McmlKernelTestCase::IsCorrectCosineTheta1()
{
	double anisotropy = 0.5;
	double cosineTheta = ComputeCosineTheta(anisotropy, randomGenerator);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.925699, cosineTheta, delta);
}

void McmlKernelTestCase::IsCorrectCosineTheta2()
{
	double anisotropy = 0.9;
	double cosineTheta = ComputeCosineTheta(anisotropy, randomGenerator);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.9974254, cosineTheta, delta);
}

void McmlKernelTestCase::IsCorrectCosineThetaIfAnisotropyIsZero()
{
	double anisotropy = 0.0;
	double cosineTheta = ComputeCosineTheta(anisotropy, randomGenerator);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.63400816, cosineTheta, delta);
}

void McmlKernelTestCase::IsCosineThetaCorrectAsCosine()
{
	double anisotropy;
	double cosineTheta;

	for (int i = 0; i < 100; ++i)
	{
		anisotropy = NextMCG59(randomGenerator);
		cosineTheta = ComputeCosineTheta(anisotropy, randomGenerator);
		CPPUNIT_ASSERT((cosineTheta >= -1.0) && (cosineTheta <= 1.0));
	}
}

//END ComputeCosineTheta=======================================================

//ComputePhotonDirection=======================================================

void McmlKernelTestCase::IsCorrectComputePhotonDirection1()
{
	double anisotropy = 0.5;
    PhotonState* photon = new PhotonState();
	
	photon->direction.x = 0.0;
	photon->direction.y = 0.0;
	photon->direction.z = 1.0;

	ComputePhotonDirection(photon, anisotropy, randomGenerator);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.3289909967, photon->direction.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.1866715735, photon->direction.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.9256990049, photon->direction.z, delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, LengthOfVector(photon->direction), delta);

    delete photon;
}

void McmlKernelTestCase::IsCorrectComputePhotonDirection2()
{
	double anisotropy = 0.5;
    PhotonState* photon = new PhotonState();
	
	photon->direction.x = 0.5;
	photon->direction.y = 0.5;
	photon->direction.z = 0.5;
    photon->direction = NormalizeVector(photon->direction);

	ComputePhotonDirection(photon, anisotropy, randomGenerator);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.2681458221, photon->direction.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5321392932, photon->direction.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8030725936, photon->direction.z, delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, LengthOfVector(photon->direction), delta);

    delete photon;
}

void McmlKernelTestCase::IsCorrectComputePhotonDirection3()
{
	double anisotropy = 0.5;
    PhotonState* photon = new PhotonState();
	
	photon->direction.x = 0.1;
	photon->direction.y = 0.2;
	photon->direction.z = 0.5;
    photon->direction = NormalizeVector(photon->direction);

	ComputePhotonDirection(photon, anisotropy, randomGenerator);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.132265401, photon->direction.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.152879525, photon->direction.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.979353722, photon->direction.z, delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, LengthOfVector(photon->direction), delta);

    delete photon;
}

//END ComputePhotonDirection===================================================

//ComputeTransmitCosine========================================================

void McmlKernelTestCase::IsCorrectComputeTransmitCosine1()
{
	double incidentCos = cos(PI / 4.0);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.2;
	
	double transmitCos = ComputeTransmitCosine(incidentRefractiveIndex, 
		transmitRefractiveIndex, incidentCos);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.7614861, transmitCos, delta);
}

void McmlKernelTestCase::IsCorrectComputeTransmitCosine2()
{
	double incidentCos = cos(PI / 4.0);
	double incidentRefractiveIndex = 1.0;
	double transmitRefractiveIndex = 1.2;
	
	double transmitCos = ComputeTransmitCosine(incidentRefractiveIndex, 
		transmitRefractiveIndex, incidentCos);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8079466, transmitCos, delta);
}

void McmlKernelTestCase::IsCorrectComputeTransmitCosineThenRefractiveIndicesAreEqual()
{
	double incidentCos = cos(PI / 4.0);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.1;
	
	double transmitCos = ComputeTransmitCosine(incidentRefractiveIndex, 
		transmitRefractiveIndex, incidentCos);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(incidentCos, transmitCos, delta);
}

void McmlKernelTestCase::IsCorrectComputeTransmitCosineThenAngleIsZero()
{
	double incidentCos = cos(0.00001);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.2;
	
	double transmitCos = ComputeTransmitCosine(incidentRefractiveIndex, 
		transmitRefractiveIndex, incidentCos);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(incidentCos, transmitCos, delta);
}

void McmlKernelTestCase::IsCorrectComputeTransmitCosineThenAngleIsRight()
{
	double incidentCos = cos(PI / 1.99999);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.2;
	
	double transmitCos = ComputeTransmitCosine(incidentRefractiveIndex, 
		transmitRefractiveIndex, incidentCos);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, transmitCos, delta);
}

void McmlKernelTestCase::IsCorrectComputeTransmitCosineThenTransmitSineIsMoreThenOne()
{
	double incidentCos = cos(PI / 3.0);
	double incidentRefractiveIndex = 1.5;
	double transmitRefractiveIndex = 1.0;
	
	double transmitCos = ComputeTransmitCosine(incidentRefractiveIndex, 
		transmitRefractiveIndex, incidentCos);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, transmitCos, delta);
}

//END ComputeTransmitCosine====================================================

//ComputeFresnelReflectance====================================================

void McmlKernelTestCase::IsCorrectComputeFresnelReflectance1()
{
	double incidentCos = cos(PI / 3.0);
	double transmitCos = cos(PI / 4.0);
	double incidentRefractiveIndex = 1.0;
	double transmitRefractiveIndex = 1.2;
	double reflectance = ComputeFrenselReflectance(incidentRefractiveIndex, transmitRefractiveIndex,
		incidentCos, transmitCos);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0384758, reflectance, delta); 
}

void McmlKernelTestCase::IsCorrectComputeFresnelReflectance2()
{
	double incidentCos = cos(PI / 3.0);
	double transmitCos = cos(PI / 6.0);
	double incidentRefractiveIndex = 1.0;
	double transmitRefractiveIndex = 1.2;
	double reflectance = ComputeFrenselReflectance(incidentRefractiveIndex, transmitRefractiveIndex,
		incidentCos, transmitCos);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.125, reflectance, delta); 
}

void McmlKernelTestCase::IsCorrectComputeFresnelReflectanceThenRefractiveIndicesAreEqual()
{
	double incidentCos = cos(PI / 3.0);
	double transmitCos = cos(PI / 6.0);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.1;
	double reflectance = ComputeFrenselReflectance(incidentRefractiveIndex, transmitRefractiveIndex,
		incidentCos, transmitCos);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, reflectance, delta); 
}

void McmlKernelTestCase::IsCorrectComputeFresnelReflectanceThenIncidentAngleIsZero()
{
	double incidentCos = cos(0.0000001);
	double transmitCos = cos(PI / 6.0);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.2;
	double reflectance = ComputeFrenselReflectance(incidentRefractiveIndex, transmitRefractiveIndex,
		incidentCos, transmitCos);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.001890359, reflectance, delta); 
}

void McmlKernelTestCase::IsCorrectComputeFresnelReflectanceThenIncidentAngleIsRight()
{
	double incidentCos = cos(PI / 2.00000001);
	double transmitCos = cos(PI / 6.0);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.2;
	double reflectance = ComputeFrenselReflectance(incidentRefractiveIndex, transmitRefractiveIndex,
		incidentCos, transmitCos);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, reflectance, delta); 
}

void McmlKernelTestCase::IsCorrectComputeFresnelReflectanceThenTransmitSineIsOne()
{
	double incidentCos = cos(PI / 3.0);
	double transmitCos = cos(0.0);
	double incidentRefractiveIndex = 1.1;
	double transmitRefractiveIndex = 1.2;
	double reflectance = ComputeFrenselReflectance(incidentRefractiveIndex, transmitRefractiveIndex,
		incidentCos, transmitCos);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, reflectance, delta); 
}

//END ComputeFresnelReflectance================================================

//FindIntersectionLayer========================================================

void McmlKernelTestCase::IsCorrectFoundIntersectionLayer1()
{
    InputInfo input;
    input.numberOfLayers = 4;
    input.layerInfo = new LayerInfo[input.numberOfLayers];
    
    input.layerInfo[0].numberOfSurfaces = 3;
    input.layerInfo[0].surfaceId = new int[input.layerInfo[0].numberOfSurfaces];
    input.layerInfo[0].surfaceId[0] = 0;
    input.layerInfo[0].surfaceId[1] = 2;
    input.layerInfo[0].surfaceId[2] = 4;
    
    input.layerInfo[1].numberOfSurfaces = 3;
    input.layerInfo[1].surfaceId = new int[input.layerInfo[1].numberOfSurfaces];
    input.layerInfo[1].surfaceId[0] = 0;
    input.layerInfo[1].surfaceId[1] = 1;
    input.layerInfo[1].surfaceId[2] = 3;
    
    input.layerInfo[2].numberOfSurfaces = 3;
    input.layerInfo[2].surfaceId = new int[input.layerInfo[2].numberOfSurfaces];
    input.layerInfo[2].surfaceId[0] = 1;
    input.layerInfo[2].surfaceId[1] = 2;
    input.layerInfo[2].surfaceId[2] = 5;
    
    input.layerInfo[3].numberOfSurfaces = 3;
    input.layerInfo[3].surfaceId = new int[input.layerInfo[3].numberOfSurfaces];
    input.layerInfo[3].surfaceId[0] = 3;
    input.layerInfo[3].surfaceId[1] = 4;
    input.layerInfo[3].surfaceId[2] = 5;

    int layerId = FindIntersectionLayer(&input, 1, 1);
    CPPUNIT_ASSERT_EQUAL(2, layerId);

    delete[] input.layerInfo[0].surfaceId;
    delete[] input.layerInfo[1].surfaceId;
    delete[] input.layerInfo[2].surfaceId;
    delete[] input.layerInfo[3].surfaceId;
    delete[] input.layerInfo;
}

void McmlKernelTestCase::IsCorrectFoundIntersectionLayer2()
{
    InputInfo input;
    input.numberOfLayers = 4;
    input.layerInfo = new LayerInfo[input.numberOfLayers];
    
    input.layerInfo[0].numberOfSurfaces = 3;
    input.layerInfo[0].surfaceId = new int[input.layerInfo[0].numberOfSurfaces];
    input.layerInfo[0].surfaceId[0] = 0;
    input.layerInfo[0].surfaceId[1] = 2;
    input.layerInfo[0].surfaceId[2] = 4;
    
    input.layerInfo[1].numberOfSurfaces = 3;
    input.layerInfo[1].surfaceId = new int[input.layerInfo[1].numberOfSurfaces];
    input.layerInfo[1].surfaceId[0] = 0;
    input.layerInfo[1].surfaceId[1] = 1;
    input.layerInfo[1].surfaceId[2] = 3;
    
    input.layerInfo[2].numberOfSurfaces = 3;
    input.layerInfo[2].surfaceId = new int[input.layerInfo[2].numberOfSurfaces];
    input.layerInfo[2].surfaceId[0] = 1;
    input.layerInfo[2].surfaceId[1] = 2;
    input.layerInfo[2].surfaceId[2] = 5;
    
    input.layerInfo[3].numberOfSurfaces = 3;
    input.layerInfo[3].surfaceId = new int[input.layerInfo[3].numberOfSurfaces];
    input.layerInfo[3].surfaceId[0] = 3;
    input.layerInfo[3].surfaceId[1] = 4;
    input.layerInfo[3].surfaceId[2] = 5;

    int layerId = FindIntersectionLayer(&input, 3, 3);
    CPPUNIT_ASSERT_EQUAL(1, layerId);

    delete[] input.layerInfo[0].surfaceId;
    delete[] input.layerInfo[1].surfaceId;
    delete[] input.layerInfo[2].surfaceId;
    delete[] input.layerInfo[3].surfaceId;
    delete[] input.layerInfo;
}

void McmlKernelTestCase::IsCorrectFoundIntersectionLayer3()
{
    InputInfo input;
    input.numberOfLayers = 4;
    input.layerInfo = new LayerInfo[input.numberOfLayers];
    
    input.layerInfo[0].numberOfSurfaces = 3;
    input.layerInfo[0].surfaceId = new int[input.layerInfo[0].numberOfSurfaces];
    input.layerInfo[0].surfaceId[0] = 0;
    input.layerInfo[0].surfaceId[1] = 2;
    input.layerInfo[0].surfaceId[2] = 4;
    
    input.layerInfo[1].numberOfSurfaces = 3;
    input.layerInfo[1].surfaceId = new int[input.layerInfo[1].numberOfSurfaces];
    input.layerInfo[1].surfaceId[0] = 0;
    input.layerInfo[1].surfaceId[1] = 1;
    input.layerInfo[1].surfaceId[2] = 3;
    
    input.layerInfo[2].numberOfSurfaces = 3;
    input.layerInfo[2].surfaceId = new int[input.layerInfo[2].numberOfSurfaces];
    input.layerInfo[2].surfaceId[0] = 1;
    input.layerInfo[2].surfaceId[1] = 2;
    input.layerInfo[2].surfaceId[2] = 5;
    
    input.layerInfo[3].numberOfSurfaces = 3;
    input.layerInfo[3].surfaceId = new int[input.layerInfo[3].numberOfSurfaces];
    input.layerInfo[3].surfaceId[0] = 3;
    input.layerInfo[3].surfaceId[1] = 4;
    input.layerInfo[3].surfaceId[2] = 5;

    int layerId = FindIntersectionLayer(&input, 2, 2);
    CPPUNIT_ASSERT_EQUAL(0, layerId);

    delete[] input.layerInfo[0].surfaceId;
    delete[] input.layerInfo[1].surfaceId;
    delete[] input.layerInfo[2].surfaceId;
    delete[] input.layerInfo[3].surfaceId;
    delete[] input.layerInfo;
}

void McmlKernelTestCase::IsCorrectFoundIntersectionLayer4()
{
    InputInfo input;
    input.numberOfLayers = 4;
    input.layerInfo = new LayerInfo[input.numberOfLayers];
    
    input.layerInfo[0].numberOfSurfaces = 3;
    input.layerInfo[0].surfaceId = new int[input.layerInfo[0].numberOfSurfaces];
    input.layerInfo[0].surfaceId[0] = 0;
    input.layerInfo[0].surfaceId[1] = 2;
    input.layerInfo[0].surfaceId[2] = 4;
    
    input.layerInfo[1].numberOfSurfaces = 3;
    input.layerInfo[1].surfaceId = new int[input.layerInfo[1].numberOfSurfaces];
    input.layerInfo[1].surfaceId[0] = 0;
    input.layerInfo[1].surfaceId[1] = 1;
    input.layerInfo[1].surfaceId[2] = 3;
    
    input.layerInfo[2].numberOfSurfaces = 3;
    input.layerInfo[2].surfaceId = new int[input.layerInfo[2].numberOfSurfaces];
    input.layerInfo[2].surfaceId[0] = 1;
    input.layerInfo[2].surfaceId[1] = 2;
    input.layerInfo[2].surfaceId[2] = 5;
    
    input.layerInfo[3].numberOfSurfaces = 3;
    input.layerInfo[3].surfaceId = new int[input.layerInfo[3].numberOfSurfaces];
    input.layerInfo[3].surfaceId[0] = 3;
    input.layerInfo[3].surfaceId[1] = 4;
    input.layerInfo[3].surfaceId[2] = 5;

    int layerId = FindIntersectionLayer(&input, 6, 1);
    CPPUNIT_ASSERT_EQUAL(-1, layerId);

    delete[] input.layerInfo[0].surfaceId;
    delete[] input.layerInfo[1].surfaceId;
    delete[] input.layerInfo[2].surfaceId;
    delete[] input.layerInfo[3].surfaceId;
    delete[] input.layerInfo;
}

//END FindIntersectionLayer====================================================

//RefractVector================================================================

void McmlKernelTestCase::IsCorrectRefractVector1()
{
    double3 incidentVector;
    incidentVector.x = 1.0;
    incidentVector.y = -1.0;
    incidentVector.z = 0.0;
    
    double3 normalVector;
    normalVector.x = 0.0;
    normalVector.y = 1.0;
    normalVector.z = 0.0;

    double3 transmitVector = RefractVector(1.1, 1.2, incidentVector, normalVector);

    double3 expected = {0.648181216, -0.7614861201, 0.0};
    assertVectorEqual(expected, transmitVector, delta);
}

void McmlKernelTestCase::IsCorrectRefractVector2()
{
    double3 incidentVector;
    incidentVector.x = 1.0;
    incidentVector.y = 1.0;
    incidentVector.z = 1.0;
    
    double3 normalVector;
    normalVector.x = 1.0;
    normalVector.y = 0.0;
    normalVector.z = 0.0;

    double3 transmitVector = RefractVector(sqrt(3.0), 2 * sqrt(3.0), incidentVector, normalVector);

    double3 expected = {sqrt(10.0) / sqrt(12.0), 1.0 / sqrt(12.0), 1.0 / sqrt(12.0)};
    assertVectorEqual(expected, transmitVector, delta);
}

//END RefractVector============================================================

//ReflectVector================================================================

void McmlKernelTestCase::IsCorrectReflectVector1()
{
    double3 incidentVector;
    incidentVector.x = 1.0;
    incidentVector.y = -1.0;
    incidentVector.z = 0.0;
    
    double3 normalVector;
    normalVector.x = 0.0;
    normalVector.y = -1.0;
    normalVector.z = 0.0;

    double3 transmitVector = ReflectVector(incidentVector, normalVector);

    double3 expected = {1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0};
    assertVectorEqual(expected, transmitVector, delta);
}

void McmlKernelTestCase::IsCorrectReflectVector2()
{
    double3 incidentVector;
    incidentVector.x = 0.0;
    incidentVector.y = -1.0;
    incidentVector.z = 0.0;
    
    double3 normalVector;
    normalVector.x = 1.0;
    normalVector.y = 1.0;
    normalVector.z = 0.0;

    double3 transmitVector = ReflectVector(incidentVector, normalVector);

    double3 expected = {1.0, 0.0, 0.0};
    assertVectorEqual(expected, transmitVector, delta);
}

void McmlKernelTestCase::IsCorrectReflectVector3()
{
    double3 incidentVector;
    incidentVector.x = 0.0;
    incidentVector.y = -1.0;
    incidentVector.z = 0.0;
    
    double3 normalVector;
    normalVector.x = 1.0;
    normalVector.y = 1.0;
    normalVector.z = 1.0;

    double3 transmitVector = ReflectVector(incidentVector, normalVector);

    double3 expected = {2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0};
    assertVectorEqual(expected, transmitVector, delta);
}

//END ReflectVector============================================================

//ComputeStepSizeInTissue======================================================

void McmlKernelTestCase::IsCorrectComputeStepSize1()
{
    PhotonState* photon = new PhotonState();
    InputInfo input;
    input.layerInfo = new LayerInfo[2];
    photon->layerId = 1;

    input.layerInfo[photon->layerId].absorptionCoefficient = 20.0;
    input.layerInfo[photon->layerId].scatteringCoefficient = 100.0;
    ComputeStepSizeInTissue(photon, &input, randomGenerator);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00168426, photon->step, delta);

    delete[] input.layerInfo;
    delete photon;
}

void McmlKernelTestCase::IsCorrectComputeStepSize2()
{
    PhotonState* photon = new PhotonState();
    InputInfo input;
    input.layerInfo = new LayerInfo[2];
    photon->layerId = 1;

    input.layerInfo[photon->layerId].absorptionCoefficient = 4.0;
    input.layerInfo[photon->layerId].scatteringCoefficient = 5.0;
    ComputeStepSizeInTissue(photon, &input, randomGenerator);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0224568, photon->step, delta);

    delete[] input.layerInfo;
    delete photon;
}

//END ComputeStepSizeInTissue==================================================

//ComputeDroppedWeightOfPhoton=================================================

void McmlKernelTestCase::IsCorrectComputeDroppedWeight1()
{
    PhotonState* photon = new PhotonState();

    InputInfo input;
    input.layerInfo = new LayerInfo[2];

    OutputInfo output;
	output.absorption = new double[1000];
	output.absorption[555] = 0.0;

	input.layerInfo[1].absorptionCoefficient = 20.0;
	input.layerInfo[1].scatteringCoefficient = 200.0;

	photon->weight = 0.8;
    photon->layerId = 1;

	ComputeDroppedWeightOfPhoton(photon, &input, &output, 555);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / input.layerInfo[1].scatteringCoefficient,
        output.absorption[555], delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.72727273, photon->weight, delta);

    delete[] input.layerInfo;
    delete[] output.absorption;
    delete photon;
}

void McmlKernelTestCase::IsCorrectComputeDroppedWeight2()
{
    PhotonState* photon = new PhotonState();

    InputInfo input;
    input.layerInfo = new LayerInfo[2];

    OutputInfo output;
	output.absorption = new double[1000];
	output.absorption[555] = 0.0;

	input.layerInfo[1].absorptionCoefficient = 10.0;
	input.layerInfo[1].scatteringCoefficient = 100.0;

	photon->weight = 0.7;
    photon->layerId = 1;

	ComputeDroppedWeightOfPhoton(photon, &input, &output, 555);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / input.layerInfo[1].scatteringCoefficient,
        output.absorption[555], delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.63636364, photon->weight, delta);

    delete[] input.layerInfo;
    delete[] output.absorption;
    delete photon;
}

void McmlKernelTestCase::IsCorrectComputeDroppedWeightOnDoubleCall()
{
    PhotonState* photon = new PhotonState();

    InputInfo input;
    input.layerInfo = new LayerInfo[2];

    OutputInfo output;
	output.absorption = new double[1000];
	output.absorption[555] = 0.0;

	input.layerInfo[1].absorptionCoefficient = 10.0;
	input.layerInfo[1].scatteringCoefficient = 90.0;

	photon->weight = 1.0;
    photon->layerId = 1;

	ComputeDroppedWeightOfPhoton(photon, &input, &output, 555);
	ComputeDroppedWeightOfPhoton(photon, &input, &output, 555);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 / input.layerInfo[1].scatteringCoefficient,
        output.absorption[555], delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.81, photon->weight, delta);

    delete[] input.layerInfo;
    delete[] output.absorption;
    delete photon;
}

//END ComputeDroppedWeightOfPhoton=============================================

//GetCorrectCriticalCos========================================================

void McmlKernelTestCase::IsCorrectCriticalCos1()
{
    int n = 3;
    LayerInfo* layer = new LayerInfo[n];

    layer[0].refractiveIndex = 1.0;
	layer[1].refractiveIndex = 2.0;
	layer[2].refractiveIndex = 1.0;

    double criticalCos = GetCriticalCos(layer, 1, 2);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(sqrt(3.0) / 2.0, criticalCos, delta);

    delete[] layer;
}

void McmlKernelTestCase::IsCorrectCriticalCos2()
{
    int n = 3;
    LayerInfo* layer = new LayerInfo[n];

    layer[0].refractiveIndex = 1.0;
	layer[1].refractiveIndex = 2.0;
	layer[2].refractiveIndex = 1.0;

    double criticalCos = GetCriticalCos(layer, 0, 1);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, criticalCos, delta);

    delete[] layer;
}

//END GetCorrectCriticalCos====================================================

//UpdateWeightInDetectors======================================================

void McmlKernelTestCase::IsCorrectUpdateWeightInDetectors1()
{
    PhotonState* photon = new PhotonState();

    InputInfo input;
    input.layerInfo = new LayerInfo[2];
    input.numberOfDetectors = 2;
    input.detector = new Detector[input.numberOfDetectors];
    
    double3 length = {0.001, 0.001, 0.001};

    double3 center1 = {0.0, 0.0, 0.0};
    input.detector[0].center = center1;
    input.detector[0].length = length;

    double3 center2 = {0.1, 0.1, -0.1};
    input.detector[1].center = center2;
    input.detector[1].length = length;

    OutputInfo output;
    output.weigthInDetector = new double[input.numberOfDetectors];
    for (int i = 0; i < input.numberOfDetectors; ++i)
    {
        output.weigthInDetector[i] = 0.0;
    }

	input.layerInfo[1].absorptionCoefficient = 20.0;
	input.layerInfo[1].scatteringCoefficient = 200.0;

	photon->weight = 0.8;
    photon->layerId = 1;
    double3 position = {0.0001, 0.0, 0.0};
    photon->position = position;

    UpdateWeightInDetectors(photon, &input, &output);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, output.weigthInDetector[0], delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, output.weigthInDetector[1], delta);

    delete[] input.layerInfo;
    delete[] input.detector;
    delete[] output.weigthInDetector;
    delete photon;
}

void McmlKernelTestCase::IsCorrectUpdateWeightInDetectors2()
{
    PhotonState* photon = new PhotonState();

    InputInfo input;
    input.layerInfo = new LayerInfo[2];
    input.numberOfDetectors = 2;
    input.detector = new Detector[input.numberOfDetectors];
    
    double3 length = {0.001, 0.001, 0.001};

    double3 center1 = {0.0, 0.0, 0.0};
    input.detector[0].center = center1;
    input.detector[0].length = length;

    double3 center2 = {0.1, 0.1, -0.1};
    input.detector[1].center = center2;
    input.detector[1].length = length;

    OutputInfo output;
    output.weigthInDetector = new double[input.numberOfDetectors];
    for (int i = 0; i < input.numberOfDetectors; ++i)
    {
        output.weigthInDetector[i] = 0.0;
    }

	input.layerInfo[1].absorptionCoefficient = 20.0;
	input.layerInfo[1].scatteringCoefficient = 200.0;

	photon->weight = 0.8;
    photon->layerId = 1;
    double3 position = {0.1001, 0.09999, -0.09999};
    photon->position = position;

    UpdateWeightInDetectors(photon, &input, &output);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, output.weigthInDetector[0], delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.8, output.weigthInDetector[1], delta);

    delete[] input.layerInfo;
    delete[] input.detector;
    delete[] output.weigthInDetector;
    delete photon;
}

void McmlKernelTestCase::IsCorrectUpdateWeightInDetectors3()
{
    PhotonState* photon = new PhotonState();

    InputInfo input;
    input.layerInfo = new LayerInfo[2];
    input.numberOfDetectors = 2;
    input.detector = new Detector[input.numberOfDetectors];
    
    double3 length = {0.001, 0.001, 0.001};

    double3 center1 = {0.0, 0.0, 0.0};
    input.detector[0].center = center1;
    input.detector[0].length = length;

    double3 center2 = {0.1, 0.1, -0.1};
    input.detector[1].center = center2;
    input.detector[1].length = length;

    OutputInfo output;
    output.weigthInDetector = new double[input.numberOfDetectors];
    for (int i = 0; i < input.numberOfDetectors; ++i)
    {
        output.weigthInDetector[i] = 0.0;
    }

	input.layerInfo[1].absorptionCoefficient = 20.0;
	input.layerInfo[1].scatteringCoefficient = 200.0;

	photon->weight = 0.8;
    photon->layerId = 1;
    double3 position = {0.1001, 0.09999, 0.09999};
    photon->position = position;

    UpdateWeightInDetectors(photon, &input, &output);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, output.weigthInDetector[0], delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, output.weigthInDetector[1], delta);

    delete[] input.layerInfo;
    delete[] input.detector;
    delete[] output.weigthInDetector;
    delete photon;
}

void McmlKernelTestCase::IsCorrectUpdateWeightInDetectorsDoubleCall()
{
    PhotonState* photon = new PhotonState();

    InputInfo input;
    input.layerInfo = new LayerInfo[2];
    input.numberOfDetectors = 2;
    input.detector = new Detector[input.numberOfDetectors];
    
    double3 length = {0.001, 0.001, 0.001};

    double3 center1 = {0.0, 0.0, 0.0};
    input.detector[0].center = center1;
    input.detector[0].length = length;

    double3 center2 = {0.1, 0.1, -0.1};
    input.detector[1].center = center2;
    input.detector[1].length = length;

    OutputInfo output;
    output.weigthInDetector = new double[input.numberOfDetectors];
    for (int i = 0; i < input.numberOfDetectors; ++i)
    {
        output.weigthInDetector[i] = 0.0;
    }

	input.layerInfo[1].absorptionCoefficient = 20.0;
	input.layerInfo[1].scatteringCoefficient = 200.0;

	photon->weight = 0.8;
    photon->layerId = 1;
    double3 position = {0.1001, 0.09999, -0.09999};
    photon->position = position;

    UpdateWeightInDetectors(photon, &input, &output);
    UpdateWeightInDetectors(photon, &input, &output);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, output.weigthInDetector[0], delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * 0.8, output.weigthInDetector[1], delta);

    delete[] input.layerInfo;
    delete[] input.detector;
    delete[] output.weigthInDetector;
    delete photon;
}

//END UpdateWeightInDetectors==================================================