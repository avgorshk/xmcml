#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include "../xmcml/mcml_intersection.h"

class McmlIntersectionTestCase : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(McmlIntersectionTestCase);
	
    //GetTriangleIntersectionDistance
    CPPUNIT_TEST(IsCorrectTriangleIntersectionDistance1);
	CPPUNIT_TEST(IsCorrectTriangleIntersectionDistance2);
	CPPUNIT_TEST(IsCorrectTriangleIntersectionDistance3);
	CPPUNIT_TEST(IsCorrectTriangleIntersectionDistance4);
    CPPUNIT_TEST(IsCorrectTriangleIntersectionDistance5);
    CPPUNIT_TEST(IsCorrectTriangleIntersectionDistance6);
    CPPUNIT_TEST(IsCorrectTriangleIntersectionDistance7);

    //ComputeSurfaceIntersection
    CPPUNIT_TEST(IsCorrectComputeSurfaceIntersection1);
    CPPUNIT_TEST(IsCorrectComputeSurfaceIntersection2);
    CPPUNIT_TEST(IsCorrectComputeSurfaceIntersection3);
    CPPUNIT_TEST(IsCorrectComputeSurfaceIntersection4);
    CPPUNIT_TEST(IsCorrectComputeSurfaceIntersection5);

    //ComputeIntersection
    CPPUNIT_TEST(IsCorrectComputeIntersection1);
    CPPUNIT_TEST(IsCorrectComputeIntersection2);
    CPPUNIT_TEST(IsCorrectComputeIntersection3);
    
	CPPUNIT_TEST_SUITE_END();

public:
    void setUp();
	void tearDown();
    static void assertVectorEqual(double3 expected, double3 actual, double delta);

private:
    //GetTriangleIntersectionDistance
    void IsCorrectTriangleIntersectionDistance1();
	void IsCorrectTriangleIntersectionDistance2();
	void IsCorrectTriangleIntersectionDistance3();
	void IsCorrectTriangleIntersectionDistance4();
    void IsCorrectTriangleIntersectionDistance5();
    void IsCorrectTriangleIntersectionDistance6();
    void IsCorrectTriangleIntersectionDistance7();

    //ComputeSurfaceIntersection
    void IsCorrectComputeSurfaceIntersection1();
    void IsCorrectComputeSurfaceIntersection2();
    void IsCorrectComputeSurfaceIntersection3();
    void IsCorrectComputeSurfaceIntersection4();
    void IsCorrectComputeSurfaceIntersection5();

    //ComputeIntersection
    void IsCorrectComputeIntersection1();
    void IsCorrectComputeIntersection2();
    void IsCorrectComputeIntersection3();

private:
    static const double delta;
};
