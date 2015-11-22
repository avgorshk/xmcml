#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include "../xmcml/mcml_math.h"

class McmlMathTestCase : public CppUnit::TestCase
{
	CPPUNIT_TEST_SUITE(McmlMathTestCase);

    //LengthOfVector
	CPPUNIT_TEST(IsCorrectLengthOfVector1);
	CPPUNIT_TEST(IsCorrectLengthOfVector2);

    //NormalizeVector
	CPPUNIT_TEST(IsCorrectNormilizeVector1);
	CPPUNIT_TEST(IsCorrectNormilizeVector2);

    //DotVector
	CPPUNIT_TEST(IsCorrectDot1);
	CPPUNIT_TEST(IsCorrectDot2);

    //CrossVector
	CPPUNIT_TEST(IsCorrectCross1);
	CPPUNIT_TEST(IsCorrectCross2);

    //SubVector
	CPPUNIT_TEST(IsCorrectSubVector1);
	CPPUNIT_TEST(IsCorrectSubVector2);

    //GetPlaneNormal
	CPPUNIT_TEST(IsCorrectGetNormal1);
	CPPUNIT_TEST(IsCorrectGetNormal2);

    //GetPlaneSegmentIntersectionPoint
    CPPUNIT_TEST(IsCorrectIntersectionPoint1);
    CPPUNIT_TEST(IsCorrectIntersectionPoint2);
    CPPUNIT_TEST(IsCorrectIntersectionPoint3);
    CPPUNIT_TEST(IsCorrectIntersectionPoint4);
    CPPUNIT_TEST(IsCorrectIntersectionPoint5);
    CPPUNIT_TEST(IsCorrectIntersectionPoint6);
    CPPUNIT_TEST(IsCorrectIntersectionPoint7);

	CPPUNIT_TEST_SUITE_END();

private:
    
    //LengthOfVector
	void IsCorrectLengthOfVector1();
	void IsCorrectLengthOfVector2();
	
    //NormalizeVector
	void IsCorrectNormilizeVector1();
	void IsCorrectNormilizeVector2();

    //DotVector
	void IsCorrectDot1();
	void IsCorrectDot2();

    //CrossVector
	void IsCorrectCross1();
	void IsCorrectCross2();

    //SubVector
	void IsCorrectSubVector1();
	void IsCorrectSubVector2();

    //GetPlaneNormal
	void IsCorrectGetNormal1();
	void IsCorrectGetNormal2();

    //GetPlaneSegmentIntersectionPoint
    void IsCorrectIntersectionPoint1();
    void IsCorrectIntersectionPoint2();
    void IsCorrectIntersectionPoint3();
    void IsCorrectIntersectionPoint4();
    void IsCorrectIntersectionPoint5();
    void IsCorrectIntersectionPoint6();
    void IsCorrectIntersectionPoint7();

	static const double delta;
};

