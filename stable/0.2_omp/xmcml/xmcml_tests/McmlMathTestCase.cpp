#include "McmlMathTestCase.h"

CPPUNIT_TEST_SUITE_REGISTRATION(McmlMathTestCase);

const double McmlMathTestCase::delta = 1.0E-7;

void McmlMathTestCase::IsCorrectLengthOfVector1()
{
	double3 vector = {2.0, 2.0, 1.0};
	double length = LengthOfVector(vector);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, length, delta);
}

void McmlMathTestCase::IsCorrectLengthOfVector2()
{
	double3 vector = {10.0, 10.0, 10.0};
	double length = LengthOfVector(vector);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(17.3205080757, length, delta);
}

void McmlMathTestCase::IsCorrectNormilizeVector1()
{
	double3 vector = {1.0, 0.0, 0.0};

	double3 result = NormalizeVector(vector);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.z, delta);
}

void McmlMathTestCase::IsCorrectNormilizeVector2()
{
	double3 vector = {2.0, 2.0, 1.0};

	double3 result = NormalizeVector(vector);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6666666, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6666666, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.3333333, result.z, delta);
}

void McmlMathTestCase::IsCorrectDot1()
{
	double3 vector1 = {1.0, 1.0, 1.0};

	double3 vector2 = {1.0, 1.0, 1.0};

	double result = DotVector(vector1, vector2);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, result, delta);
}

void McmlMathTestCase::IsCorrectDot2()
{
	double3 vector1 = {1.0, 2.0, 2.0};

	double3 vector2 = {0.0, 2.5, 0.5};

	double result = DotVector(vector1, vector2);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(6.0, result, delta);
}

void McmlMathTestCase::IsCorrectCross1()
{
	double3 vector1 = {1.0, 1.0, 1.0};
	double3 vector2 = {2.0, 1.0, 1.0};

	double3 result = CrossVector(vector1, vector2);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, result.z, delta);
}

void McmlMathTestCase::IsCorrectCross2()
{
	double3 vector1 = {1.0, 1.0, 1.0};
	double3 vector2 = {2.0, 1.0, 2.0};

	double3 result = CrossVector(vector1, vector2);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, result.z, delta);
}

void McmlMathTestCase::IsCorrectSubVector1()
{
	double3 vector1 = {1.0, 1.0, 1.0};
	double3 vector2 = {1.0, 1.0, 1.0};

	double3 result = SubVector(vector1, vector2);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.z, delta);
}

void McmlMathTestCase::IsCorrectSubVector2()
{
	double3 vector1 = {1.0, 0.3, 1.0};
	double3 vector2 = {-1.0, 1.0, 1.5};

	double3 result = SubVector(vector1, vector2);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.7, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.5, result.z, delta);
}

void McmlMathTestCase::IsCorrectGetNormal1()
{
	double3 a = {0.0, 0.0, 0.0};
	double3 b = {1.0, 0.0, 0.0};
	double3 c = {0.0, 1.0, 0.0};

	double3 result = GetPlaneNormal(a, b, c);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, result.z, delta);
}

void McmlMathTestCase::IsCorrectGetNormal2()
{
	double3 a = {0.0, 0.0, 0.0};
	double3 b = {0.0, 0.0, 1.0};
	double3 c = {0.0, 1.0, 0.0};

	double3 result = GetPlaneNormal(a, b, c);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0, result.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, result.z, delta);
}

