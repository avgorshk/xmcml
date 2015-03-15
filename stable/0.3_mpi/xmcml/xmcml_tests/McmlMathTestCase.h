#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include "../xmcml/mcml_math.h"

class McmlMathTestCase : public CppUnit::TestCase
{
	CPPUNIT_TEST_SUITE(McmlMathTestCase);
	CPPUNIT_TEST(IsCorrectLengthOfVector1);
	CPPUNIT_TEST(IsCorrectLengthOfVector2);
	CPPUNIT_TEST(IsCorrectNormilizeVector1);
	CPPUNIT_TEST(IsCorrectNormilizeVector2);
	CPPUNIT_TEST(IsCorrectDot1);
	CPPUNIT_TEST(IsCorrectDot2);
	CPPUNIT_TEST(IsCorrectCross1);
	CPPUNIT_TEST(IsCorrectCross2);
	CPPUNIT_TEST(IsCorrectSubVector1);
	CPPUNIT_TEST(IsCorrectSubVector2);
	CPPUNIT_TEST(IsCorrectGetNormal1);
	CPPUNIT_TEST(IsCorrectGetNormal2);
	CPPUNIT_TEST_SUITE_END();

private:
	void IsCorrectLengthOfVector1();
	void IsCorrectLengthOfVector2();
	
	void IsCorrectNormilizeVector1();
	void IsCorrectNormilizeVector2();

	void IsCorrectDot1();
	void IsCorrectDot2();

	void IsCorrectCross1();
	void IsCorrectCross2();

	void IsCorrectSubVector1();
	void IsCorrectSubVector2();

	void IsCorrectGetNormal1();
	void IsCorrectGetNormal2();

	static const double delta;
};

