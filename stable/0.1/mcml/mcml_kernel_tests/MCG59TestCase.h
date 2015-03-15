#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include "../mcml_kernel/mcml_mcg59.h"

class MCG59TestCase : public CppUnit::TestCase
{
	CPPUNIT_TEST_SUITE(MCG59TestCase);
	CPPUNIT_TEST(IsCorrectRandomValue);
	CPPUNIT_TEST(IsCorrectExpectationForSerialGenerator);	
	CPPUNIT_TEST(IsCorrectDispersionForSerialGenerator);
	CPPUNIT_TEST(IsCorrectExpectationForParallelGenerator);
	CPPUNIT_TEST(IsCorrectDispersionForParallelGenerator);
	CPPUNIT_TEST_SUITE_END();

public:
	void setUp();
	void tearDown();

private:
	void IsCorrectRandomValue();
	void IsCorrectExpectationForSerialGenerator();
	void IsCorrectDispersionForSerialGenerator();
	void IsCorrectExpectationForParallelGenerator();
	void IsCorrectDispersionForParallelGenerator();

private:
	void InitSerialRandomGenerator();
	void InitParallelRandomGenerator(uint id, uint step);

private:
	static const double deltaForSerialGenerator;
	static const double deltaForParallelGenerator;
	MCG59* randomGenerator;
};
