#include "MCG59TestCase.h"

CPPUNIT_TEST_SUITE_REGISTRATION(MCG59TestCase);

const double MCG59TestCase::deltaForSerialGenerator = 0.001;
const double MCG59TestCase::deltaForParallelGenerator = 0.01;

void MCG59TestCase::setUp()
{
	randomGenerator = new MCG59();
}

void MCG59TestCase::tearDown()
{
	delete randomGenerator;
	randomGenerator = NULL;
}

void MCG59TestCase::InitSerialRandomGenerator()
{
	uint64 seed = 777;
	uint step = 1;
	uint id = 0;
	InitMCG59(randomGenerator, seed, id, step);
}

void MCG59TestCase::InitParallelRandomGenerator(uint id, uint step)
{
	uint64 seed = 777;
	InitMCG59(randomGenerator, seed, id, step);
}

void MCG59TestCase::IsCorrectRandomValue()
{
	InitSerialRandomGenerator();	

	int n = 1000;
	double randomValue;
	for (int i = 0; i < n; ++i)
	{
		randomValue = NextMCG59(randomGenerator);
		CPPUNIT_ASSERT((randomValue >= 0.0) && (randomValue <= 1.0));
	}
}

void MCG59TestCase::IsCorrectExpectationForSerialGenerator()
{
	InitSerialRandomGenerator();

	double expectation = 0.0;
	int n = 10000;
	for (int i = 0; i < n; ++i)

	{
		expectation += NextMCG59(randomGenerator); 
	}
	expectation /= n;

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, expectation, deltaForSerialGenerator);
}

void MCG59TestCase::IsCorrectDispersionForSerialGenerator()
{
	InitSerialRandomGenerator();

	int n = 10000;
	double* randomValue = new double[n];
	if (randomValue == NULL)
	{
		CPPUNIT_FAIL("No memory");
	}

	for (int i = 0; i < n; ++i)
	{
		randomValue[i] = NextMCG59(randomGenerator);
	}

	double expectation = 0.0;
	for (int i = 0; i < n; ++i)
	{
		expectation += randomValue[i];
	}
	expectation /= n;

	double expectationOfSquare = 0.0;
	for (int i = 0; i < n; ++i)
	{
		expectationOfSquare += randomValue[i] * randomValue[i]; 
	}
	expectationOfSquare /= n;

	double dispersion = expectationOfSquare - expectation * expectation;

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / 12.0, dispersion, deltaForSerialGenerator);

	delete[] randomValue;
}

void MCG59TestCase::IsCorrectExpectationForParallelGenerator()
{
	uint step = 7;
	int n = 10000;
	double expectation;

	for (uint id = 0; id < step; ++id)
	{
		InitParallelRandomGenerator(id, step);
		
		expectation = 0.0;
		for (int i = 0; i < n; ++i)
		{
			expectation += NextMCG59(randomGenerator); 
		}
		expectation /= n;

		CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, expectation, deltaForParallelGenerator);
	}
}

void MCG59TestCase::IsCorrectDispersionForParallelGenerator()
{
	uint step = 4;
	int n = 10000;

	double* randomValue = new double[n];
	if (randomValue == NULL)
	{
		CPPUNIT_FAIL("No memory");
	}

	double expectation;
	double expectationOfSquare;
	double dispersion;
	for (uint id = 0; id < step; ++id)
	{
		InitParallelRandomGenerator(id, step);

		for (int i = 0; i < n; ++i)
		{
			randomValue[i] = NextMCG59(randomGenerator);
		}

		expectation = 0.0;
		for (int i = 0; i < n; ++i)
		{
			expectation += randomValue[i];
		}
		expectation /= n;

		expectationOfSquare = 0.0;
		for (int i = 0; i < n; ++i)
		{
			expectationOfSquare += randomValue[i] * randomValue[i]; 
		}
		expectationOfSquare /= n;

		dispersion = expectationOfSquare - expectation * expectation;

		CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / 12.0, dispersion, deltaForParallelGenerator);
	}

	delete[] randomValue;
}