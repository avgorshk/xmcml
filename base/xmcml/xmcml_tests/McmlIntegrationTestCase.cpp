#include "McmlIntegrationTestCase.h"

#define PI 3.14159265358979323846

CPPUNIT_TEST_SUITE_REGISTRATION(McmlIntegrationTestCase);

void McmlIntegrationTestCase::IsCorrectComputeWeightIntegral1()
{
    double dot = 0.4;
    double anisotropy = 0.9;
    uint eps = 0;
    uint n = 1024;
    double expected = 1.0;
    double actual = ComputeWeightIntegral(dot, anisotropy, eps, n);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1.0E-10);
}

void McmlIntegrationTestCase::IsCorrectComputeWeightIntegral2()
{
    double dot = 0.0;
    double anisotropy = 0.9;
    uint eps = 1;
    uint n = 1024;
    double expected = 0.55;
    double actual = ComputeWeightIntegral(dot, anisotropy, eps, n);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1.0E-10);
}

void McmlIntegrationTestCase::IsCorrectComputeWeightIntegral3()
{
    double dot = -0.875;
    double anisotropy = 0.9;
    uint eps = 2;
    uint n = 128;
    double expected = 0.321033894859 / (2.0 * PI);
    double actual = ComputeWeightIntegral(dot, anisotropy, eps, n);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1.0E-10);
}