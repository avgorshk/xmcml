#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include "../xmcml/mcml_integration.h"

class McmlIntegrationTestCase : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(McmlIntegrationTestCase);

    //ComputeWeightIntegral
    CPPUNIT_TEST(IsCorrectComputeWeightIntegral1);
    CPPUNIT_TEST(IsCorrectComputeWeightIntegral2);
    CPPUNIT_TEST(IsCorrectComputeWeightIntegral3);

    CPPUNIT_TEST_SUITE_END();

    private:
        void IsCorrectComputeWeightIntegral1();
        void IsCorrectComputeWeightIntegral2();
        void IsCorrectComputeWeightIntegral3();
};
