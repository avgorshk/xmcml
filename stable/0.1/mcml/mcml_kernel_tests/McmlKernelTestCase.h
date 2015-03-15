#pragma once

#include <cppunit/extensions/HelperMacros.h>
#include "../mcml_kernel/mcml_kernel.h"

class McmlKernelTestCase : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(McmlKernelTestCase);
    
    //ComputeSpecularReflectance
	CPPUNIT_TEST(IsCorrectSpecularReflectance1);
    CPPUNIT_TEST(IsCorrectSpecularReflectance2);
    CPPUNIT_TEST(IsCorrectSpecularReflectanceInGlass1);
    CPPUNIT_TEST(IsCorrectSpecularReflectanceInGlass2);
	
    //GetIntersectionInfo
    CPPUNIT_TEST(IsCorrectGettingIntersectionInfo1);
    CPPUNIT_TEST(IsCorrectGettingIntersectionInfo2);
    CPPUNIT_TEST(IsCorrectGettingIntersectionInfo3);

    //MovePhoton
    CPPUNIT_TEST(IsCorrectPhotonMoving1);
    CPPUNIT_TEST(IsCorrectPhotonMoving2);

    //GetAreaIndex
    CPPUNIT_TEST(IsGetCorrectAreaIndex1);
	CPPUNIT_TEST(IsGetCorrectAreaIndex2);
	CPPUNIT_TEST(IsGetCorrectAreaIndex3);
	CPPUNIT_TEST(IsGetCorrectAreaIndex4);
	CPPUNIT_TEST(IsGetCorrectAreaIndex5);
    CPPUNIT_TEST(IsGetCorrectAreaIndex6);
	CPPUNIT_TEST(IsGetCorrectAreaIndexIfIndexOutOfRange1);
	CPPUNIT_TEST(IsGetCorrectAreaIndexIfIndexOutOfRange2);

    //ComputeCosineTheta
    CPPUNIT_TEST(IsCorrectCosineTheta1);
	CPPUNIT_TEST(IsCorrectCosineTheta2);
	CPPUNIT_TEST(IsCorrectCosineThetaIfAnisotropyIsZero);
	CPPUNIT_TEST(IsCosineThetaCorrectAsCosine);

    //ComputePhotonDirection
    CPPUNIT_TEST(IsCorrectComputePhotonDirection1);
	CPPUNIT_TEST(IsCorrectComputePhotonDirection2);
	CPPUNIT_TEST(IsCorrectComputePhotonDirection3);
    
    //ComputeTransmitCosine
    CPPUNIT_TEST(IsCorrectComputeTransmitCosine1);
	CPPUNIT_TEST(IsCorrectComputeTransmitCosine2);
	CPPUNIT_TEST(IsCorrectComputeTransmitCosineThenRefractiveIndicesAreEqual);
	CPPUNIT_TEST(IsCorrectComputeTransmitCosineThenAngleIsZero);
	CPPUNIT_TEST(IsCorrectComputeTransmitCosineThenAngleIsRight);
	CPPUNIT_TEST(IsCorrectComputeTransmitCosineThenTransmitSineIsMoreThenOne);

    //ComputeFresnelReflectance
    CPPUNIT_TEST(IsCorrectComputeFresnelReflectance1);
	CPPUNIT_TEST(IsCorrectComputeFresnelReflectance2);
	CPPUNIT_TEST(IsCorrectComputeFresnelReflectanceThenRefractiveIndicesAreEqual);
	CPPUNIT_TEST(IsCorrectComputeFresnelReflectanceThenIncidentAngleIsZero);
	CPPUNIT_TEST(IsCorrectComputeFresnelReflectanceThenIncidentAngleIsRight);
	CPPUNIT_TEST(IsCorrectComputeFresnelReflectanceThenTransmitSineIsOne);

    //FindIntersectionLayer
    CPPUNIT_TEST(IsCorrectFoundIntersectionLayer1);
    CPPUNIT_TEST(IsCorrectFoundIntersectionLayer2);
    CPPUNIT_TEST(IsCorrectFoundIntersectionLayer3);
    CPPUNIT_TEST(IsCorrectFoundIntersectionLayer4);

    //RefractVector
    CPPUNIT_TEST(IsCorrectRefractVector1);
    CPPUNIT_TEST(IsCorrectRefractVector2);

    //ReflectVector
    CPPUNIT_TEST(IsCorrectReflectVector1);
    CPPUNIT_TEST(IsCorrectReflectVector2);
    CPPUNIT_TEST(IsCorrectReflectVector3);

    //ComputeStepSizeInTissue
    CPPUNIT_TEST(IsCorrectComputeStepSize1);
    CPPUNIT_TEST(IsCorrectComputeStepSize2);

    //ComputeDroppedWeightOfPhoton
    CPPUNIT_TEST(IsCorrectComputeDroppedWeight1);
	CPPUNIT_TEST(IsCorrectComputeDroppedWeight2);
	CPPUNIT_TEST(IsCorrectComputeDroppedWeightOnDoubleCall);

    //GetCriticalCos
    CPPUNIT_TEST(IsCorrectCriticalCos1);
    CPPUNIT_TEST(IsCorrectCriticalCos2);

    //UpdateWeightInDetectors
    CPPUNIT_TEST(IsCorrectUpdateWeightInDetectors1);
    CPPUNIT_TEST(IsCorrectUpdateWeightInDetectors2);
    CPPUNIT_TEST(IsCorrectUpdateWeightInDetectors3);
    CPPUNIT_TEST(IsCorrectUpdateWeightInDetectorsDoubleCall);

    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();
	void tearDown();
    void assertVectorEqual(double3 expected, double3 actual, double delta);
    double3 getVector(double x, double y, double z);

private:
    //ComputeSpecularReflectance
    void IsCorrectSpecularReflectance1();
	void IsCorrectSpecularReflectance2();
	void IsCorrectSpecularReflectanceInGlass1();
	void IsCorrectSpecularReflectanceInGlass2();

    //GetIntersectionInfo
    void IsCorrectGettingIntersectionInfo1();
    void IsCorrectGettingIntersectionInfo2();
    void IsCorrectGettingIntersectionInfo3();

    //MovePhoton
    void IsCorrectPhotonMoving1();
    void IsCorrectPhotonMoving2();

    //GetAreaIndex
    void IsGetCorrectAreaIndex1();
	void IsGetCorrectAreaIndex2();
	void IsGetCorrectAreaIndex3();
	void IsGetCorrectAreaIndex4();
	void IsGetCorrectAreaIndex5();
    void IsGetCorrectAreaIndex6();
	void IsGetCorrectAreaIndexIfIndexOutOfRange1();
	void IsGetCorrectAreaIndexIfIndexOutOfRange2();

    //ComputeCosineTheta
    void IsCorrectCosineTheta1();
	void IsCorrectCosineTheta2();
	void IsCorrectCosineThetaIfAnisotropyIsZero();
	void IsCosineThetaCorrectAsCosine();

    //ComputePhotonDirection
    void IsCorrectComputePhotonDirection1();
	void IsCorrectComputePhotonDirection2();
	void IsCorrectComputePhotonDirection3();

    //ComputeTransmitCosine
    void IsCorrectComputeTransmitCosine1();
	void IsCorrectComputeTransmitCosine2();
	void IsCorrectComputeTransmitCosineThenRefractiveIndicesAreEqual();
	void IsCorrectComputeTransmitCosineThenAngleIsZero();
	void IsCorrectComputeTransmitCosineThenAngleIsRight();
	void IsCorrectComputeTransmitCosineThenTransmitSineIsMoreThenOne();

    //ComputeFresnelReflectance
    void IsCorrectComputeFresnelReflectance1();
	void IsCorrectComputeFresnelReflectance2();
	void IsCorrectComputeFresnelReflectanceThenRefractiveIndicesAreEqual();
	void IsCorrectComputeFresnelReflectanceThenIncidentAngleIsZero();
	void IsCorrectComputeFresnelReflectanceThenIncidentAngleIsRight();
	void IsCorrectComputeFresnelReflectanceThenTransmitSineIsOne();

    //FindIntersectionLayer
    void IsCorrectFoundIntersectionLayer1();
    void IsCorrectFoundIntersectionLayer2();
    void IsCorrectFoundIntersectionLayer3();
    void IsCorrectFoundIntersectionLayer4();

    //RefractVector
    void IsCorrectRefractVector1();
    void IsCorrectRefractVector2();

    //ReflectVector
    void IsCorrectReflectVector1();
    void IsCorrectReflectVector2();
    void IsCorrectReflectVector3();

    //ComputeStepSizeInTissue
    void IsCorrectComputeStepSize1();
    void IsCorrectComputeStepSize2();
    void IsCorrectComputeStepSizeIfLeftoverStepIsNotZero1();
    void IsCorrectComputeStepSizeIfLeftoverStepIsNotZero2();

    //ComputeDroppedWeightOfPhoton
    void IsCorrectComputeDroppedWeight1();
	void IsCorrectComputeDroppedWeight2();
	void IsCorrectComputeDroppedWeightOnDoubleCall();

    //GetCriticalCos
    void IsCorrectCriticalCos1();
    void IsCorrectCriticalCos2();

    //UpdateWeightInDetectors
    void IsCorrectUpdateWeightInDetectors1();
    void IsCorrectUpdateWeightInDetectors2();
    void IsCorrectUpdateWeightInDetectors3();
    void IsCorrectUpdateWeightInDetectorsDoubleCall();

private:
    static const double delta;
    MCG59* randomGenerator;
};

