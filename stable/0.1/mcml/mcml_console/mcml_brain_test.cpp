#include "mcml_brain_test.h"
#include "..\mcml_kernel\mcml_kernel.h"
#include "mcml_writer.h"
#include <stdio.h>
#include <time.h>

#define PLANE_WIDTH 200
#define MIN_WEIGHT 1.0E-4

static Surface GetPlane(double z, int triangles)
{
    if (triangles < 2)
        triangles = 2;

    Surface plane;
    plane.numberOfVertices = triangles + 2;
    plane.vertices = new double3[plane.numberOfVertices];

    double step = PLANE_WIDTH / (triangles - 1);

    plane.vertices[0].x = -PLANE_WIDTH / 2.0;
    plane.vertices[0].y = PLANE_WIDTH / 2.0;
    plane.vertices[0].z = z;

    plane.vertices[1].x = -PLANE_WIDTH / 2.0;
    plane.vertices[1].y = -PLANE_WIDTH / 2.0;
    plane.vertices[1].z = z;

    int sign = 1;
    for (int i = 2; i < triangles + 1; ++i)
    {
        plane.vertices[i].x = -PLANE_WIDTH / 2.0 + step * (i - 1);
        plane.vertices[i].y = sign * PLANE_WIDTH / 2.0;
        plane.vertices[i].z = z;
        sign = -sign; 
    }

    plane.vertices[triangles + 1].x = PLANE_WIDTH / 2.0;
    plane.vertices[triangles + 1].y = sign * PLANE_WIDTH / 2.0;
    plane.vertices[triangles + 1].z = z;

    double3 vertice1 = {-PLANE_WIDTH / 2.0, -PLANE_WIDTH / 2.0, z};
    double3 vertice2 = {-PLANE_WIDTH / 2.0,  PLANE_WIDTH / 2.0, z};
    double3 vertice3 = { PLANE_WIDTH / 2.0, -PLANE_WIDTH / 2.0, z};
    double3 vertice4 = { PLANE_WIDTH / 2.0,  PLANE_WIDTH / 2.0, z};

    plane.vertices[0] = vertice1;
    plane.vertices[1] = vertice2;
    plane.vertices[2] = vertice3;
    plane.vertices[3] = vertice4;

    return plane;
}

static Area* GetArea()
{
    Area* area = new Area();
    
    double3 corner = {-PLANE_WIDTH / 2.0, -PLANE_WIDTH / 2.0, 0};
    double3 length = {PLANE_WIDTH, PLANE_WIDTH, PLANE_WIDTH / 2.0};
    int3 partitionNumber = {40, 40, 40};
    
    area->corner = corner;
    area->length = length;
    area->partitionNumber = partitionNumber;

    return area;
}

static LayerInfo* GetLayers()
{
    LayerInfo* layers = new LayerInfo[6];
    
    layers[0].numberOfSurfaces = 2;
    layers[0].surfaceId = new int[layers[0].numberOfSurfaces];
    layers[0].surfaceId[0] = 0;
    layers[0].surfaceId[1] = 5;
    layers[0].absorptionCoefficient = 0.0;
    layers[0].scatteringCoefficient = 0.0;
    layers[0].refractiveIndex = 1.0;
    layers[0].anisotropy = 0.0;

    layers[1].numberOfSurfaces = 2;
    layers[1].surfaceId = new int[layers[1].numberOfSurfaces];
    layers[1].surfaceId[0] = 0;
    layers[1].surfaceId[1] = 1;
    layers[1].absorptionCoefficient = 0.06;
    //layers[1].absorptionCoefficient = 0.05;
    layers[1].scatteringCoefficient = 20.0;
    //layers[1].scatteringCoefficient = 15.0;
    layers[1].refractiveIndex = 1.4;
    layers[1].anisotropy = 0.82;
    //layers[1].anisotropy = 0.86;

    layers[2].numberOfSurfaces = 2;
    layers[2].surfaceId = new int[layers[2].numberOfSurfaces];
    layers[2].surfaceId[0] = 1;
    layers[2].surfaceId[1] = 2;
    layers[2].absorptionCoefficient = 0.035;
    //layers[2].absorptionCoefficient = 0.025;
    layers[2].scatteringCoefficient = 36.0;
    //layers[2].scatteringCoefficient = 28.0;
    layers[2].refractiveIndex = 1.55;
    layers[2].anisotropy = 0.925;
    //layers[2].anisotropy = 0.94;

    layers[3].numberOfSurfaces = 2;
    layers[3].surfaceId = new int[layers[3].numberOfSurfaces];
    layers[3].surfaceId[0] = 2;
    layers[3].surfaceId[1] = 3;
    layers[3].absorptionCoefficient = 0.001;
    //layers[3].absorptionCoefficient = 0.001;
    layers[3].scatteringCoefficient = 0.1;
    //layers[3].scatteringCoefficient = 0.1;
    layers[3].refractiveIndex = 1.4;
    layers[3].anisotropy = 0.999;
    //layers[3].anisotropy = 0.999;

    layers[4].numberOfSurfaces = 2;
    layers[4].surfaceId = new int[layers[4].numberOfSurfaces];
    layers[4].surfaceId[0] = 3;
    layers[4].surfaceId[1] = 4;
    layers[4].absorptionCoefficient = 0.05;
    //layers[4].absorptionCoefficient = 0.03;
    layers[4].scatteringCoefficient = 60.0;
    //layers[4].scatteringCoefficient = 60.0;
    layers[4].refractiveIndex = 1.4;
    layers[4].anisotropy = 0.95;
    //layers[4].anisotropy = 0.96;

    layers[5].numberOfSurfaces = 2;
    layers[5].surfaceId = new int[layers[5].numberOfSurfaces];
    layers[5].surfaceId[0] = 4;
    layers[5].surfaceId[1] = 5;
    layers[5].absorptionCoefficient = 0.02;
    //layers[5].absorptionCoefficient = 0.01;
    layers[5].scatteringCoefficient = 50.0;
    //layers[5].scatteringCoefficient = 55.0;
    layers[5].refractiveIndex = 1.4;
    layers[5].anisotropy = 0.8;
    //layers[5].anisotropy = 0.85;

    return layers;
}

static Detector* GetDetectors()
{
    int numberOfDetectors = 60;
    Detector* detector = new Detector[numberOfDetectors];
    for (int i = 0; i < numberOfDetectors; ++i)
    {
        double3 center = {0.5 + 1.0 * i, 0.0, 0.0};
        double3 length = {1.0, 10.0, 1.0};
        detector[i].length = length;
        detector[i].center = center;
    }

    return detector;
}

static InputInfo* GetInput(int photons)
{
    InputInfo* input = new InputInfo();

    input->area = GetArea();
    input->numberOfSurfaces = 6;

    Surface* surfaces = new Surface[input->numberOfSurfaces];
    surfaces[0] = GetPlane(0.0, 2);
    surfaces[1] = GetPlane(3.0, 2);
    surfaces[2] = GetPlane(13.0, 2);
    surfaces[3] = GetPlane(15.0, 2);
    surfaces[4] = GetPlane(19.0, 2);
    surfaces[5] = GetPlane(39.0, 2);

    input->surface = surfaces;
    
    input->minWeight = MIN_WEIGHT;
    
    input->numberOfLayers = 6;
    input->layerInfo = GetLayers();

    input->numberOfDetectors = 60;
    input->detector = GetDetectors();
    
    input->numberOfPhotons = photons;

    return input;
}

static OutputInfo* GetOutput(Area* area)
{
    OutputInfo* output = new OutputInfo();
    int absorbtionSize = area->partitionNumber.x * 
        area->partitionNumber.y * area->partitionNumber.z;
    output->absorption = new double[absorbtionSize];
    output->absorptionSize = absorbtionSize;
    for (int i = 0; i < absorbtionSize; ++i)
    {
        output->absorption[i] = 0.0;
    }
    output->numberOfDetectors = 60;
    output->weigthInDetector = new double[output->numberOfDetectors];
    for (int i = 0; i < output->numberOfDetectors; ++i)
    {
        output->weigthInDetector[i] = 0.0;
    }

    return output;
}

static double ScaleAbsorbtion(OutputInfo* output, int photons)
{
    double absorbtion = 0.0;
    for (int i = 0; i < output->absorptionSize; ++i)
    {
        absorbtion += output->absorption[i];
    }
    absorbtion /= photons;

    return absorbtion;
}

static void FreeMemory(InputInfo* input, OutputInfo* output)
{
    for (int i = 0; i < input->numberOfLayers; ++i)
        delete[] input->layerInfo[i].surfaceId;
    delete[] input->layerInfo;
    
    for (int i = 0; i < input->numberOfSurfaces; ++i)
        delete[] input->surface[i].vertices;
    delete[] input->surface;

    delete input->area;
    delete[] input->detector;

    delete input;

    delete[] output->absorption;
    delete[] output->weigthInDetector;
    delete output;
}

void BrainTest(int photons)
{
    clock_t start, finish;

    InputInfo* input = GetInput(photons);
    OutputInfo* output = GetOutput(input->area);

    start = clock();

    MCG59* randomGenerator = new MCG59();
    InitMCG59(randomGenerator, 777, 0, 1);

    output->specularReflectance = ComputeSpecularReflectance(input->layerInfo);
    for (int i = 0; i < input->numberOfPhotons; ++i)
    {
        ComputePhoton(output->specularReflectance, input, output, randomGenerator);
        if ((i % 10000) == 0)
        {
            printf(".");
        }
    }
    printf("\n");

    finish = clock();

    printf("Write data into file...");
    int error = WriteOutputToFile(input, output, "brain_test.mcml.out");
    printf("%s\n", error == 0 ? "OK" : "FALSE");

    double absorbtion = ScaleAbsorbtion(output, photons);
    printf("Middle Trajectory = %.6f\n", absorbtion);
    printf("Time: %.2f sec\n", (double)(finish - start) / CLOCKS_PER_SEC);
    for (int i = 0; i < input->numberOfDetectors; ++i)
    {
        printf("Detector %d: %.10f\n", i, output->weigthInDetector[i] / photons);
    }

    delete randomGenerator;
    FreeMemory(input, output);
}
