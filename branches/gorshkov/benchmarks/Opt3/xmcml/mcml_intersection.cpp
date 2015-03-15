#include "mcml_intersection.h"
#include "mcml_math.h"

#define EPSILON 1E-6
#define MAX_DISTANCE 1.0E+256

double GetTriangleIntersectionDistance(double3 origin, double3 direction, double3 a, double3 b, double3 c)
{
    double3 edge1;
	double3 edge2;
	double3 tvec; 
	double3 pvec;
	double3 qvec;

	double det, inv_det;
    double u, v;

	edge1 = SubVector(b, a);
	edge2 = SubVector(c, a);
	
	pvec = CrossVector(direction, edge2);

	det = DotVector(edge1, pvec);

	if (det < EPSILON && det > -EPSILON)
	{
		return MAX_DISTANCE;
	}
	inv_det = 1.0 / det;

	tvec = SubVector(origin, a);

	u = DotVector(tvec, pvec) * inv_det;
	
	if (u < 0.0 || u > 1.0)
	{
		return MAX_DISTANCE;
	}

	qvec = CrossVector(tvec, edge1);
	v = DotVector(direction, qvec) * inv_det;

	if (v < 0.0 || u + v > 1.0)
	{
		return MAX_DISTANCE;
	}

	return DotVector(edge2, qvec) * inv_det;
}

IntersectionInfo ComputeSurfaceIntersection(double3 origin, double3 direction, Surface surface)
{
    IntersectionInfo result;
    result.isFindIntersection = 0;

    int triangle_index = -1;
    double distance;
    double minimal_distance = MAX_DISTANCE;
    for (int i = 0; i < surface.numberOfTriangles; ++i)
    {
        int3 triangle = surface.triangles[i];
        distance = GetTriangleIntersectionDistance(origin, direction, surface.vertices[triangle.x], 
            surface.vertices[triangle.y], surface.vertices[triangle.z]);
        if (distance >= 0.0 && distance < minimal_distance)
        {
            minimal_distance = distance;
            triangle_index = i;
        }
    }
    
    if (triangle_index >= 0)
    {
        int3 triangle = surface.triangles[triangle_index];
        result.isFindIntersection = 1;
        result.distance = minimal_distance;
        result.normal = GetPlaneNormal(surface.vertices[triangle.x], surface.vertices[triangle.y],
            surface.vertices[triangle.z]);
        result.surfaceId = -1;
    }

    return result;
}

IntersectionInfo ComputeIntersection(double3 origin, double3 direction, Surface* surfaces, int numberOfSurfaces)
{
    IntersectionInfo result;
    result.isFindIntersection = 0;

    IntersectionInfo surfaceResult;
    int surfaceId = -1;
    double minimal_distance = MAX_DISTANCE;
    double3 normal;
    for (int i = 0; i < numberOfSurfaces; ++i)
    {
        surfaceResult = ComputeSurfaceIntersection(origin, direction, surfaces[i]);
        if (surfaceResult.isFindIntersection)
        {
            if (surfaceResult.distance < minimal_distance)
            {
                surfaceId = i;
                minimal_distance = surfaceResult.distance;
                normal = surfaceResult.normal;
            }
        }
    }

    if (surfaceId >= 0)
    {
        result.isFindIntersection = 1;
        result.distance = minimal_distance;
        result.normal = NormalizeVector(normal);
        result.surfaceId = surfaceId;
    }

    return result;
}

IntersectionInfo GetIntersectionInfo(PhotonState* photon, InputInfo* input)
{
    int layerId = photon->layerId;
    int numberOfSurfaces = input->layerInfo[layerId].numberOfSurfaces;
    Surface* surfaces = new Surface[numberOfSurfaces];

    for (int i = 0; i < numberOfSurfaces; ++i)
    {
        surfaces[i] = input->surface[input->layerInfo[layerId].surfaceId[i]];
    }

    IntersectionInfo intersection = ComputeIntersection(photon->position, photon->direction,
        surfaces, numberOfSurfaces);

    if (intersection.isFindIntersection)
    {
        intersection.surfaceId = input->layerInfo[layerId].surfaceId[intersection.surfaceId];
    }

    delete[] surfaces;

    return intersection;
}