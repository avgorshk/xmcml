#include "mcml_intersection.h"
#include "mcml_math.h"

bool IntersectAABB(AABB Box, floatVec3 origin, floatVec3 direction, float& t_near, float& t_far)
{
	float tmin = - MAX_DISTANCE, tmax = MAX_DISTANCE;
	float t1, t2, t;
	for ( int i = 0; i < 3; i++ ) {
		if ( abs(direction.cell[i]) > EPSILON ) {
			t1 = ( Box.Ver1.cell[i] - origin.cell[i] ) / direction.cell[i];
			t2 = ( Box.Ver2.cell[i] - origin.cell[i] ) / direction.cell[i];
			if ( t2 < t1 ) { t = t1; t1 = t2; t2 = t; }
			if ( t1 > tmin ) tmin = t1;
			if ( t2 < tmax ) tmax = t2;
			if ( tmin > tmax || tmax < 0) 
			{
				t_near = MAX_DISTANCE;
				t_far = MAX_DISTANCE;
				return false;
			}
		}
		else {
			if ((Box.Ver1.cell[i] - origin.cell[i] > 0) || (Box.Ver2.cell[i] - origin.cell[i] < 0)) 
			{
				t_near = MAX_DISTANCE;
				t_far = MAX_DISTANCE;
				return false;
			}
		}
	}
	t_near = tmin; t_far = tmax; 
	return true;
}

float GetTriangleIntersectionDistance(floatVec3 origin, floatVec3 direction, floatVec3 a, floatVec3 b, floatVec3 c)
{
    floatVec3 edge1;
	floatVec3 edge2;
	floatVec3 tvec; 
	floatVec3 pvec;
	floatVec3 qvec;

	float det, inv_det;
    float u, v;

	edge1 = SubVector(b, a);
	edge2 = SubVector(c, a);
	
	pvec = CrossVector(direction, edge2);

	det = DotVector(edge1, pvec);

	if (det < EPSILON && det > -EPSILON)
	{
		return MAX_DISTANCE;
	}
	inv_det = 1.f / det;

	tvec = SubVector(origin, a);

	u = DotVector(tvec, pvec) * inv_det;
	
	if (u < 0.f || u > 1.f)
	{
		return MAX_DISTANCE;
	}

	qvec = CrossVector(tvec, edge1);
	v = DotVector(direction, qvec) * inv_det;

	if (v < 0.f || u + v > 1.f)
	{
		return MAX_DISTANCE;
	}

	return DotVector(edge2, qvec) * inv_det;
}

IntersectionInfo ComputeSurfaceIntersection(floatVec3 origin, floatVec3 direction, Surface& surface)
{
    IntersectionInfo result;
    result.isFindIntersection = 0;

    int triangle_index = -1;
    float distance;
    float minimal_distance = MAX_DISTANCE;
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

IntersectionInfo ComputeIntersection(floatVec3 origin, floatVec3 direction, Surface* surfaces, int numberOfSurfaces)
{
    IntersectionInfo result;
    result.isFindIntersection = 0;	
    result.distance = MAX_DISTANCE;
	int tr_index = -1;

    IntersectionInfo surfaceResult;
    int surfaceId = -1;
    float minimal_distance = MAX_DISTANCE;
    floatVec3 normal;
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