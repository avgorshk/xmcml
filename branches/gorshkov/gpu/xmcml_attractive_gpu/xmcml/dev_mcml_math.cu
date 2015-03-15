#include <math.h>

#define MAX_VALUE 1.0E+256

__device__ Double3 gpuSubVector(Double3 a, Double3 b)
{
    Double3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

__device__ Double3 gpuDivVector(Double3 a, double b)
{
    Double3 result;
    result.x = a.x / b;
    result.y = a.y / b;
    result.z = a.z / b;
    return result;
}

__device__ double gpuDotVector(Double3 a, Double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ double gpuLengthOfVector(Double3 a)
{
    return sqrt(gpuDotVector(a, a));
}

__device__ Double3 gpuNormalizeVector(Double3 a)
{
    double length = gpuLengthOfVector(a);
    return gpuDivVector(a, length);
}

__device__ Double3 gpuCrossVector(Double3 a, Double3 b)
{
    Double3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

__device__ Double3 gpuGetPlaneNormal(Double3 a, Double3 b, Double3 c)
{
    Double3 vector1 = gpuSubVector(a, b);
	Double3 vector2 = gpuSubVector(a, c);
	return gpuCrossVector(vector1, vector2);
}

__device__ Double3 gpuGetPlaneSegmentIntersectionPoint(Double3 a, Double3 b, double z)
{
    Double3 n = {0.0, 0.0, 1.0};
    Double3 v = {-a.x, -a.y, z - a.z};
    double d = gpuDotVector(n, v);
    Double3 w = gpuSubVector(b, a);
    double e = gpuDotVector(n, w);
    
    Double3 result;
    if (e != 0.0)
    {
        result.x = a.x + w.x * (d / e);
        result.y = a.y + w.y * (d / e);
        result.z = a.z + w.z * (d / e);
        if (gpuDotVector(gpuSubVector(a, result), gpuSubVector(b, result)) > 0.0)
        {
            result.x = MAX_VALUE;
            result.y = MAX_VALUE;
            result.z = MAX_VALUE;
        }
    }
    else
    {
        result.x = MAX_VALUE;
        result.y = MAX_VALUE;
        result.z = MAX_VALUE;
    }

    return result;
}
