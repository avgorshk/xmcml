#include <math.h>
#include "mcml_math.h"

#define MAX_VALUE 1.0E+256

double3 SubVector(double3 a, double3 b)
{
    double3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

double3 DivVector(double3 a, double b)
{
    double3 result;
    result.x = a.x / b;
    result.y = a.y / b;
    result.z = a.z / b;
    return result;
}

double3 MulVector(double3 a, double b)
{
    double3 result;
    result.x = a.x * b;
    result.y = a.y * b;
    result.z = a.z * b;
    return result;
}

double DotVector(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double LengthOfVector(double3 a)
{
    return sqrt(DotVector(a, a));
}

double3 NormalizeVector(double3 a)
{
    double length = LengthOfVector(a);
    return DivVector(a, length);
}

double3 CrossVector(double3 a, double3 b)
{
    double3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

double3 GetPlaneNormal(double3 a, double3 b, double3 c)
{
    double3 vector1 = SubVector(a, b);
	double3 vector2 = SubVector(a, c);
	return CrossVector(vector1, vector2);
}

double3 GetPlaneSegmentIntersectionPoint(double3 a, double3 b, double z)
{
    double3 n = {0.0, 0.0, 1.0};
    double3 v = {-a.x, -a.y, z - a.z};
    double d = DotVector(n, v);
    double3 w = SubVector(b, a);
    double e = DotVector(n, w);
    
    double3 result;
    if (e != 0.0)
    {
        result.x = a.x + w.x * (d / e);
        result.y = a.y + w.y * (d / e);
        result.z = a.z + w.z * (d / e);
        if (DotVector(SubVector(a, result), SubVector(b, result)) > 0.0)
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
