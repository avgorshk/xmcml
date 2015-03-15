#include <math.h>
#include "mcml_math.h"

#define MAX_VALUE 1.0E+256

Double3 SubVector(Double3 a, Double3 b)
{
    Double3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

Double3 DivVector(Double3 a, double b)
{
    Double3 result;
    result.x = a.x / b;
    result.y = a.y / b;
    result.z = a.z / b;
    return result;
}

double DotVector(Double3 a, Double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double LengthOfVector(Double3 a)
{
    return sqrt(DotVector(a, a));
}

Double3 NormalizeVector(Double3 a)
{
    double length = LengthOfVector(a);
    return DivVector(a, length);
}

Double3 CrossVector(Double3 a, Double3 b)
{
    Double3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

Double3 GetPlaneNormal(Double3 a, Double3 b, Double3 c)
{
    Double3 vector1 = SubVector(a, b);
	Double3 vector2 = SubVector(a, c);
	return CrossVector(vector1, vector2);
}

Double3 GetPlaneSegmentIntersectionPoint(Double3 a, Double3 b, double z)
{
    Double3 n = {0.0, 0.0, 1.0};
    Double3 v = {-a.x, -a.y, z - a.z};
    double d = DotVector(n, v);
    Double3 w = SubVector(b, a);
    double e = DotVector(n, w);
    
    Double3 result;
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
