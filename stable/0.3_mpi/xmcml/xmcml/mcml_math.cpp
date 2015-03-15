#include <math.h>
#include "mcml_math.h"

double3 SubVector(double3 a, double3 b)
{
    double3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
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
    double3 result;
    result.x = a.x / length;
    result.y = a.y / length;
    result.z = a.z / length;
    return result;
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