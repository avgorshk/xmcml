#include "mcml_math.h"

#define MAX_VALUE 1.0E+38f

floatVec3 SubVector(floatVec3 a, floatVec3 b)
{
    floatVec3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

float DotVector(floatVec3 a, floatVec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float LengthOfVector(floatVec3 a)
{
    return sqrt(DotVector(a, a));
}

floatVec3 NormalizeVector(floatVec3 a)
{
    float length = LengthOfVector(a);
    floatVec3 result;
    result.x = a.x / length;
    result.y = a.y / length;
    result.z = a.z / length;
    return result;
}

floatVec3 CrossVector(floatVec3 a, floatVec3 b)
{
    floatVec3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

floatVec3 GetPlaneNormal(floatVec3 a, floatVec3 b, floatVec3 c)
{
    floatVec3 vector1 = SubVector(a, b);
	floatVec3 vector2 = SubVector(a, c);
	return CrossVector(vector1, vector2);
}

floatVec3 GetPlaneSegmentIntersectionPoint(floatVec3 a, floatVec3 b, float z)
{
    floatVec3 n = {0.0, 0.0, 1.0};
    floatVec3 v = {-a.x, -a.y, z - a.z};
    float d = DotVector(n, v);
    floatVec3 w = SubVector(b, a);
    float e = DotVector(n, w);
    
    floatVec3 result;
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
