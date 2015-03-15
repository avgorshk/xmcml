using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace surfaceVisualizer
{
    struct int3
    {
        public int x, y, z;
    }

    struct double3
    {
        public double x, y, z;

        public float3 ToFloat()
        {
            float3 result = new float3();
            result.x = (float)x;
            result.y = (float)y;
            result.z = (float)z;
            return result;
        }
    }

    struct float3
    {
        public float x, y, z;

        public static float3 SubVector(float3 a, float3 b)
        {
            float3 result;
            result.x = a.x - b.x;
            result.y = a.y - b.y;
            result.z = a.z - b.z;
            return result;
        }

        public static float3 CrossVector(float3 a, float3 b)
        {
            float3 result;
            result.x = a.y * b.z - a.z * b.y;
            result.y = a.z * b.x - a.x * b.z;
            result.z = a.x * b.y - a.y * b.x;
            return result;
        }

        public static float3 GetPlaneNormal(float3 a, float3 b, float3 c)
        {
            float3 vector1 = SubVector(a, b);
            float3 vector2 = SubVector(a, c);
            return CrossVector(vector1, vector2);
        }
    }

    struct Triangle
    {
        public float3 a, b, c;

        public float3 GetMin(float3 min)
        {
            if (a.x < min.x) min.x = a.x;
            if (b.x < min.x) min.x = b.x;
            if (c.x < min.x) min.x = c.x;

            if (a.y < min.y) min.y = a.y;
            if (b.y < min.y) min.y = b.y;
            if (c.y < min.y) min.y = c.y;

            if (a.z < min.z) min.z = a.z;
            if (b.z < min.z) min.z = b.z;
            if (c.z < min.z) min.z = c.z;

            return min;
        }

        public float3 GetMax(float3 max)
        {
            if (a.x > max.x) max.x = a.x;
            if (b.x > max.x) max.x = b.x;
            if (c.x > max.x) max.x = c.x;

            if (a.y > max.y) max.y = a.y;
            if (b.y > max.y) max.y = b.y;
            if (c.y > max.y) max.y = c.y;

            if (a.z > max.z) max.z = a.z;
            if (b.z > max.z) max.z = b.z;
            if (c.z > max.z) max.z = c.z;

            return max;
        }

        public float3 GetNormal()
        {
            return float3.GetPlaneNormal(a, b, c);
        }
    }

}
