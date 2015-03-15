using System;
using System.Collections.Generic;
using System.Text;

namespace surfaceConverter
{
    static class VectorMath
    {
        public const double EPSILON = 1.0E-6;
        public const double MAX_DISTANCE = 1.0E+256;

        public static double3 SubVector(double3 a, double3 b)
        {
            double3 result;
            result.x = a.x - b.x;
            result.y = a.y - b.y;
            result.z = a.z - b.z;
            return result;
        }

        public static double DotVector(double3 a, double3 b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        public static double LengthOfVector(double3 a)
        {
            return Math.Sqrt(DotVector(a, a));
        }

        public static double3 NormalizeVector(double3 a)
        {
            double length = LengthOfVector(a);
            double3 result;
            result.x = a.x / length;
            result.y = a.y / length;
            result.z = a.z / length;
            return result;
        }

        public static double3 CrossVector(double3 a, double3 b)
        {
            double3 result;
            result.x = a.y * b.z - a.z * b.y;
            result.y = a.z * b.x - a.x * b.z;
            result.z = a.x * b.y - a.y * b.x;
            return result;
        }

        public static double GetTriangleIntersectionDistance(double3 origin, 
            double3 direction, double3 a, double3 b, double3 c)
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
    }
}
