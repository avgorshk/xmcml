using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace surfaceConverter
{
    class Surface
    {
        public double3[] vertices { get; private set; }
        public int3[] triangles { get; private set; }

        public Surface(Triangle[] triangleArray)
        {
            List<double3> vertice = new List<double3>();
            List<int3> triangle = new List<int3>();

            int counter = 0;
            foreach (Triangle t in triangleArray)
            {
                int3 currentTriangle = new int3();

                // a
                currentTriangle.x = vertice.Count;
                for (int i = 0; i < vertice.Count; ++i)
                {
                    bool isEqual = ((t.a.x == vertice[i].x) && (t.a.y == vertice[i].y) && (t.a.z == vertice[i].z));
                    if (isEqual)
                    {
                        currentTriangle.x = i;
                        break;
                    }
                }
                if (currentTriangle.x == vertice.Count)
                {
                    vertice.Add(t.a);
                }

                // b
                currentTriangle.y = vertice.Count;
                for (int i = 0; i < vertice.Count; ++i)
                {
                    bool isEqual = ((t.b.x == vertice[i].x) && (t.b.y == vertice[i].y) && (t.b.z == vertice[i].z));
                    if (isEqual)
                    {
                        currentTriangle.y = i;
                        break;
                    }
                }
                if (currentTriangle.y == vertice.Count)
                {
                    vertice.Add(t.b);
                }

                // c
                currentTriangle.z = vertice.Count;
                for (int i = 0; i < vertice.Count; ++i)
                {
                    bool isEqual = ((t.c.x == vertice[i].x) && (t.c.y == vertice[i].y) && (t.c.z == vertice[i].z));
                    if (isEqual)
                    {
                        currentTriangle.z = i;
                        break;
                    }
                }
                if (currentTriangle.z == vertice.Count)
                {
                    vertice.Add(t.c);
                }

                // Add triangle
                if ((currentTriangle.x != currentTriangle.y) && (currentTriangle.y != currentTriangle.z) &&
                    (currentTriangle.z != currentTriangle.x))
                {
                    triangle.Add(currentTriangle);
                }

                ++counter;

                if (counter % 10000 == 0)
                {
                    Console.Write(".");
                }
            }

            vertices = vertice.ToArray();
            triangles = triangle.ToArray();
        }

        public void Translate(double x, double y, double z)
        {
            for (int i = 0; i < vertices.Length; ++i)
            {
                vertices[i].x += x;
                vertices[i].y += y;
                vertices[i].z += z;
            }
        }

        public void Scale(double coeff)
        {
            for (int i = 0; i < vertices.Length; ++i)
            {
                vertices[i].x *= coeff;
                vertices[i].y *= coeff;
                vertices[i].z *= coeff;
            }
        }
    }
}
