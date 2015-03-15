using System;
using System.Collections.Generic;
using System.Text;

namespace surfaceConverter
{
    class Surface
    {
        public double3[] vertices { get; private set; }
        public int3[] triangles { get; set; }

        public Surface(double3[] v, int3[] t)
        {
            vertices = v;
            triangles = t;
        }

        public Surface(Surface copy)
        {
            this.vertices = new double3[copy.vertices.Length];
            for (int i = 0; i < copy.vertices.Length; ++i)
            {
                this.vertices[i] = copy.vertices[i];
            }

            this.triangles = new int3[copy.triangles.Length];
            for (int i = 0; i < copy.triangles.Length; ++i)
            {
                this.triangles[i] = copy.triangles[i];
            }
        }

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

        public void Scale(double cx, double cy, double cz)
        {
            for (int i = 0; i < vertices.Length; ++i)
            {
                vertices[i].x *= cx;
                vertices[i].y *= cy;
                vertices[i].z *= cz;
            }
        }

        public double ComputeIntersection(double3 origin, double3 direction)
        {
            Surface surface = this;

            int triangle_index = -1;
            double distance;
            double minimal_distance = VectorMath.MAX_DISTANCE;
            for (int i = 0; i < surface.triangles.Length; ++i)
            {
                int3 triangle = surface.triangles[i];
                distance = VectorMath.GetTriangleIntersectionDistance(origin, direction, surface.vertices[triangle.x],
                    surface.vertices[triangle.y], surface.vertices[triangle.z]);
                if (distance >= 0.0 && distance < minimal_distance)
                {
                    minimal_distance = distance;
                    triangle_index = i;
                }
            }

            return minimal_distance;
        }

        public void Truncate(double z)
        {
            List<int3> finish_triangles = new List<int3>();

            foreach (int3 t in triangles)
            {
                if ((vertices[t.x].z + vertices[t.y].z + vertices[t.z].z) / 3.0 < z)
                    finish_triangles.Add(t);
            }

            triangles = finish_triangles.ToArray();

            List<double3> finish_vertices = new List<double3>();
            for (int k = 0; k < triangles.Length; ++k)
            {
                //x
                bool isFound = false;
                for (int i = 0; i < finish_vertices.Count; ++i)
                {
                    bool isEqual = (vertices[triangles[k].x].x == finish_vertices[i].x) &&
                        (vertices[triangles[k].x].y == finish_vertices[i].y) &&
                        (vertices[triangles[k].x].z == finish_vertices[i].z);
                    if (isEqual)
                    {
                        triangles[k].x = i;
                        isFound = true;
                        break;
                    }
                }
                if (!isFound)
                {
                    finish_vertices.Add(vertices[triangles[k].x]);
                    triangles[k].x = finish_vertices.Count - 1;
                }

                //y
                isFound = false;
                for (int i = 0; i < finish_vertices.Count; ++i)
                {
                    bool isEqual = (vertices[triangles[k].y].x == finish_vertices[i].x) &&
                        (vertices[triangles[k].y].y == finish_vertices[i].y) &&
                        (vertices[triangles[k].y].z == finish_vertices[i].z);
                    if (isEqual)
                    {
                        triangles[k].y = i;
                        isFound = true;
                        break;
                    }
                }
                if (!isFound)
                {
                    finish_vertices.Add(vertices[triangles[k].y]);
                    triangles[k].y = finish_vertices.Count - 1;
                }

                //z
                isFound = false;
                for (int i = 0; i < finish_vertices.Count; ++i)
                {
                    bool isEqual = (vertices[triangles[k].z].x == finish_vertices[i].x) &&
                        (vertices[triangles[k].z].y == finish_vertices[i].y) &&
                        (vertices[triangles[k].z].z == finish_vertices[i].z);
                    if (isEqual)
                    {
                        triangles[k].z = i;
                        isFound = true;
                        break;
                    }
                }
                if (!isFound)
                {
                    finish_vertices.Add(vertices[triangles[k].z]);
                    triangles[k].z = finish_vertices.Count - 1;
                }

                if (finish_vertices.Count % 10000 == 0)
                    Console.Write(".");
            }

            vertices = finish_vertices.ToArray();
        }

        public void RotateZ(double alpha)
        {
            double3 result;
            for (int i = 0; i < vertices.Length; ++i)
            {
                result.x = Math.Cos(alpha) * vertices[i].x - Math.Sin(alpha) * vertices[i].y;
                result.y = Math.Sin(alpha) * vertices[i].x + Math.Cos(alpha) * vertices[i].y;
                result.z = vertices[i].z;
                vertices[i] = result;
            }
        }
    }
}
