using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace surfaceVisualizer
{
    class SurfaceReader
    {
        public Surface[] surface { get; private set; }

        public SurfaceReader(string fileName)
        {
            BinaryReader reader = new BinaryReader(File.Open(fileName, FileMode.Open));
            reader.ReadUInt32(); //section id
            int numberOfSurfaces = reader.ReadInt32();
            surface = new Surface[numberOfSurfaces];
            for (int i = 0; i < numberOfSurfaces; ++i)
            {
                int numberOfVertices = reader.ReadInt32();
                double3[] vertices = new double3[numberOfVertices];
                for (int j = 0; j < numberOfVertices; ++j)
                {
                    vertices[j] = new double3();
                    vertices[j].x = reader.ReadDouble();
                    vertices[j].y = reader.ReadDouble();
                    vertices[j].z = reader.ReadDouble();
                }

                int numberOfTriangles = reader.ReadInt32();
                int3[] triangles = new int3[numberOfTriangles];
                for (int j = 0; j < numberOfTriangles; ++j)
                {
                    triangles[j] = new int3();
                    triangles[j].x = reader.ReadInt32();
                    triangles[j].y = reader.ReadInt32();
                    triangles[j].z = reader.ReadInt32();

                }

                Triangle[] triangle = new Triangle[numberOfTriangles];
                for (int j = 0; j < numberOfTriangles; ++j)
                {
                    triangle[j] = new Triangle();
                    triangle[j].a = vertices[triangles[j].x].ToFloat();
                    triangle[j].b = vertices[triangles[j].y].ToFloat();
                    triangle[j].c = vertices[triangles[j].z].ToFloat();
                }

                surface[i] = new Surface(triangle);
            }
        }
    }
}
