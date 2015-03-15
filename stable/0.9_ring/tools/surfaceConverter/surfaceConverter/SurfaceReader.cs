using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace surfaceConverter
{
    class SurfaceReader
    {
        public Surface[] surface {get; private set;}

        public SurfaceReader(string fileName)
        {
            BinaryReader reader = new BinaryReader(File.Open(fileName, FileMode.Open));
            reader.ReadUInt32();
            int numberOfSurface = reader.ReadInt32();
            surface = new Surface[numberOfSurface];
            for (int i = 0; i < numberOfSurface; ++i)
            {
                int numberOfVertices = reader.ReadInt32();
                double3[] vertices = new double3[numberOfVertices];
                for (int j = 0; j < numberOfVertices; ++j)
                {
                    vertices[j].x = reader.ReadDouble();
                    vertices[j].y = reader.ReadDouble();
                    vertices[j].z = reader.ReadDouble();
                }

                int numberOfTriangles = reader.ReadInt32();
                int3[] triangles = new int3[numberOfTriangles];
                for (int j = 0; j < numberOfTriangles; ++j)
                {
                    triangles[j].x = reader.ReadInt32();
                    triangles[j].y = reader.ReadInt32();
                    triangles[j].z = reader.ReadInt32();
                }

                surface[i] = new Surface(vertices, triangles);
            }
        }
    }
}
