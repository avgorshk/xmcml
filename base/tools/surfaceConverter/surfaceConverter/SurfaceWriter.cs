using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace surfaceConverter
{
    class SurfaceWriter
    {
        const uint MCML_SECTION_SURFACES = 0x0100;
        
        private Surface[] surface = null;

        public SurfaceWriter(Surface[] surface)
        {
            this.surface = surface;
        }

        public void Write(string fileName)
        {
            BinaryWriter writer = new BinaryWriter(File.Open(fileName, FileMode.OpenOrCreate));
            writer.Write(MCML_SECTION_SURFACES);
            writer.Write(surface.Length);
            foreach (Surface s in surface)
            {
                writer.Write(s.vertices.Length);
                foreach (double3 v in s.vertices)
                {
                    writer.Write(v.x);
                    writer.Write(v.y);
                    writer.Write(v.z);
                }

                writer.Write(s.triangles.Length);
                foreach (int3 t in s.triangles)
                {
                    writer.Write(t.x);
                    writer.Write(t.y);
                    writer.Write(t.z);
                }
            }
            writer.Flush();
            writer.Close();
        }
    }
}
