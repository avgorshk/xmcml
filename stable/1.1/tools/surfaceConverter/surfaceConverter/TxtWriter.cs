using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace surfaceConverter
{
    class TxtWriter
    {
        private TextWriter writer = null;

        public TxtWriter(string fileName)
        {
            writer = new StreamWriter(fileName);
        }

        public void Write(Triangle[] triangles)
        {
            var format = new System.Globalization.NumberFormatInfo
                {
                    NumberDecimalSeparator = "."
                };

            writer.WriteLine(triangles.Length);
            writer.WriteLine();
            foreach (Triangle t in triangles)
            {
                writer.WriteLine(string.Format("{0} {1} {2}", t.a.x.ToString("F6", format), t.a.y.ToString("F6", format), t.a.z.ToString("F6", format)));
                writer.WriteLine(string.Format("{0} {1} {2}", t.b.x.ToString("F6", format), t.b.y.ToString("F6", format), t.b.z.ToString("F6", format)));
                writer.WriteLine(string.Format("{0} {1} {2}", t.c.x.ToString("F6", format), t.c.y.ToString("F6", format), t.c.z.ToString("F6", format)));
                writer.WriteLine();
            }
            writer.Flush();
        }
    }
}
