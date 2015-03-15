using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace surfaceConverter
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Too few arguments\n");
                return;
            }

            TxtReader txtReader = new TxtReader(args[0]);
            Surface[] surface = new Surface[1];
            surface[0] = new Surface(txtReader.triangle);
            surface[0].Translate(-0.1960065, -0.1553699, -0.165087953);
            surface[0].Scale(160.0);
            SurfaceWriter writer = new SurfaceWriter(surface);
            writer.Write(args[1]);
        }
    }
}
