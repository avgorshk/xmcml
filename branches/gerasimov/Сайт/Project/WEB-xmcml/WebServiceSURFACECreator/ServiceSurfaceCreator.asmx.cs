using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;

namespace WebServiceSURFACECreator
{
    /// <summary>
    /// Сводное описание для Service1
    /// </summary>
    [WebService(Namespace = "WEB-xmcml")]
    [WebServiceBinding(ConformsTo = WsiProfiles.BasicProfile1_1)]
    [System.ComponentModel.ToolboxItem(false)]
    // Чтобы разрешить вызывать веб-службу из скрипта с помощью ASP.NET AJAX, раскомментируйте следующую строку. 
    // [System.Web.Script.Services.ScriptService]
    public class Service1 : System.Web.Services.WebService
    {
        private const int MCML_SECTION_SURFACES = 0x0100;
        private const String TMP_SURFACE_FOLDER = "TMP_SURFACE_FOLDER";

        private const String FILE_PATHS = "D://PATHS.cfg";

        private struct double3 
        {
            public double x, y, z;
        };

        private struct int3
        {
            public int x, y, z;
        };

        private struct Surface
        {
	        public double3[] vertices;
	        public int numberOfVertices;
            public int3[] triangles;
            public int numberOfTriangles;
        };

        private String GetValueFromPaths(String tag)
        {
            if (!System.IO.File.Exists(FILE_PATHS))
            {
                return null;
            }

            System.IO.StreamReader reader = new System.IO.StreamReader(FILE_PATHS);

            int first_index, last_index;
            String line, value = null;

            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();

                first_index = line.IndexOf(tag);

                if (first_index != -1)
                {
                    first_index = line.IndexOf("\"") + 1;
                    last_index = line.LastIndexOf("\"");

                    if ((first_index == -1) || (last_index == -1))
                    {
                        continue;
                    }

                    value = line.Substring(first_index, last_index - first_index);
                    break;
                }
            }
            reader.Close();

            return value;
        }

        private void WriteDouble3(System.IO.BinaryWriter writer, double3 vector)
        {
            writer.Write(vector.x);
            writer.Write(vector.y);
            writer.Write(vector.z);
        }

        private void WriteInt3(System.IO.BinaryWriter writer, int3 vector)
        {
            writer.Write(vector.x);
            writer.Write(vector.y);
            writer.Write(vector.z);
        }

        private void WriteSurface(System.IO.BinaryWriter writer, Surface surface)
        {
            writer.Write(surface.numberOfVertices);

            for (int i = 0; i < surface.numberOfVertices; ++i)
            {
                WriteDouble3(writer, surface.vertices[i]);
            }

            writer.Write(surface.numberOfTriangles);

            for (int i = 0; i < surface.numberOfTriangles; ++i)
            {
                WriteInt3(writer, surface.triangles[i]);
            }
        }

        private void WriteOutputToFile(Surface[] surface, int numberOfSurfaces, String pathSURFACEFile)
        {
            System.IO.BinaryWriter writer
                = new System.IO.BinaryWriter(System.IO.File.Open(pathSURFACEFile, System.IO.FileMode.Append));

            UInt32 section_id = MCML_SECTION_SURFACES;
            writer.Write(section_id);

            writer.Write(numberOfSurfaces);

            for (int i = 0; i < numberOfSurfaces; ++i)
            {
                WriteSurface(writer, surface[i]);
            }

            writer.Flush();
            writer.Close();
        }

        private Surface GeneratePlane(double3 center, double3 length)
        {
            Surface plane = new Surface();

            plane.numberOfVertices = 4;
            plane.vertices = new double3[plane.numberOfVertices];

            plane.vertices[0].x = center.x - length.x / 2.0;
            plane.vertices[0].y = center.y - length.y / 2.0;
            plane.vertices[0].z = center.z;

            plane.vertices[1].x = center.x - length.x / 2.0;
            plane.vertices[1].y = center.y + length.y / 2.0;
            plane.vertices[1].z = center.z;

            plane.vertices[2].x = center.x + length.x / 2.0;
            plane.vertices[2].y = center.y - length.y / 2.0;
            plane.vertices[2].z = center.z;

            plane.vertices[3].x = center.x + length.x / 2.0;
            plane.vertices[3].y = center.y + length.y / 2.0;
            plane.vertices[3].z = center.z;

            plane.numberOfTriangles = 2;
            plane.triangles = new int3[plane.numberOfTriangles];

            plane.triangles[0].x = 0;
            plane.triangles[0].y = 1;
            plane.triangles[0].z = 2;

            plane.triangles[1].x = 1;
            plane.triangles[1].y = 2;
            plane.triangles[1].z = 3;

            return plane;
        }

        private String[] NormalizeStrNums(String[] strNums)
        {
            String[] strRes = new String[strNums.Length];
            for (int i = 0; i < strNums.Length; i++)
            {
                strRes[i] = strNums[i].Replace(".", ",");
            }
            return strRes;
        }

        private String GetPath()
        {
            String pathTmpSurfaceFolder = (String)Application[TMP_SURFACE_FOLDER];
            if (!System.IO.Directory.Exists(pathTmpSurfaceFolder))
            {
                return null;
            }

            System.DateTime now = System.DateTime.Now;

            Random rand = new Random(now.Millisecond + now.Second + now.Minute + now.Hour);

            String fileName = now.Day.ToString() + "_" + now.Month.ToString()
                + "_" + now.Year.ToString() + "key=" + rand.Next().ToString() + ".surface";

            String pathSURFACEFile = pathTmpSurfaceFolder + "/" + fileName;

            if (System.IO.File.Exists(pathSURFACEFile))
            {
                System.IO.File.Delete(pathSURFACEFile);
            }

            return pathSURFACEFile;
        }

        [WebMethod]
        public String GetPathSURFACEFile(int numSurfaces,
            String[] centerX, String[] centerY, String[] centerZ, String[] lengthX, String[] lengthY)
        {
            Application[TMP_SURFACE_FOLDER] = GetValueFromPaths("TMP_SURFACE_FOLDER");

            centerX = NormalizeStrNums(centerX);
            centerY = NormalizeStrNums(centerY);
            centerZ = NormalizeStrNums(centerZ);
            lengthX = NormalizeStrNums(lengthX);
            lengthY = NormalizeStrNums(lengthY);

            Surface[] surface = new Surface[numSurfaces];

            for (int i = 0; i < numSurfaces; i++)
            {
                double3 center;

                try {Convert.ToDouble(centerX[i]);}
                catch {return null;}

                center.x = Convert.ToDouble(centerX[i]);

                try { Convert.ToDouble(centerY[i]); }
                catch { return null; }

                center.y = Convert.ToDouble(centerY[i]);

                try { Convert.ToDouble(centerZ[i]); }
                catch { return null; }

                center.z = Convert.ToDouble(centerZ[i]);

                double3 length;

                try { Convert.ToDouble(lengthX[i]); }
                catch { return null; }

                length.x = Convert.ToDouble(lengthX[i]);

                try { Convert.ToDouble(lengthY[i]); }
                catch { return null; }

                length.y = Convert.ToDouble(lengthY[i]);

                length.z = 0.0;

                surface[i] = GeneratePlane(center, length);
            }

            String pathSURFACEFile = GetPath();

            if (pathSURFACEFile == null)
            {
                return null;
            }

            WriteOutputToFile(surface, numSurfaces, pathSURFACEFile);

            return pathSURFACEFile;
        }
    }
}