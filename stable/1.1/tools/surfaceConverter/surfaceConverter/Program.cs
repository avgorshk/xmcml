using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace surfaceConverter
{
    class Program
    {
        static void main_1(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Too few arguments\n");
                return;
            }

            Surface[] surface = new Surface[2];

            //SURAFCE 1 =======================================================

            TxtReader txtReader = new TxtReader(args[0]);

            surface[0] = new Surface(txtReader.triangle);
            surface[0].Scale(512.0, 512.0, 512.0);

            double max, min;

            max = surface[0].vertices[0].z;
            min = surface[0].vertices[0].z;
            for (int i = 0; i < surface[0].vertices.Length; ++i)
            {
                if (surface[0].vertices[i].z > max)
                    max = surface[0].vertices[i].z;
                if (surface[0].vertices[i].z < min)
                    min = surface[0].vertices[i].z;
            }

            double translate_z = -min;
            double cond_z = max / 5.0;

            max = surface[0].vertices[0].x;
            min = surface[0].vertices[0].x;
            for (int i = 0; i < surface[0].vertices.Length; ++i)
            {
                if (surface[0].vertices[i].x > max)
                    max = surface[0].vertices[i].x;
                if (surface[0].vertices[i].x < min)
                    min = surface[0].vertices[i].x;
            }

            double translate_x = -(max + min) / 2.0;
            double int_x = (max - min) / 2.0;

            max = surface[0].vertices[0].y;
            min = surface[0].vertices[0].y;
            for (int i = 0; i < surface[0].vertices.Length; ++i)
            {
                if (surface[0].vertices[i].y > max)
                    max = surface[0].vertices[i].y;
                if (surface[0].vertices[i].y < min)
                    min = surface[0].vertices[i].y;
            }

            double translate_y = -(max + min) / 2.0;
            double int_y = (max - min) / 2.0;

            surface[0].Translate(translate_x, translate_y, translate_z);

            double cond = 0.2 * Math.Sqrt(int_x * int_x + int_y * int_y);
            List<int3> triangles = new List<int3>();
            for (int i = 0; i < surface[0].triangles.Length; ++i)
            {
                double3 v1 = surface[0].vertices[surface[0].triangles[i].x];
                double3 v2 = surface[0].vertices[surface[0].triangles[i].y];
                double3 v3 = surface[0].vertices[surface[0].triangles[i].z];

                bool is_del = false;
                is_del = is_del || (Math.Sqrt(v1.x * v1.x + v1.y * v1.y) < cond);
                is_del = is_del || (Math.Sqrt(v2.x * v2.x + v2.y * v2.y) < cond);
                is_del = is_del || (Math.Sqrt(v3.x * v3.x + v3.y * v3.y) < cond);

                if (v1.z < cond_z || v2.z < cond_z || v3.z < cond_z)
                    is_del = false;

                if (!is_del)
                    triangles.Add(surface[0].triangles[i]);
            }

            surface[0].triangles = triangles.ToArray();

            double3 origin;
            origin.x = 0;
            origin.y = 0;
            origin.z = 0;
            double3 direction;
            direction.x = 0;
            direction.y = 0;
            direction.z = 1;
            double translate_dist = surface[0].ComputeIntersection(origin, direction);
            surface[0].Translate(0, 0, -translate_dist);

            double distance = surface[0].ComputeIntersection(origin, direction);
            Console.WriteLine();
            Console.WriteLine("Distance = " + distance.ToString());

            //SURAFCE 2 =======================================================
            surface[1] = new Surface(surface[0]);

            double s2_translate_z, s2_max_z = 0;
            for (int i = 0; i < surface[1].vertices.Length; ++i)
            {
                if (surface[1].vertices[i].z > s2_max_z)
                    s2_max_z = surface[1].vertices[i].z;
            }
            s2_translate_z = s2_max_z / 2.0;

            surface[1].Translate(0, 0, -s2_translate_z);

            double s2_coeff = 0.97054;
            surface[1].Scale(s2_coeff, s2_coeff, s2_coeff);

            surface[1].Translate(0, 0, s2_translate_z);

            distance = surface[1].ComputeIntersection(origin, direction);
            Console.WriteLine();
            Console.WriteLine("Distance = " + distance.ToString());


            //SURAFCE 2, 3 ====================================================

            //txtReader = new TxtReader(args[1]);
            //surface[1] = new Surface(txtReader.triangle);
            //surface[1].Scale(512.0, 512.0, 512.0);
            //surface[1].Translate(translate_x, translate_y, translate_z);

            //SURFACE 4 ========================================================

            //txtReader = new TxtReader(args[2]);
            //surface[2] = new Surface(txtReader.triangle);
            //surface[2].Scale(512.0, 512.0, 512.0);
            //surface[2].Translate(translate_x, translate_y, translate_z);

            //SurfaceWriter writer = new SurfaceWriter(surface);
            //writer.Write(args[args.Length - 1]);

            StreamWriter writer = new StreamWriter("params.txt");
            writer.WriteLine(translate_x);
            writer.WriteLine(translate_y);
            writer.WriteLine(translate_z - translate_dist);
            writer.Close();
        }

        static bool isHasEdge(int3 t1, int3 t2)
        {
            bool cond1 = ((t1.x == t2.x) || (t1.y == t2.x) || (t1.z == t2.x));
            bool cond2 = ((t1.x == t2.y) || (t1.y == t2.y) || (t1.z == t2.y));
            bool cond3 = ((t1.x == t2.z) || (t1.y == t2.z) || (t1.z == t2.z));
            return (cond1 && cond2) || (cond1 && cond3) || (cond2 && cond3);
        }

        static Surface GetNewSurface(Surface surface)
        {
            List<int3> start = new List<int3>(surface.triangles);
            List<int3> finish = new List<int3>();
            Stack<int> stack = new Stack<int>();

            stack.Push(0);
            finish.Add(start[0]);
            start.RemoveAt(0);

            while (stack.Count != 0)
            {
                int3 current = finish[stack.Pop()];

                for (int i = 0; i < start.Count; ++i)
                {
                    if (isHasEdge(start[i], current))
                    {
                        finish.Add(start[i]);
                        stack.Push(finish.Count - 1);
                        start.RemoveAt(i);
                        --i;
                    }
                }

                if (finish.Count % 1000 == 0)
                    Console.Write(".");
            }

            Surface result = new Surface(surface.vertices, finish.ToArray());
            return result;
        }

        static void main_2(string[] args)
        {
            Surface[] surface = new Surface[1];
            TxtReader reader = new TxtReader("bone.txt");
            surface[0] = new Surface(reader.triangle);
            surface[0].Scale(512.0, 512.0, 512.0);

            StreamReader sr = new StreamReader("params.txt");
            double translate_x = double.Parse(sr.ReadLine());
            double translate_y = double.Parse(sr.ReadLine());
            double translate_z = double.Parse(sr.ReadLine());

            surface[0].Translate(translate_x, translate_y, translate_z);

            SurfaceWriter writer = new SurfaceWriter(surface);
            writer.Write("bone_stage_1.surface");
        }

        static void main_3(string[] args)
        {
            SurfaceReader r1 = new SurfaceReader("bone_stage_2.surface");
            Surface result = GetNewSurface(r1.surface[0]);

            SurfaceWriter writer = new SurfaceWriter(new Surface[] {result});
            writer.Write("bone_stage_3.surface");
        }

        static void main_4(string[] args)
        {
            TxtReader reader = new TxtReader("bone_bk.txt");
            Triangle[] new_triangle = new Triangle[reader.triangle.Length / 2];
            for (int i = 0; i < reader.triangle.Length / 2; ++i)
            {
                new_triangle[i] = reader.triangle[2 * i];
            }

            Surface result = new Surface(new_triangle);

            result.Scale(512.0, 512.0, 512.0);

            StreamReader sr = new StreamReader("params.txt");
            double translate_x = double.Parse(sr.ReadLine());
            double translate_y = double.Parse(sr.ReadLine());
            double translate_z = double.Parse(sr.ReadLine());

            result.Translate(translate_x, translate_y, translate_z);

            SurfaceWriter writer = new SurfaceWriter(new Surface[] {result});
            writer.Write("bone_new.surface");
        }

        static void main_5(string[] args)
        {
            SurfaceReader reader1 = new SurfaceReader("brain_12.surface");
            SurfaceReader reader2 = new SurfaceReader("bone_stage_4.surface");

            Surface[] result = new Surface[4];
            result[0] = reader1.surface[0];
            result[1] = reader1.surface[1];
            result[2] = reader2.surface[0];

            double s2_translate_z, s2_max_z = 0;
            for (int i = 0; i < result[2].vertices.Length; ++i)
            {
                if (result[2].vertices[i].z > s2_max_z)
                    s2_max_z = result[2].vertices[i].z;
            }
            s2_translate_z = s2_max_z / 2.0;

            result[2].Translate(0, 0, -s2_translate_z);

            double s2_coeff = 0.95;
            result[2].Scale(s2_coeff, s2_coeff, s2_coeff);

            result[2].Translate(0, 0, s2_translate_z);

            double3 origin;
            origin.x = 0;
            origin.y = 0;
            origin.z = 0;
            double3 direction;
            direction.x = 0;
            direction.y = 0;
            direction.z = 1;
            double dist = result[2].ComputeIntersection(origin, direction);
            Console.WriteLine(dist);
            dist = 5.3 - dist;
            result[2].Translate(0, 0, dist);

            dist = result[2].ComputeIntersection(origin, direction);
            Console.WriteLine("Finish " + dist.ToString());

            result[3] = new Surface(result[2]);

            double s3_translate_z, s3_max_z = 0;
            for (int i = 0; i < result[3].vertices.Length; ++i)
            {
                if (result[3].vertices[i].z > s3_max_z)
                    s3_max_z = result[3].vertices[i].z;
            }
            s3_translate_z = s3_max_z / 2.0;

            result[3].Translate(0, 0, -s3_translate_z);

            double s3_coeff = 0.9;
            result[3].Scale(s3_coeff, s3_coeff, s3_coeff);

            result[3].Translate(0, 0, s3_translate_z);

            origin.x = 0;
            origin.y = 0;
            origin.z = 0;
            direction.x = 0;
            direction.y = 0;
            direction.z = 1;
            dist = result[3].ComputeIntersection(origin, direction);
            Console.WriteLine(dist);
            dist = 12.2 - dist;
            result[3].Translate(0, 0, dist);

            dist = result[3].ComputeIntersection(origin, direction);
            Console.WriteLine("Finish " + dist.ToString());

            SurfaceWriter writer = new SurfaceWriter(result);
            writer.Write("brain_1234.surface");
        }

        static void main_6(string[] args)
        {
            TxtReader reader = new TxtReader("brain_level2.txt");
            Surface result = new Surface(reader.triangle);
            result.Scale(512, 512, 512);

            double min, max;
            min = result.vertices[0].x;
            max = result.vertices[0].x;
            for (int i = 1; i < result.vertices.Length; ++i)
            {
                if (min > result.vertices[i].x) min = result.vertices[i].x;
                if (max < result.vertices[i].x) max = result.vertices[i].x;
            }
            double translate_x = -(max + min) / 2.0;

            min = result.vertices[0].y;
            max = result.vertices[0].y;
            for (int i = 1; i < result.vertices.Length; ++i)
            {
                if (min > result.vertices[i].y) min = result.vertices[i].y;
                if (max < result.vertices[i].y) max = result.vertices[i].y;
            }
            double translate_y = -(max + min) / 2.0;

            min = result.vertices[0].z;
            max = result.vertices[0].z;
            for (int i = 1; i < result.vertices.Length; ++i)
            {
                if (min > result.vertices[i].z) min = result.vertices[i].z;
                if (max < result.vertices[i].z) max = result.vertices[i].z;
            }
            double translate_z = -(max + min) / 2.0;

            result.Translate(translate_x, translate_y, translate_z);

            SurfaceWriter writer = new SurfaceWriter(new Surface[] { result });
            writer.Write("matter_stage_1.surface");
        }

        static void main_7(string[] args)
        {
            SurfaceReader reader1 = new SurfaceReader("brain_1234.surface");
            SurfaceReader reader2 = new SurfaceReader("matter_stage_2.surface");

            Surface[] surface = new Surface[6];
            surface[0] = reader1.surface[0];
            surface[1] = reader1.surface[1];
            surface[2] = reader1.surface[2];
            surface[3] = reader1.surface[3];
            surface[4] = reader2.surface[0];

            surface[4].RotateZ(0.5 * Math.PI);

            surface[5] = new Surface(surface[4]);

            double coeff = 0.52;
            surface[4].Scale(coeff, coeff, coeff);
            surface[4].Translate(3.5, 7.5, 0);

            surface[4].Translate(0, 0, 52.0485);

            double3 origin;
            origin.x = 0;
            origin.y = 0;
            origin.z = 0;
            double3 direction;
            direction.x = 0;
            direction.y = 0;
            direction.z = 1;
            double dist = surface[4].ComputeIntersection(origin, direction);
            Console.WriteLine(dist);

            coeff = 0.45117;
            surface[5].Scale(coeff, coeff, coeff);
            surface[5].Translate(3.5, 7.5, 0);

            surface[5].Translate(0, 0, 52.0485);

            origin.x = 0;
            origin.y = 0;
            origin.z = 0;
            direction.x = 0;
            direction.y = 0;
            direction.z = 1;
            dist = surface[5].ComputeIntersection(origin, direction);
            Console.WriteLine(dist);

            for (int i = 0; i < surface.Length; ++i)
            {
                origin.x = 0;
                origin.y = 0;
                origin.z = 0;
                direction.x = 0;
                direction.y = 0;
                direction.z = 1;
                dist = surface[i].ComputeIntersection(origin, direction);
                Console.WriteLine(dist);
            }

            SurfaceWriter writer = new SurfaceWriter(surface);
            writer.Write("human_head_stage_1.surface");
        }

        static void main_8(string[] args)
        {
            SurfaceReader reader = new SurfaceReader("human_head_stage_1.surface");
            
            double min = 0;
            for (int i = 0; i < reader.surface[0].vertices.Length; ++i)
            {
                if (min > reader.surface[0].vertices[i].z)
                    min = reader.surface[0].vertices[i].z;
            }
            Console.WriteLine(min);

            for (int i = 0; i < reader.surface.Length; ++i)
            {
                reader.surface[i].Translate(0, 0, -min);
            }

            SurfaceWriter writer = new SurfaceWriter(reader.surface);
            writer.Write("human_head.surface");
        }

        static void main_9(string[] args)
        {
            SurfaceReader reader = new SurfaceReader("human_head_180.surface");
            Surface surface = reader.surface[0];
            int numberOfDetectors = 401;
            double3[] detector_pos = new double3[numberOfDetectors];
            double EPS = 0.0000001;
            double detector_dist = 0.1;

            for (int i = 0; i < numberOfDetectors; ++i)
            {
                double step = 0.1 * detector_dist;
                double dist = 0;
                double3 origin;
                origin.x = (i == 0 ? 0 : detector_pos[i - 1].x) + 0.5 * detector_dist;
                origin.y = 0;
                origin.z = 0;
                double3 direction;
                direction.x = 0;
                direction.y = 0;
                direction.z = 1;

                double3 pos;
                pos.x = 0;
                pos.y = 0;
                pos.z = 0;

                double3 prev_pos;
                if (i == 0)
                {
                    prev_pos.x = 0;
                    prev_pos.y = 0;
                    prev_pos.z = 0.587662;
                }
                else
                {
                    prev_pos = detector_pos[i - 1];
                }

                while (Math.Abs(dist - detector_dist) > EPS)
                {
                    do
                    {
                        origin.x += step;
                        double distZ = surface.ComputeIntersection(origin, direction);
                        pos.x = origin.x + distZ * direction.x;
                        pos.y = origin.y + distZ * direction.y;
                        pos.z = origin.z + distZ * direction.z;
                        dist = Math.Sqrt((pos.x - prev_pos.x) * (pos.x - prev_pos.x) +
                            (pos.y - prev_pos.y) * (pos.y - prev_pos.y) +
                            (pos.z - prev_pos.z) * (pos.z - prev_pos.z));

                    } while (dist < detector_dist);

                    origin.x -= step;
                    step *= 0.1;
                }

                detector_pos[i] = pos;
                Console.Write('.');
            }
            Console.WriteLine();

            StreamWriter writer = new StreamWriter("res.xml");
            int n = (int)(1 / detector_dist);
            for (int i = n - 1; i < numberOfDetectors; i += n)
            {
                double left = (i < n ? 0 : detector_pos[i - n].x);
                double rigth = detector_pos[i].x;
                double center = (left + rigth) / 2.0;
                double length = rigth - left;
                //double3 a1;
                //if (i < n)
                //{
                //    a1.x = 0;
                //    a1.y = 0;
                //    a1.z = 0.587662;
                //}
                //else
                //{
                //    a1 = detector_pos[i - n];
                //}
                //double3 a2 = detector_pos[i - 1];
                //double3 a3 = detector_pos[i];
                //double dist1 = Math.Sqrt((a1.x - a2.x) * (a1.x - a2.x) + (a1.z - a2.z) * (a1.z - a2.z));
                //double dist2 = Math.Sqrt((a3.x - a2.x) * (a3.x - a2.x) + (a3.z - a2.z) * (a3.z - a2.z));
                //double square = dist1 + dist2;

                var format = new System.Globalization.NumberFormatInfo();
                format.NumberDecimalSeparator = ".";
                Console.WriteLine("c = {0} | l = {1}", center.ToString(format), length.ToString(format));

                writer.WriteLine("\t\t<Detector>");
                writer.WriteLine("\t\t\t<Center>");
                writer.WriteLine(String.Format("\t\t\t\t<X>{0}</X>", center.ToString(format)));
                writer.WriteLine(String.Format("\t\t\t\t<Y>{0}</Y>", 0));
                writer.WriteLine(String.Format("\t\t\t\t<Z>{0}</Z>", 0));
                writer.WriteLine("\t\t\t</Center>");
                writer.WriteLine("\t\t\t<Length>");
                writer.WriteLine(String.Format("\t\t\t\t<X>{0}</X>", length.ToString(format)));
                writer.WriteLine(String.Format("\t\t\t\t<Y>{0}</Y>", 1));
                writer.WriteLine(String.Format("\t\t\t\t<Z>{0}</Z>", 100));
                writer.WriteLine("\t\t\t</Length>");
                writer.WriteLine("\t\t</Detector>");
            }
            writer.Close();

            //double total_distance = 0.0;
            //for (int i = 1; i < numberOfDetectors; ++i)
            //{
            //    double3 a = detector_pos[i - 1];
            //    double3 b = detector_pos[i];
            //    double distance = Math.Sqrt((a.x - b.x) * (a.x - b.x) +
            //        (a.y - b.y) * (a.y - b.y) +
            //        (a.z - b.z) * (a.z - b.z));
            //    total_distance += distance;
            //    if (i % 2 == 1) Console.WriteLine(total_distance);
            //}

            //Console.WriteLine("Distance between source and the last detector:");
            //double3 c;
            //c.x = 0;
            //c.y = 0;
            //c.z = 0.587662;
            //double3 d = detector_pos[numberOfDetectors - 1];
            //double source_dist = Math.Sqrt((c.x - d.x) * (c.x - d.x) +
            //        (c.y - d.y) * (c.y - d.y) +
            //        (c.z - d.z) * (c.z - d.z));
            //Console.WriteLine(source_dist);
        }

        static void main_10(string[] args)
        {
            SurfaceReader reader = new SurfaceReader("human_head.surface");
            for (int i = 0; i < reader.surface.Length; ++i)
            {
                reader.surface[i].RotateZ(Math.PI);
            }

            SurfaceWriter writer = new SurfaceWriter(reader.surface);
            writer.Write("human_head_180.surface");
        }

        static void Main(string[] args)
        {
            main_9(args);
        }
    }
}
