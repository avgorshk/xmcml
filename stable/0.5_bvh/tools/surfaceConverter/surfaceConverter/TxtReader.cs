﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Globalization;

namespace surfaceConverter
{
    class TxtReader
    {
        public Triangle[] triangle { get; private set; }

        public TxtReader(string fileName)
        {
            string[] lines = File.ReadAllLines(fileName);
            int numberOfTriangles = int.Parse(lines[0]);
            triangle = new Triangle[numberOfTriangles];

            int i = 1;
            int triangleIndex = 0;
            while (i < lines.Length)
            {
                if (lines[i].Length == 0)
                {
                    ++i;
                    continue;
                }

                Triangle tr = new Triangle();
                tr.a = GetVertex(lines[i]);
                ++i;
                tr.b = GetVertex(lines[i]);
                ++i;
                tr.c = GetVertex(lines[i]);

                triangle[triangleIndex] = tr;
                ++i;
                ++triangleIndex;
            }
        }

        private double3 GetVertex(string line)
        {
            NumberFormatInfo format = new NumberFormatInfo();
            format.NumberDecimalSeparator = ".";

            double3 vertex = new double3();

            int index = 0;
            while (index < line.Length)
            {
                if (line[index] == ' ' || line[index] == '\t')
                    ++index;
                else
                    break;
            }

            int k = index;
            while (char.IsDigit(line[index]) || line[index] == '.')
                ++index;

            vertex.x = double.Parse(line.Substring(k, index - k), format);

            while (index < line.Length)
            {
                if (line[index] == ' ' || line[index] == '\t')
                    ++index;
                else
                    break;
            }

            k = index;
            while (char.IsDigit(line[index]) || line[index] == '.')
                ++index;

            vertex.y = double.Parse(line.Substring(k, index - k), format);

            while (index < line.Length)
            {
                if (line[index] == ' ' || line[index] == '\t')
                    ++index;
                else
                    break;
            }

            k = index;
            while (char.IsDigit(line[index]) || line[index] == '.')
                ++index;

            vertex.z = double.Parse(line.Substring(k, index - k), format);

            return vertex;
        }
    }
}
