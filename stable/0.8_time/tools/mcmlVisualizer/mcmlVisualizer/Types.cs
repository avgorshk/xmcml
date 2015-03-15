using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace mcmlVisualizer
{
    class Double3
    {
        public double x { get; private set; }
        public double y { get; private set; }
        public double z { get; private set; }
        
        public Double3(double x, double y, double z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    class Int3
    {
        public int x { get; private set; }
        public int y { get; private set; }
        public int z { get; private set; }

        public Int3(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    class Area
    {
        public Double3 corner { get; private set; }
        public Double3 length { get; private set; }
        public Int3 partitionNumber { get; private set; }

        public Area(Double3 corner, Double3 length, Int3 partitionNumber)
        {
            this.corner = corner;
            this.length = length;
            this.partitionNumber = partitionNumber;
        }
    }

    class Detector
    {
        public Double3 center { get; private set; }
        public Double3 length { get; private set; }
        public double weight { get; private set; }

        public Detector(Double3 center, Double3 lenght, double weight)
        {
            this.center = center;
            this.length = lenght;
            this.weight = weight;
        }
    }

    class TimeInfo
    {
        public double timeStart;
        public double timeFinish;
        public UInt64 numberOfPhotons;
        public double weight;
    }
}
