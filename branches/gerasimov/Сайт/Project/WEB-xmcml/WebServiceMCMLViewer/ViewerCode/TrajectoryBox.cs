using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace mcmlVisualizer
{
    class TrajectoryBox
    {
        public Area area { get; private set; }
        public double[] trajectories { get; private set; }

        public TrajectoryBox(Area area, double[] trajectories)
        {
            this.area = area;
            this.trajectories = trajectories;
        }

        public TrajectoryBox(Area area, UInt64[] trajectories)
        {
            this.area = area;
            this.trajectories = new double[trajectories.Length];
            for (int i = 0; i < trajectories.Length; ++i)
            {
                this.trajectories[i] = (double)(trajectories[i]);
            }
        }

        public double[] GetSectionXY(double z)
        {
            int iz = (int)(area.partitionNumber.z * (z - area.corner.z) / area.length.z);
            if (iz < 0) { iz = 0; }
            else if (iz >= area.partitionNumber.z) { iz = area.partitionNumber.z - 1; }

            if ((iz < 0) || (iz >= area.partitionNumber.z))
                { return null; }

            double[] section = new double[area.partitionNumber.x * area.partitionNumber.y];
            for (int ix = 0; ix < area.partitionNumber.x; ++ix)
            {
                for (int iy = 0; iy < area.partitionNumber.y; ++iy)
                {
                    int index = ix * area.partitionNumber.y * area.partitionNumber.z +
                        iy * area.partitionNumber.z + iz;
                    section[ix * area.partitionNumber.y + iy] = trajectories[index];
                }
            }
            return section;
        }

        public double[] GetSectionXZ(double y)
        {
            int iy = (int)(area.partitionNumber.y * (y - area.corner.y) / area.length.y);
            if (iy < 0) { iy = 0; }
            else if (iy >= area.partitionNumber.y) { iy = area.partitionNumber.y - 1; }

            if ((iy < 0) || (iy >= area.partitionNumber.y))
                { return null; }

            double[] section = new double[area.partitionNumber.x * area.partitionNumber.z];
            for (int iz = 0; iz < area.partitionNumber.z; ++iz)
            {
                for (int ix = 0; ix < area.partitionNumber.x; ++ix)
                {
                    int index = ix * area.partitionNumber.y * area.partitionNumber.z +
                        iy * area.partitionNumber.z + iz;
                    section[iz * area.partitionNumber.x + ix] = trajectories[index];
                }
            }
            return section;
        }

        public double[] GetSectionYZ(double x)
        {
            int ix = (int)(area.partitionNumber.x * (x - area.corner.x) / area.length.x);
            if (ix < 0) { ix = 0; }
            else if (ix >= area.partitionNumber.x) { ix = area.partitionNumber.x - 1; }

            if ((ix < 0) || (ix >= area.partitionNumber.x))
                { return null; }

            double[] section = new double[area.partitionNumber.y * area.partitionNumber.z];
            for (int iz = 0; iz < area.partitionNumber.z; ++iz)
            {
                for (int iy = 0; iy < area.partitionNumber.y; ++iy)
                {
                    int index = ix * area.partitionNumber.y * area.partitionNumber.z +
                        iy * area.partitionNumber.z + iz;
                    section[iz * area.partitionNumber.y + iy] = trajectories[index];
                }
            }
            return section;
        }
    }
}
