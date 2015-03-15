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

        public double[] GetSectionXY(double z)
        {
            int iz = (int)(area.partitionNumber.z * (z - area.corner.z) / area.length.z);
            bool isInArea = (iz >= 0) && (iz < area.partitionNumber.z);

            if (isInArea)
            {
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

            return null;   
        }

        public double[] GetSectionXZ(double y)
        {
            int iy = (int)(area.partitionNumber.y * (y - area.corner.y) / area.length.y);
            bool isInArea = (iy >= 0) && (iy < area.partitionNumber.y);

            if (isInArea)
            {
                double[] section = new double[area.partitionNumber.x * area.partitionNumber.z];
                for (int ix = 0; ix < area.partitionNumber.x; ++ix)
                {
                    for (int iz = 0; iz < area.partitionNumber.z; ++iz)
                    {
                        int index = ix * area.partitionNumber.y * area.partitionNumber.z +
                            iy * area.partitionNumber.z + iz;
                        section[ix * area.partitionNumber.z + iz] = trajectories[index];
                    }
                }

                return section;
            }

            return null;
        }

        public double[] GetSectionYZ(double x)
        {
            int ix = (int)(area.partitionNumber.x * (x - area.corner.x) / area.length.x);
            bool isInArea = (ix >= 0) && (ix < area.partitionNumber.x);

            if (isInArea)
            {
                double[] section = new double[area.partitionNumber.y * area.partitionNumber.z];
                for (int iy = 0; iy < area.partitionNumber.y; ++iy)
                {
                    for (int iz = 0; iz < area.partitionNumber.z; ++iz)
                    {
                        int index = ix * area.partitionNumber.y * area.partitionNumber.z +
                            iy * area.partitionNumber.z + iz;
                        section[iy * area.partitionNumber.z + iz] = trajectories[index];
                    }
                }

                return section;
            }

            return null;
        }
    }
}
