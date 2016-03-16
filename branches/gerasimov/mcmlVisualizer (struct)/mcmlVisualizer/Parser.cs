using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace mcmlVisualizer
{
    class Parser
    {
        private const uint MCML_SECTION_NUMBER_OF_PHOTONS = 1;
        private const uint MCML_SECTION_AREA = 2;
        private const uint MCML_SECTION_CUBE_DETECTORS = 3;
        private const uint MCML_SECTION_SPECULAR_REFLECTANCE = 4;
        private const uint MCML_SECTION_COMMON_TRAJECTORIES = 5;
        private const uint MCML_SECTION_SCATTERING_MAP = 6;
        private const uint MCML_SECTION_DEPTH_MAP = 7;
        private const uint MCML_SECTION_DETECTOR_WEIGHTS = 8;
        private const uint MCML_SECTION_GRID_DETECTOR = 9;
        private const uint MCML_SECTION_DETECTOR_TRAJECTORIES = 10;
        private const uint MCML_SECTION_DETECTOR_TIME_SCALE = 11;
        private const uint MCML_SECTION_RING_DETECTORS = 13;
        private const uint MCML_SECTION_DETECTOR_RANGES = 14;

        private FileStream file;
        private Hashtable sections;

        private UInt64 numberOfPhotons;
        private Area area;
        private double specularReflecrance;

        private double[] detectorWeights = new double[0];
        private double[] gridDetectorWeights = new double[0];
        private double[] detectorTargetRanges = new double[0];
        private UInt64[] numPhotonsInDetector = new UInt64[0];

        private double[] absorptionMap = new double[0];
        private double[] scatteringMap = new double[0];
        private double[] depthMap = new double[0];

        static public Parser getInstance(string fileName)
        {
            FileStream file = null;
            try
            {
                file = File.Open(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            }
            catch (Exception)
            {
                return null;
            }

            Hashtable sections = new Hashtable();
            bool isOk = GetSections(file, sections);
            if (isOk && CheckSections(sections))
            {
                return new Parser(file, sections);
            }

            return null;
        }

        static private bool GetSections(FileStream file, Hashtable sections)
        {
            uint section, lenght, offset;
            BinaryReader reader = new BinaryReader(file);

            try
            {
                offset = 0;
                while (true)
                {
                    section = reader.ReadUInt32();
                    lenght = reader.ReadUInt32();
                    offset += 8;
                    sections[section] = offset;
                    offset += lenght;
                    reader.BaseStream.Seek(lenght, SeekOrigin.Current);
                }
            }
            catch (EndOfStreamException)
            {
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        static private bool CheckSections(Hashtable sections)
        {
            return
                sections.ContainsKey(MCML_SECTION_NUMBER_OF_PHOTONS) &&
                sections.ContainsKey(MCML_SECTION_AREA) &&
                sections.ContainsKey(MCML_SECTION_SPECULAR_REFLECTANCE);
        }

        private Parser(FileStream file, Hashtable sections)
        {
            this.file = file;
            this.sections = sections;
            Init();
        }

        ~Parser()
        {
            file.Close();
            sections.Clear();
        }

        private void Init()
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = 0;

            // MCML_SECTION_NUMBER_OF_PHOTONS
            offset = (uint)this.sections[MCML_SECTION_NUMBER_OF_PHOTONS];
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            numberOfPhotons = reader.ReadUInt64();

            // MCML_SECTION_AREA
            offset = (uint)this.sections[MCML_SECTION_AREA];
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            Double3 corner = new Double3(reader.ReadDouble(), reader.ReadDouble(), reader.ReadDouble());
            Double3 length = new Double3(reader.ReadDouble(), reader.ReadDouble(), reader.ReadDouble());
            Int3 partitionNumber = new Int3(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
            area = new Area(corner, length, partitionNumber);

            // MCML_SECTION_SPECULAR_REFLECTANCE
            offset = (uint)this.sections[MCML_SECTION_SPECULAR_REFLECTANCE];
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            specularReflecrance = reader.ReadDouble();

            // MCML_SECTION_COMMON_TRAJECTORIES
            if (sections.ContainsKey(MCML_SECTION_COMMON_TRAJECTORIES))
            {
                offset = (uint)this.sections[MCML_SECTION_COMMON_TRAJECTORIES];
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);
                int size = reader.ReadInt32();
                absorptionMap = new double[size];
                for (int i = 0; i < size; ++i)
                {
                    absorptionMap[i] = reader.ReadDouble();
                }
            }

            // MCML_SECTION_SCATTERING_MAP
            if (sections.ContainsKey(MCML_SECTION_SCATTERING_MAP))
            {
                offset = (uint)this.sections[MCML_SECTION_SCATTERING_MAP];
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);
                int size = reader.ReadInt32();
                scatteringMap = new double[size];
                for (int i = 0; i < size; ++i)
                {
                    scatteringMap[i] = reader.ReadDouble();
                }
            }

            // MCML_SECTION_DEPTH_MAP
            if (sections.ContainsKey(MCML_SECTION_DEPTH_MAP))
            {
                offset = (uint)this.sections[MCML_SECTION_DEPTH_MAP];
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);
                int size = reader.ReadInt32();
                depthMap = new double[size];
                for (int i = 0; i < size; ++i)
                {
                    depthMap[i] = reader.ReadDouble();
                }
            }

            // MCML_SECTION_DETECTOR_WEIGHTS
            if (sections.ContainsKey(MCML_SECTION_DETECTOR_WEIGHTS))
            {
                offset = (uint)this.sections[MCML_SECTION_DETECTOR_WEIGHTS];
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);
                int numberOfDetectors = reader.ReadInt32();
                detectorWeights = new double[numberOfDetectors];
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    detectorWeights[i] = reader.ReadDouble();
                }
            }

            // MCML_SECTION_GRID_DETECTOR
            if (sections.ContainsKey(MCML_SECTION_GRID_DETECTOR))
            {
                offset = (uint)this.sections[MCML_SECTION_GRID_DETECTOR];
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);
                int gridDetectorSize = reader.ReadInt32();
                gridDetectorWeights = new double[gridDetectorSize];
                for (int i = 0; i < gridDetectorSize; ++i)
                {
                    gridDetectorWeights[i] = reader.ReadDouble();
                }
            }

            // MCML_SECTION_DETECTOR_TRAJECTORIES
            if (sections.ContainsKey(MCML_SECTION_DETECTOR_TRAJECTORIES))
            {
                offset = (uint)this.sections[MCML_SECTION_DETECTOR_TRAJECTORIES];
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);
                int numberOfDetectors = reader.ReadInt32();
                numPhotonsInDetector = new UInt64[numberOfDetectors];
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    numPhotonsInDetector[i] = reader.ReadUInt64();
                    int size = reader.ReadInt32();
                    offset = (uint)(size * sizeof(double));
                    reader.BaseStream.Seek(offset, SeekOrigin.Current);
                }
            }

            // MCML_SECTION_DETECTOR_RANGES
            if (sections.ContainsKey(MCML_SECTION_DETECTOR_RANGES))
            {
                offset = (uint)this.sections[MCML_SECTION_DETECTOR_RANGES];
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);
                int numberOfDetectors = reader.ReadInt32();
                detectorTargetRanges = new double[numberOfDetectors];
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    reader.ReadDouble();
                    detectorTargetRanges[i] = reader.ReadDouble();
                }
            }
        }

        public UInt64 GetNumberOfPhotons() 
        {
            return numberOfPhotons;
        }

        public Area GetArea()
        {
            return area;
        }
        
        public double GetSpecularReflectance()
        {
            return specularReflecrance;
        }

        public int GetNumberOfDetectors()
        {
            return detectorWeights.Length;
        }

        public double[] GetDetectorWeights()
        {
            return detectorWeights;
        }

        public double[] GetGridDetectorWeights()
        {
            return gridDetectorWeights;
        }

        public double[] GetDetectorTargetRanges()
        {
            return detectorTargetRanges;
        }

        public double[] GetTrajectoriesApsorption()
        {
            return absorptionMap;
        }

        public double[] GetTrajectoriesScattering()
        {
            return scatteringMap;
        }

        public double[] GetTrajectoriesDepth()
        {
            return depthMap;
        }

        public UInt64[] GetDetectorTrajectories(int detectorId)
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = (uint)(this.sections[(uint?)MCML_SECTION_DETECTOR_TRAJECTORIES]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);

            int numberOfDetectors = reader.ReadInt32();
            for (int i = 0; i < numberOfDetectors; ++i)
            {
                ulong numberOfPhotons = reader.ReadUInt64();
                int numberOfValues = reader.ReadInt32();
                if (i == detectorId)
                {
                    UInt64[] trajectory = new UInt64[numberOfValues];
                    for (int j = 0; j < numberOfValues; ++j)
                    {
                        trajectory[j] = reader.ReadUInt64();
                    }
                    return trajectory;
                }
                else
                {
                    offset = (uint)(numberOfValues * sizeof(double));
                    reader.BaseStream.Seek(offset, SeekOrigin.Current);
                }
            }

            return new UInt64[0];
        }

        public TimeInfo[] GetDetectorTimeScale(int detectorId)
        {
            BinaryReader reader = new BinaryReader(this.file);
            if (this.sections[(uint?)MCML_SECTION_DETECTOR_TIME_SCALE] != null)
            {
                uint offset = (uint)(this.sections[(uint?)MCML_SECTION_DETECTOR_TIME_SCALE]);
                reader.BaseStream.Seek(offset, SeekOrigin.Begin);

                int numberOfDetectors = reader.ReadInt32();
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    int timeScaleSize = reader.ReadInt32();
                    if (i == detectorId)
                    {
                        TimeInfo[] timeInfo = new TimeInfo[timeScaleSize];
                        for (int j = 0; j < timeScaleSize; ++j)
                        {
                            timeInfo[j] = new TimeInfo();
                            timeInfo[j].timeStart = reader.ReadDouble();
                            timeInfo[j].timeFinish = reader.ReadDouble();
                            timeInfo[j].numberOfPhotons = reader.ReadUInt64();
                            timeInfo[j].weight = reader.ReadDouble();
                        }
                        return timeInfo;
                    }
                    else
                    {
                        offset = (uint)(timeScaleSize * (3 * sizeof(double) + sizeof(UInt64)));
                        reader.BaseStream.Seek(offset, SeekOrigin.Current);
                    }
                }
            }

            return null;
        }

        public UInt64 GetNumberOfPhotonsInDetector(int detectorId)
        {
            if (detectorId < numPhotonsInDetector.Length)
            {
                return numPhotonsInDetector[detectorId];
            }

            return 0;
        }

        public UInt64[] GetNumberOfPhotonsInDetectorAsArray()
        {
            return numPhotonsInDetector;
        }

        public StringBuilder GetText()
        {
            StringBuilder text = new StringBuilder();

            text.AppendFormat("Number of photons: {0}\n", numberOfPhotons);
            text.AppendFormat("Specular reflectance: {0}\n", specularReflecrance);

            text.AppendFormat("Photons in detector ({0}):\n", numPhotonsInDetector.Length);
            for (int i = 0; i < numPhotonsInDetector.Length; ++i)
            {
                text.AppendFormat("{0} ", numPhotonsInDetector[i]);
            }
            text.AppendFormat("\n");

            text.AppendFormat("Weights ({0}):\n", detectorWeights.Length);
            for (int i = 0; i < detectorWeights.Length; ++i)
            {
                text.AppendFormat("{0} ", detectorWeights[i]);
            }
            text.AppendFormat("\n");

            text.AppendFormat("Weights in grid detectors ({0}):\n", gridDetectorWeights.Length);
            for (int i = 0; i < gridDetectorWeights.Length; ++i)
            {
                text.AppendFormat("{0} ", gridDetectorWeights[i]);
            }
            text.AppendFormat("\n");

            text.AppendFormat("Target ranges ({0}):\n", detectorTargetRanges.Length);
            for (int i = 0; i < detectorTargetRanges.Length; ++i)
            {
                text.AppendFormat("{0} ", detectorTargetRanges[i]);
            }
            text.AppendFormat("\n");

            text.AppendFormat("Absorption Map ({0}):\n", absorptionMap.Length);
            for (int i = 0; i < absorptionMap.Length; ++i)
            {
                text.AppendFormat("{0} ", absorptionMap[i]);
            }
            text.AppendFormat("\n");

            text.AppendFormat("Scattering Map ({0}):\n", scatteringMap.Length);
            for (int i = 0; i < scatteringMap.Length; ++i)
            {
                text.AppendFormat("{0} ", scatteringMap[i]);
            }
            text.AppendFormat("\n");

            text.AppendFormat("Depth Map ({0}):\n", depthMap.Length);
            for (int i = 0; i < depthMap.Length; ++i)
            {
                text.AppendFormat("{0} ", depthMap[i]);
            }
            text.AppendFormat("\n");

            ulong[] commonTrajectory = GetDetectorTrajectories(0);
            text.AppendFormat("Detector trajectories ({0})\n", detectorWeights.Length);
            for (int i = 1; i < detectorWeights.Length; ++i)
            {
                ulong[] trajectory = GetDetectorTrajectories(i);
                for (int j = 0; j < commonTrajectory.Length; ++j)
                {
                    commonTrajectory[j] += trajectory[j];
                }
                
            }
            text.AppendFormat("Common trajectory ({0}):\n", commonTrajectory.Length);
            for (int i = 0; i < commonTrajectory.Length; ++i)
            {
                text.AppendFormat("{0} ", commonTrajectory[i]);
            }
            text.AppendFormat("\n");

            text.AppendFormat("Detector Time Scales ({0}):\n", detectorWeights.Length);
            for (int i = 0; i < detectorWeights.Length; ++i)
            {
                TimeInfo[] timeInfo = GetDetectorTimeScale(i);
                text.AppendFormat("Detector {0} ({1}):\n", i, timeInfo.Length);
                for (int j = 0; j < timeInfo.Length; ++j)
                {
                    text.AppendFormat("[{0}, {1}] -> ({2}, {3})\n", timeInfo[j].timeStart, timeInfo[j].timeFinish,
                        timeInfo[j].numberOfPhotons, timeInfo[j].weight);
                }
                text.AppendFormat("\n");
            }

            return text;
        }
    }
}
