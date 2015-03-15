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
        private const int MCML_SECTION_NUMBER_OF_PHOTONS = 1;
        private const int MCML_SECTION_AREA = 2;
        private const int MCML_SECTION_DETECTORS = 3;
        private const int MCML_SECTION_SPECULAR_REFLECTANCE = 4;
        private const int MCML_SECTION_COMMON_TRAJECTORIES = 5;
        private const int MCML_SECTION_DETECTOR_WEIGHTS = 6;
        private const int MCML_SECTION_DETECTOR_TRAJECTORIES = 7;

        private FileStream file;
        private Hashtable sections;

        public string fileName { get; private set; }

        public Parser(string fileName)
        {
            this.fileName = fileName;
            this.file = File.Open(this.fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            this.sections = new Hashtable();
            GetSections();
        }

        ~Parser()
        {
            file.Close();
            sections.Clear();
        }

        private void GetSections()
        {
            uint section, lenght, offset;
            BinaryReader reader = new BinaryReader(this.file);

            try
            {
                offset = 0;
                while (true)
                {
                    section = reader.ReadUInt32();
                    lenght = reader.ReadUInt32();
                    offset += 8;
                    this.sections[section] = offset;
                    offset += lenght;
                    reader.BaseStream.Seek(lenght, SeekOrigin.Current);
                }
            }
            catch (EndOfStreamException)
            { }
        }

        public UInt64 GetNumberOfPhotons() 
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = (uint)(this.sections[(uint?)MCML_SECTION_NUMBER_OF_PHOTONS]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            UInt64 numberOfPhotons = reader.ReadUInt64();
            return numberOfPhotons;
        }

        public Area GetArea()
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = (uint)(this.sections[(uint?)MCML_SECTION_AREA]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);

            Double3 corner = new Double3(reader.ReadDouble(), reader.ReadDouble(),
                reader.ReadDouble());
            Double3 length = new Double3(reader.ReadDouble(), reader.ReadDouble(),
                reader.ReadDouble());
            Int3 partitionNumber = new Int3(reader.ReadInt32(), reader.ReadInt32(),
                reader.ReadInt32());
            Area area = new Area(corner, length, partitionNumber);

            return area;
        }
        
        public double GetSpecularReflectance()
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = (uint)(this.sections[(uint?)MCML_SECTION_SPECULAR_REFLECTANCE]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            double specularReflecrance = reader.ReadDouble();
            return specularReflecrance;
        }

        public Detector[] GetDetectors()
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = (uint)(this.sections[(uint?)MCML_SECTION_DETECTORS]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            int numberOfDetectors = reader.ReadInt32();
            
            Detector[] detectors = new Detector[numberOfDetectors];
            Double3[] center = new Double3[numberOfDetectors];
            Double3[] length = new Double3[numberOfDetectors];
            for (int i = 0; i < numberOfDetectors; ++i)
            {
                center[i] = new Double3(reader.ReadDouble(), reader.ReadDouble(),
                    reader.ReadDouble());
                length[i] = new Double3(reader.ReadDouble(), reader.ReadDouble(),
                    reader.ReadDouble());
            }
            
            offset = (uint)(this.sections[(uint?)MCML_SECTION_DETECTOR_WEIGHTS]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            reader.ReadInt32();
            for (int i = 0; i < numberOfDetectors; ++i)
            {
                detectors[i] = new Detector(center[i], length[i], reader.ReadDouble());
            }

            return detectors;
        }

        public double[] GetTrajectories()
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = (uint)(this.sections[(uint?)MCML_SECTION_COMMON_TRAJECTORIES]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);
            
            int numberOfValues = reader.ReadInt32();
            double[] trajectories = new double[numberOfValues];

            for (int i = 0; i < numberOfValues; ++i)
            {
                trajectories[i] = reader.ReadDouble();
            }

            return trajectories;
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
                    for (int j = 0; j < numberOfValues; ++j)
                    {
                        reader.ReadDouble();
                    }
                }
            }

            return null;
        }

        public UInt64 GetNumberOfPhotonsInDetector(int detectorId)
        {
            BinaryReader reader = new BinaryReader(this.file);
            uint offset = (uint)(this.sections[(uint?)MCML_SECTION_DETECTOR_TRAJECTORIES]);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);

            int numberOfDetectors = reader.ReadInt32();
            for (int i = 0; i < numberOfDetectors; ++i)
            {
                UInt64 numberOfPhotons = reader.ReadUInt64();
                int numberOfValues = reader.ReadInt32();
                if (i == detectorId)
                {
                    return numberOfPhotons;
                }
                for (int j = 0; j < numberOfValues; ++j)
                {
                    reader.ReadDouble();
                }
            }

            return 0;
        }
    }
}
