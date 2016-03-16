using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;
using mcmlVisualizer;
using System.Drawing;

namespace WebServiceMCMLViewer
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
        private const String MCML_TEXT = "MCML_TEXT";
        private const String MCML_NUMBER_DETECTORS = "NUMBER_DETECTORS";
        private const String MCML_LABEL_MODE = "LABEL_MODE";
        private const String SCALED_COEFFICIENT = "SCALED_COEFFICIENT";
        private const String PARSER = "PARSER";
        private const String TRAJECTORY_BOX = "TRAJECTORY_BOX";
        private const String GRID_DETECTORS_DATA = "GRID_DETECTORS_DATA";
        private const String INIT_FILES = "INIT_FILES";

        private const String RESULTS_FOLDER = "RESULTS_FOLDER";
        private const String TMP_LOGO_FOLDER = "TMP_LOGO_FOLDER";
        private const String TMP_IMAGE_FOLDER = "TMP_IMAGE_FOLDER";

        private const String FILE_PATHS = "D://PATHS.cfg";

        private const String VAR_X = "VAR_X";
        private const String VAR_Y = "VAR_Y";
        private const String VAR_Z = "VAR_Z";

        private String[] modeNameString
            = { "Карта поглощения", "Карта рассеивания", "Глубинная карта", "Детекторы", "Сетка детекторов" };

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

        private Area GetAreaOfTrajectoryBox()
        {
            if (Application[TRAJECTORY_BOX] == null)
            {
                return null;
            }
            else
            {
                return ((TrajectoryBox)Application[TRAJECTORY_BOX]).area;
            }
        }

        [WebMethod]
        public String GetMode()
        {
            if (Application[MCML_LABEL_MODE] == null)
            {
                return null;
            }
            else
            {
                return (String)Application[MCML_LABEL_MODE];
            }
        }

        [WebMethod]
        public bool OpenMCML(String ID_MCML, int MODE_OPEN, int detectorID)
        {
            Application[RESULTS_FOLDER] = GetValueFromPaths("RESULTS_FOLDER");
            Application[TMP_IMAGE_FOLDER] = GetValueFromPaths("TMP_IMAGE_FOLDER");
            Application[TMP_LOGO_FOLDER] = GetValueFromPaths("TMP_LOGO_FOLDER");

            String path
                = (String)Application[RESULTS_FOLDER] + "/" + ID_MCML + "/" + ID_MCML + ".mcml.out";

            Parser parser = null;
            TrajectoryBox trajectoryBox = null;
            try
            {
                parser = Parser.getInstance(path);
                if (parser == null)
                    { return false; }

                SetInformation(parser);
                Application[MCML_NUMBER_DETECTORS] = parser.GetNumberOfDetectors();
                switch(MODE_OPEN)
                {
                    case 1:
                        {
                            Application[MCML_LABEL_MODE] = modeNameString[1];
                            trajectoryBox = new TrajectoryBox(parser.GetArea(), parser.GetTrajectoriesScattering());
                            break;
                        }
                    case 2:
                        {
                            Application[MCML_LABEL_MODE] = modeNameString[2];
                            trajectoryBox = new TrajectoryBox(parser.GetArea(), parser.GetTrajectoriesDepth());
                            break;
                        }
                    case 3:
                        {
                            Application[MCML_LABEL_MODE] = modeNameString[3];
                            trajectoryBox = new TrajectoryBox(parser.GetArea(), parser.GetDetectorTrajectories(detectorID));
                            break;
                        }
                    case 4:
                        {
                            Application[MCML_LABEL_MODE] = modeNameString[4];
                            Area area = parser.GetArea();
                            Int3 partitionNumber = new Int3(area.partitionNumber.x,
                                area.partitionNumber.y, 0);

                            Area new_area = new Area(area.corner, area.length, partitionNumber); 
                            trajectoryBox = new TrajectoryBox(new_area, parser.GetGridDetectorWeights());
                            break;
                        }
                    default:
                        {
                            Application[MCML_LABEL_MODE] = modeNameString[0];
                            trajectoryBox = new TrajectoryBox(parser.GetArea(), parser.GetTrajectoriesAbsorption());
                            break;
                        }
                }
                Application[SCALED_COEFFICIENT] = ComputeScaledCoefficient(trajectoryBox);
                Application[PARSER] = parser;
                Application[TRAJECTORY_BOX] = trajectoryBox;
                return true;
            }
            catch
            {
                return false;
            }
        }

        private double ComputeScaledCoefficient(TrajectoryBox trajectoryBox)
        {
            if (trajectoryBox.trajectories.Length == 0)
            {
                return 0.0;
            }

            double max = trajectoryBox.trajectories[0];
            for (int i = 1; i < trajectoryBox.trajectories.Length; ++i)
            {
                if (trajectoryBox.trajectories[i] > max)
                {
                    max = trajectoryBox.trajectories[i];
                }
            }
            return max;
        }

        private void SetInformation(Parser parser)
        {
            Application[MCML_TEXT] = "";

            ulong numberOfPhotons = parser.GetNumberOfPhotons();
            Application[MCML_TEXT] += "Number of photons: " + numberOfPhotons + "\n";
            Application[MCML_TEXT] += "Specular reflectance: " + parser.GetSpecularReflectance() + "\n";

            double[] weights = parser.GetDetectorWeights();
            double[] targetRanges = parser.GetDetectorTargetRanges();

            UInt64[] numberOfPhotonsPerDetector = parser.GetNumberOfPhotonsInDetectorAsArray();
            UInt64 totalNumberOfPhotonsInDetectors = 0;
            for (int i = 0; i < numberOfPhotonsPerDetector.Length; ++i)
            {
                totalNumberOfPhotonsInDetectors += numberOfPhotonsPerDetector[i];
            }

            Application[MCML_TEXT] += "Detectors: (" + totalNumberOfPhotonsInDetectors.ToString() + ")\n";
            for (int i = 0; i < weights.Length; ++i)
            {
                Application[MCML_TEXT] += "\tDetector " + i.ToString() + ": " + weights[i] / numberOfPhotons;
                if (i < targetRanges.Length)
                {
                    Application[MCML_TEXT] += " | " + targetRanges[i] / numberOfPhotons;
                }
                Application[MCML_TEXT] += " (" + numberOfPhotonsPerDetector[i] + ")" + "\n";
            }
        }

        [WebMethod]
        public String GetInfoMCML()
        {
            if (Application[MCML_TEXT] == null)
            {
                return null;
            }
            else
            {
                return (String)Application[MCML_TEXT];
            }
        }

        private double GetScaledCoefficient()
        {
            if (Application[SCALED_COEFFICIENT] == null)
            {
                return -1;
            }
            else
            {
                return (double)Application[SCALED_COEFFICIENT];
            }
        }

        private Color GetColor(double weight)
        {
            if (weight < 0) weight = 0;
            if (weight > 1) weight = 1;

            if (weight < 1.0 / 3.0)
            {
                weight *= 3;
                return Color.FromArgb((byte)(weight * 255), 0, 0);
            }
            else if (weight < 2.0 / 3.0)
            {
                weight = 3 * weight - 1.0;
                return Color.FromArgb(255, (byte)(255 * weight), 0);
            }
            else
            {
                weight = 3 * weight - 2.0;
                return Color.FromArgb(255, 255, (byte)(255 * weight));
            }
        }

        private Byte[] GetBitmap(double[] section, double width, double height,
            int partitionWidth, int partitionHeight)
        {
            double stepWidth, stepHeight;
            stepWidth = width / partitionWidth;
            stepHeight = height / partitionHeight;

            double scaledCoefficient = GetScaledCoefficient();
            Byte[] bitmap = new Byte[4 * partitionHeight * partitionWidth];

            for (int i = 0; i < partitionWidth; ++i)
            {
                for (int j = 0; j < partitionHeight; ++j)
                {
                    int index = i * partitionHeight + j;
                    double weight = section[index];

                    if (weight < 1.0)
                        weight = 1.0;

                    Color color = GetColor(Math.Log10(section[index]) / Math.Log10(scaledCoefficient));

                    bitmap[4 * index] = color.B;
                    bitmap[4 * index + 1] = color.G;
                    bitmap[4 * index + 2] = color.R;
                    bitmap[4 * index + 3] = color.A;
                }
            }
            return bitmap;
        }

        private String GetPathDirectory(String KEY_SESSION)
        {
            String dirName = (String)Application[TMP_IMAGE_FOLDER];

            if (!System.IO.Directory.Exists(dirName))
            {
                return null;
            }

            dirName += "/" + KEY_SESSION;

            if (!System.IO.Directory.Exists(dirName))
            {
                System.IO.Directory.CreateDirectory(dirName);
            }

            return dirName;
        }

        private String GetPathLogoDir(String KEY_SESSION)
        {
            String dirName = (String)Application[TMP_LOGO_FOLDER];

            if (!System.IO.Directory.Exists(dirName))
            {
                return null;
            }

            dirName += "/" + KEY_SESSION;

            if (!System.IO.Directory.Exists(dirName))
            {
                System.IO.Directory.CreateDirectory(dirName);
            }

            return dirName;
        }

        private Bitmap GetBitmap(Byte[] bitmap, int partitionWidth, int partitionHeight)
        {
            Bitmap imgOriginal = new Bitmap(partitionWidth, partitionHeight);

            Rectangle rect = new Rectangle(0, 0, partitionWidth, partitionHeight);
            System.Drawing.Imaging.BitmapData bmpData =
                imgOriginal.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly,
                imgOriginal.PixelFormat);

            IntPtr ptr = bmpData.Scan0;// получение адреса первой строки
            System.Runtime.InteropServices.Marshal.Copy(bitmap, 0, ptr, bitmap.Length);

            imgOriginal.UnlockBits(bmpData);

            return imgOriginal;
        }

        private void SetVar(String type, double var)
        {
            String strVar = var.ToString();
            switch (type)
            {
                case "XZ":
                    {
                        { Application[VAR_Y] = strVar; }
                        break;
                    }
                case "YZ":
                    {
                        { Application[VAR_X] = strVar; }
                        break;
                    }
                default:
                    {
                        { Application[VAR_Z] = strVar; }
                        break;
                    }
            }
        }

        private String GetVar(String type)
        {
            String var = null;
            switch (type)
            {
                case "XZ":
                    {
                        if (Application[VAR_Y] != null)
                        { var = (String)Application[VAR_Y]; }
                        break;
                    }
                case "YZ":
                    {
                        if (Application[VAR_X] != null)
                        { var = (String)Application[VAR_X]; }
                        break;
                    }
                default:
                    {
                        if (Application[VAR_Z] != null)
                        { var = (String)Application[VAR_Z]; }
                        break;
                    }
            }
            try { Convert.ToDouble(var);}
            catch { var = null; }        
            return var;
        }

        private String AdditiveName(String type)
        {
            String additive = null;
            switch (type)
            {
                case "XZ":
                    {
                        if (Application[VAR_Y] != null)
                        { additive = "Y=" + (String)Application[VAR_Y]; }
                        break;
                    }
                case "YZ":
                    {
                        if (Application[VAR_X] != null)
                        { additive = "X=" + (String)Application[VAR_X]; }
                        break;
                    }
                default:
                    {
                        if (Application[VAR_Z] != null)
                        { additive = "Z=" + (String)Application[VAR_Z]; }
                        break;    
                    }
            }
            return additive;
        }

        private String WriteMatrixTXT(String KEY_SESSION, String type)
        {
            String dirName;
            int partitionWidth, partitionHeight;
            double[] section;
            double var;

            var = Convert.ToDouble(GetVar(type));

            bool isOk = SetMainValues(KEY_SESSION, type, var, out dirName, out section, out partitionWidth, out partitionHeight);
            if (isOk == false)
            {
                return null;
            }

            String pathMatrix = dirName + "/MATRIX_" + AdditiveName(type) + ".txt";

            System.IO.StreamWriter stream =
                new System.IO.StreamWriter(pathMatrix);

            for (int i = 0; i < partitionWidth; ++i)
            {
                for (int j = 0; j < partitionHeight; ++j)
                {
                    int index = i * partitionHeight + j;
                    stream.Write(section[index]);
                    stream.Write('\t');
                }
                stream.WriteLine();
            }
            stream.Flush();
            stream.Close();

            return pathMatrix;
        }

        private String WriteTimeScales(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];
            ulong numberOfPhotons = parser.GetNumberOfPhotons();
            int numberOfDetectors = parser.GetNumberOfDetectors();

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathTimeScales = dirName + "/TIME_SCALES.txt";
            System.IO.StreamWriter writer =
                new System.IO.StreamWriter(pathTimeScales);

            writer.Write("Detector/Time");
            TimeInfo[] timeInfo = parser.GetDetectorTimeScale(0);

            if (timeInfo == null)
            {
                writer.Write("\tEmpty");
            }
            else
            {
                foreach (TimeInfo element in timeInfo)
                {
                    writer.Write('\t');
                    writer.Write(element.timeStart);
                }
            }
            writer.WriteLine();

            for (int i = 0; i < numberOfDetectors; ++i)
            {
                timeInfo = parser.GetDetectorTimeScale(i);
                writer.Write(i);
                foreach (TimeInfo element in timeInfo)
                {
                    writer.Write('\t');
                    writer.Write(element.weight / numberOfPhotons);
                }
                writer.WriteLine();
            }
            writer.Flush();
            writer.Close();

            return pathTimeScales;
        }

        private String WriteAllAsText(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathAllAsText = dirName + "/ALL_AS_TEXT.txt";

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathAllAsText);
            writer.Write(parser.GetText().ToString());
            writer.Flush();
            writer.Close();

            return pathAllAsText;
        }

        private String WriteDetectorsWeigths(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathWeigths= dirName + "/WEIGHTS.txt";

            double[] weights = parser.GetDetectorWeights();
            UInt64 numberOfPhotons = parser.GetNumberOfPhotons();

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathWeigths);

            if (weights.Length == 0)
            {
                writer.WriteLine("Empty");
            }
            else
            {
                foreach (double weight in weights)
                {
                    writer.WriteLine(weight / numberOfPhotons);
                }
            }

            writer.Flush();
            writer.Close();

            return pathWeigths;
        }

        private String WriteDetectorsWeigthsWithoutNorma(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathWeigths = dirName + "/WEIGHTS.txt";

            double[] weights = parser.GetDetectorWeights();

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathWeigths);

            if (weights.Length == 0)
            {
                writer.WriteLine("Empty");
            }
            else
            {
                foreach (double weight in weights)
                {
                    writer.WriteLine(weight);
                }
            }

            writer.Flush();
            writer.Close();

            return pathWeigths;
        }

        private String WriteNumberOfPhotonsPerDetector(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathWeigths = dirName + "/WEIGHTS.txt";

            int numberOfDetectors = parser.GetNumberOfDetectors();
            UInt64[] numberOfPhotonsPerDetector = parser.GetNumberOfPhotonsInDetectorAsArray();

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathWeigths);

            if (numberOfDetectors == 0)
            {
                writer.WriteLine("Empty");
            }
            else
            {
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(numberOfPhotonsPerDetector[i]);
                }
            }

            writer.Flush();
            writer.Close();

            return pathWeigths;
        }

        private String WriteOtherRanges(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathRanges = dirName + "/RANGES.txt";

            int numberOfDetectors = parser.GetNumberOfDetectors();
            double[] weights = parser.GetDetectorWeights();
            double[] targetRanges = parser.GetDetectorTargetRanges();
            UInt64 numberOfPhotons = parser.GetNumberOfPhotons();

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathRanges);
            if (numberOfDetectors == 0)
            {
                writer.WriteLine("Empty");
            }
            else
            {
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine((weights[i] - targetRanges[i]) / numberOfPhotons);
                }
            }

            writer.Flush();
            writer.Close();

            return pathRanges;
        }

        private String WriteOtherRangesWithoutNorma(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathRanges = dirName + "/RANGES.txt";

            int numberOfDetectors = parser.GetNumberOfDetectors();
            double[] weights = parser.GetDetectorWeights();
            double[] targetRanges = parser.GetDetectorTargetRanges();

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathRanges);
            if (numberOfDetectors == 0)
            {
                writer.WriteLine("Empty");
            }
            else
            {
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(weights[i] - targetRanges[i]);
                }
            }
            writer.Flush();
            writer.Close();

            return pathRanges;
        }

        private String WriteTargetRanges(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathRanges = dirName + "/RANGES.txt";

            int numberOfDetectors = parser.GetNumberOfDetectors();
            double[] weights = parser.GetDetectorWeights();
            double[] targetRanges = parser.GetDetectorTargetRanges();
            UInt64 numberOfPhotons = parser.GetNumberOfPhotons();

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathRanges);
            if (numberOfDetectors == 0)
            {
                writer.WriteLine("Empty");
            }
            else
            {
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(targetRanges[i] / numberOfPhotons);
                }
            }

            writer.Flush();
            writer.Close();

            return pathRanges;
        }

        private String WriteTargetRangesWithoutNorma(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }

            Parser parser = (Parser)Application[PARSER];

            String dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
            {
                return null;
            }

            String pathRanges = dirName + "/RANGES.txt";

            int numberOfDetectors = parser.GetNumberOfDetectors();
            double[] weights = parser.GetDetectorWeights();
            double[] targetRanges = parser.GetDetectorTargetRanges();

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathRanges);
            if (numberOfDetectors == 0)
            {
                writer.WriteLine("Empty");
            }
            else
            {
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(targetRanges[i]);
                }
            }

            writer.Flush();
            writer.Close();

            return pathRanges;
        }

        private bool isCorrectPartition(String type, double var)
        {
            Area area = GetAreaOfTrajectoryBox();
            bool isCorrect = true;
            switch(type)
            {
                case "XZ":
                    {
                        if (var < area.corner.y || var > area.corner.y + area.length.y)
                            { isCorrect = false; }
                        break;
                    }
                case "YZ":
                    {
                        if (var < area.corner.x || var > area.corner.x + area.length.x)
                            { isCorrect = false; }
                        break;
                    }
                default:
                    {
                        if (var < area.corner.z || var > area.corner.z + area.length.z)
                            { isCorrect = false; }
                        break;
                    }
            }
            return isCorrect;
        }

        private bool SetMainValues(String KEY_SESSION, String type, double var, out String dirName,
            out double[] section, out int partitionWidth, out int partitionHeight)
        {
            dirName = null;
            partitionWidth = 0;
            partitionHeight = 0;
            section = null;
          
            if (((TrajectoryBox)Application[TRAJECTORY_BOX]).trajectories.Length == 0)
                { return false; }

            dirName = GetPathDirectory(KEY_SESSION);

            if (dirName == null)
                { return false; }
            
            bool isOk = isCorrectPartition(type, var);
            if (!isOk) { return false; }

            Area area = GetAreaOfTrajectoryBox();

            if (String.Equals(GetMode(), "Сетка детекторов"))
            {
                section = ((TrajectoryBox)Application[TRAJECTORY_BOX]).trajectories;
                partitionWidth = area.partitionNumber.x;
                partitionHeight = area.partitionNumber.y;
            }
            else
            {
                switch (type)
                {
                    case "XZ":
                        {
                            section = ((TrajectoryBox)Application[TRAJECTORY_BOX]).GetSectionXZ(var);
                            partitionWidth = area.partitionNumber.x;
                            partitionHeight = area.partitionNumber.z;
                            break;
                        }
                    case "YZ":
                        {
                            section = ((TrajectoryBox)Application[TRAJECTORY_BOX]).GetSectionYZ(var);
                            partitionWidth = area.partitionNumber.y;
                            partitionHeight = area.partitionNumber.z;
                            break;
                        }
                    default:
                        {
                            section = ((TrajectoryBox)Application[TRAJECTORY_BOX]).GetSectionXY(var);
                            partitionWidth = area.partitionNumber.x;
                            partitionHeight = area.partitionNumber.y;
                            break;
                        }
                }
            }
            return true;
        }

        [WebMethod]
        public bool InitFiles(String KEY_SESSION, double width, double height, String type, double var)
        {
            Application[INIT_FILES] = null;
            SetVar(type, var);

            String dirName;
            int partitionWidth, partitionHeight;
            double[] section;

            bool isOk;      
            isOk = SetMainValues(KEY_SESSION, type, var, out dirName, out section, out partitionWidth, out partitionHeight);
            if (isOk == false)
            {
                return false;
            }

            Byte[] bitmap = GetBitmap(section, width, height, partitionWidth, partitionHeight);

            System.IO.Stream file;

            Bitmap imgOriginal = GetBitmap(bitmap, partitionWidth, partitionHeight);

            bool isChange = false;
            const int minPx = 10;
            if (partitionWidth < minPx)
            {
                int newPartitionWidth = minPx;
                partitionHeight = partitionHeight / partitionWidth * newPartitionWidth;
                partitionWidth = newPartitionWidth;
                isChange = true;
            }
            if (partitionHeight < minPx)
            {
                int newPartitionHeight = minPx;
                partitionWidth = partitionWidth / partitionHeight * newPartitionHeight;
                partitionHeight = newPartitionHeight;
                isChange = true;
            }
            if (isChange)
            {
                imgOriginal = new Bitmap(imgOriginal, (int)partitionWidth, (int)partitionHeight);
            }

            file = System.IO.File.Create(dirName + "/ORIGINAL_" + AdditiveName(type) + ".jpg");
            
            imgOriginal.Save(file, System.Drawing.Imaging.ImageFormat.Jpeg);
            file.Close();

            String dirLogoFolder = GetPathLogoDir(KEY_SESSION);

            if (dirLogoFolder != null)
            {
                Bitmap imgLogo = new Bitmap(imgOriginal, (int)width, (int)height);

                file = System.IO.File.Create(dirLogoFolder + "/LOGO_" + AdditiveName(type) + ".jpg");

                imgLogo.Save(file, System.Drawing.Imaging.ImageFormat.Jpeg);
                file.Close();
            }

            Application[INIT_FILES] = true;
            return true;
        }

        [WebMethod]
        public String GetNameImageLogo(String KEY_SESSION, String type)
        {
            if(Application[INIT_FILES] == null)
            {
                return null;
            }
            else
            {
                return KEY_SESSION + "/LOGO_" + AdditiveName(type) + ".jpg";
            }
        }

        [WebMethod]
        public String GetPathImageOriginal(String KEY_SESSION, String type)
        {
            if (Application[INIT_FILES] == null)
            {
                return null;
            }
            else
            {
                return GetPathDirectory(KEY_SESSION) + "/ORIGINAL_" + AdditiveName(type) + ".jpg";
            }
        }

        [WebMethod]
        public String GetPathMatrix(String KEY_SESSION, String type)
        {
            if (Application[INIT_FILES] == null)
            {
                return null;
            }
            else
            {
                String pathMatrix = WriteMatrixTXT(KEY_SESSION, type);
                return pathMatrix;
            }
        }

        [WebMethod]
        public String GetPathTimeScales(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }
            else
            {
                String pathTimeScales = WriteTimeScales(KEY_SESSION);
                return pathTimeScales;
            }
        }

        [WebMethod]
        public String GetPathAllAsText(String KEY_SESSION)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }
            else
            {
                String pathAllAsText = WriteAllAsText(KEY_SESSION);
                return pathAllAsText;
            }
        }

        [WebMethod]
        public String GetPathWeights(String KEY_SESSION, int mode)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }
            else
            {
                String pathWeights;

                switch (mode)
                {
                    case 1:
                        {
                            pathWeights = WriteDetectorsWeigthsWithoutNorma(KEY_SESSION);
                            break;
                        }
                    case 2:
                        {
                            pathWeights = WriteNumberOfPhotonsPerDetector(KEY_SESSION);
                            break;
                        }
                    default:
                        {
                            pathWeights = WriteDetectorsWeigths(KEY_SESSION); 
                            break;
                        }
                }

                return pathWeights;
            }
        }

        [WebMethod]
        public String GetPathRanges(String KEY_SESSION, int mode)
        {
            if (Application[PARSER] == null)
            {
                return null;
            }
            else
            {
                String pathRanges;

                switch (mode)
                {
                    case 1:
                        {
                            pathRanges = WriteOtherRangesWithoutNorma(KEY_SESSION);
                            break;
                        }
                    case 2:
                        {
                            pathRanges = WriteTargetRanges(KEY_SESSION);
                            break;
                        }
                    case 3:
                        {
                            pathRanges = WriteTargetRangesWithoutNorma(KEY_SESSION);
                            break;
                        }
                    default:
                        {
                            pathRanges = WriteOtherRanges(KEY_SESSION);
                            break;
                        }
                }

                return pathRanges;
            }
        }

        [WebMethod]
        public List<double> GetInfoOfArea(String type)
        {
            if (Application[TRAJECTORY_BOX] == null)
            {
                return null;
            }
            else
            {
                List<double> infoOfArea = new List<double>();

                switch (type)
                {
                    case "XY":
                        {
                            infoOfArea.Add(((TrajectoryBox)Application[TRAJECTORY_BOX]).area.corner.z);
                            infoOfArea.Add(((TrajectoryBox)Application[TRAJECTORY_BOX]).area.length.z);
                            infoOfArea.Add((double)((TrajectoryBox)Application[TRAJECTORY_BOX]).area.partitionNumber.z);
                            break;
                        }
                    case "XZ":
                        {
                            infoOfArea.Add(((TrajectoryBox)Application[TRAJECTORY_BOX]).area.corner.y);
                            infoOfArea.Add(((TrajectoryBox)Application[TRAJECTORY_BOX]).area.length.y);
                            infoOfArea.Add((double)((TrajectoryBox)Application[TRAJECTORY_BOX]).area.partitionNumber.y);
                            break;
                        }
                    case "YZ":
                        {
                            infoOfArea.Add(((TrajectoryBox)Application[TRAJECTORY_BOX]).area.corner.x);
                            infoOfArea.Add(((TrajectoryBox)Application[TRAJECTORY_BOX]).area.length.x);
                            infoOfArea.Add((double)((TrajectoryBox)Application[TRAJECTORY_BOX]).area.partitionNumber.x);
                            break;
                        }
                    default:
                        {
                            infoOfArea = null;
                            break;
                        }
                }

                return infoOfArea;
            }
        }

        [WebMethod]
        public int GetNumberOfDetectors()
        {
            if (Application[MCML_NUMBER_DETECTORS] == null)
            {
                return 0;
            }
            else
            {
                return (int)Application[MCML_NUMBER_DETECTORS];
            }
        }
    }
}