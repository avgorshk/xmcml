using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.IO;
using Microsoft.Win32;

namespace mcmlVisualizer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Parser parser = null;
        private TrajectoryBox trajectoryBox = null;
        private double scaledCoefficient = 0.0;

        private double[] sectionXY = null;
        private double[] sectionXZ = null;
        private double[] sectionYZ = null;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void MainMenu_Exit_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        private void MainMenu_Open_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.FileOk += new System.ComponentModel.CancelEventHandler(openFileDialog_FileOk);
            openFileDialog.Filter = "MCML Output files (*.mcml.out, *.mcml.bk)|*.mcml.out;*.mcml.bk";
            openFileDialog.Title = "Open MCML Output files...";
            openFileDialog.ShowDialog();
        }

        void openFileDialog_FileOk(object sender, System.ComponentModel.CancelEventArgs e)
        {
            try
            {
                OpenFileDialog openFileDialog = sender as OpenFileDialog;
                this.Title = "mcmlVisualizer: " + openFileDialog.FileName;
                parser = new Parser(openFileDialog.FileName);
                
                SetInformation(parser);

                int numberOfDetectors = parser.GetNumberOfDetectors();
                ((MenuItem)MainMenu.Items[1]).Items.Clear();
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    MenuItem item = new MenuItem();
                    item.Header = i.ToString();
                    item.Click += new RoutedEventHandler(item_Click);
                    ((MenuItem)MainMenu.Items[1]).Items.Add(item);
                }
                
                trajectoryBox = new TrajectoryBox(parser.GetArea(), parser.GetTrajectories());
                scaledCoefficient = ComputeScaledCoefficient();

                double stepSizeX = trajectoryBox.area.length.x / trajectoryBox.area.partitionNumber.x;
                sliderX.Minimum = trajectoryBox.area.corner.x + stepSizeX / 2.0;
                sliderX.Maximum = trajectoryBox.area.corner.x + trajectoryBox.area.length.x - stepSizeX / 2.0;
                sliderX.Value = (sliderX.Maximum + sliderX.Minimum) / 2.0;
                sliderX.TickFrequency = stepSizeX;
                sliderX.ToolTip = sliderX.Value.ToString();
                PaintYZ(sliderX.Value);

                double stepSizeY = trajectoryBox.area.length.y / trajectoryBox.area.partitionNumber.y;
                sliderY.Minimum = trajectoryBox.area.corner.y + stepSizeY / 2.0;
                sliderY.Maximum = trajectoryBox.area.corner.y + trajectoryBox.area.length.y - stepSizeY / 2.0;
                sliderY.Value = (sliderY.Maximum + sliderY.Minimum) / 2.0;
                sliderY.TickFrequency = stepSizeY;
                sliderY.ToolTip = sliderY.Value.ToString();
                PaintXZ(sliderY.Value);

                double stepSizeZ = trajectoryBox.area.length.z / trajectoryBox.area.partitionNumber.z;
                sliderZ.Minimum = trajectoryBox.area.corner.z + stepSizeZ / 2.0;
                sliderZ.Maximum = trajectoryBox.area.corner.z + trajectoryBox.area.length.z - stepSizeZ / 2.0;
                sliderZ.Value = sliderZ.Minimum;
                sliderZ.TickFrequency = stepSizeZ;
                sliderZ.ToolTip = sliderZ.Value.ToString();
                PaintXY(sliderZ.Value);
            }
            catch (Exception exc)
            {
                MessageBox.Show("Can\'t read file. " + exc.Message, "Error");
            }
        }

        void item_Click(object sender, RoutedEventArgs e)
        {
            int detectorId = int.Parse(((MenuItem)sender).Header.ToString());
            trajectoryBox = new TrajectoryBox(parser.GetArea(), parser.GetDetectorTrajectories(detectorId));
            scaledCoefficient = parser.GetNumberOfPhotonsInDetector(detectorId);
            scaledCoefficient += 2;

            double stepSizeX = trajectoryBox.area.length.x / trajectoryBox.area.partitionNumber.x;
            sliderX.Minimum = trajectoryBox.area.corner.x + stepSizeX / 2.0;
            sliderX.Maximum = trajectoryBox.area.corner.x + trajectoryBox.area.length.x - stepSizeX / 2.0;
            sliderX.Value = (sliderX.Maximum + sliderX.Minimum) / 2.0;
            sliderX.TickFrequency = stepSizeX;
            sliderX.ToolTip = sliderX.Value.ToString();
            PaintYZ(sliderX.Value);

            double stepSizeY = trajectoryBox.area.length.y / trajectoryBox.area.partitionNumber.y;
            sliderY.Minimum = trajectoryBox.area.corner.y + stepSizeY / 2.0;
            sliderY.Maximum = trajectoryBox.area.corner.y + trajectoryBox.area.length.y - stepSizeY / 2.0;
            sliderY.Value = (sliderY.Maximum + sliderY.Minimum) / 2.0;
            sliderY.TickFrequency = stepSizeY;
            sliderY.ToolTip = sliderY.Value.ToString();
            PaintXZ(sliderY.Value);

            double stepSizeZ = trajectoryBox.area.length.z / trajectoryBox.area.partitionNumber.z;
            sliderZ.Minimum = trajectoryBox.area.corner.z + stepSizeZ / 2.0;
            sliderZ.Maximum = trajectoryBox.area.corner.z + trajectoryBox.area.length.z - stepSizeZ / 2.0;
            sliderZ.Value = sliderZ.Minimum;
            sliderZ.TickFrequency = stepSizeZ;
            sliderZ.ToolTip = sliderZ.Value.ToString();
            PaintXY(sliderZ.Value);
        }

        void SetInformation(Parser parser)
        {
            textBoxInformation.Text = "";

            ulong numberOfPhotons = parser.GetNumberOfPhotons();
            textBoxInformation.Text += "Number of photons: " + numberOfPhotons + "\n";
            textBoxInformation.Text += "Specular reflectance: " + parser.GetSpecularReflectance() + "\n";

            double[] weights = parser.GetDetectorWeights();
            double[] otherRanges = parser.GetDetectorOtherRanges();
            double[] targetRanges = parser.GetDetectorTargetRanges();

            UInt64[] numberOfPhotonsPerDetector = parser.GetNumberOfPhotonsInDetectorAsArray();
            UInt64 totalNumberOfPhotonsInDetectors = 0;
            for (int i = 0; i < numberOfPhotonsPerDetector.Length; ++i)
            {
                totalNumberOfPhotonsInDetectors += numberOfPhotonsPerDetector[i];
            }

            textBoxInformation.Text += "Detector weights: (" + totalNumberOfPhotonsInDetectors.ToString() +")\n";
            for (int i = 0; i < weights.Length; ++i)
            {
                textBoxInformation.Text += "\tDetector " + i.ToString() + ": " +
                    weights[i] / numberOfPhotons + " (" + numberOfPhotonsPerDetector[i] + ")" + "\n";
            }            

            textBoxInformation.Text += "Detector ranges: (" + totalNumberOfPhotonsInDetectors.ToString() + ")\n";
            for (int i = 0; i < weights.Length; ++i)
            {
                textBoxInformation.Text += "\tDetector " + i.ToString() + ": " +
                    (otherRanges[i]  / numberOfPhotons) + " | " +
                    (targetRanges[i] / numberOfPhotons) + 
                    " (" + numberOfPhotonsPerDetector[i] + ")" + "\n";
            }
        }

        double ComputeScaledCoefficient()
        {
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

        void PaintXY(double z)
        {
            gridXY.Children.Clear();

            double[] section = trajectoryBox.GetSectionXY(z);
            sectionXY = section;
            
            double height = gridXY.ActualHeight;
            double width = gridXY.ActualWidth;
            double stepX = width / trajectoryBox.area.partitionNumber.x;
            double stepY = height / trajectoryBox.area.partitionNumber.y;

            for (int ix = 0; ix < trajectoryBox.area.partitionNumber.x; ++ix)
            {
                for (int iy = 0; iy < trajectoryBox.area.partitionNumber.y; ++iy)
                {
                    int index = ix * trajectoryBox.area.partitionNumber.y + iy;
                    double weight = section[index];
                    if (weight < 1.0) 
                        weight = 1.0;
                    
                    SolidColorBrush brush = new SolidColorBrush(
                        GetColor(Math.Log10(weight) / Math.Log10(scaledCoefficient)));
                    Rectangle rectangle = new Rectangle();
                    rectangle.Height = stepY;
                    rectangle.Width = stepX;
                    rectangle.Margin = new Thickness(stepX * ix, stepY * iy, 0, 0);
                    rectangle.Fill = brush;
                    rectangle.HorizontalAlignment = System.Windows.HorizontalAlignment.Left;
                    rectangle.VerticalAlignment = System.Windows.VerticalAlignment.Top;
                    gridXY.Children.Add(rectangle);
                }
            }
        }

        void PaintXZ(double y)
        {
            gridXZ.Children.Clear();

            double[] section = trajectoryBox.GetSectionXZ(y);
            sectionXZ = section;

            double height = gridXZ.ActualHeight;
            double width = gridXZ.ActualWidth;
            double stepX = width / trajectoryBox.area.partitionNumber.x;
            double stepZ = height / trajectoryBox.area.partitionNumber.z;

            for (int ix = 0; ix < trajectoryBox.area.partitionNumber.x; ++ix)
            {
                for (int iz = 0; iz < trajectoryBox.area.partitionNumber.z; ++iz)
                {
                    int index = ix * trajectoryBox.area.partitionNumber.z + iz;
                    double weight = section[index];
                    if (weight < 1.0)
                        weight = 1.0;

                    SolidColorBrush brush = new SolidColorBrush(
                        GetColor(Math.Log10(weight) / Math.Log10(scaledCoefficient)));
                    Rectangle rectangle = new Rectangle();
                    rectangle.Height = stepZ;
                    rectangle.Width = stepX;
                    rectangle.Margin = new Thickness(stepX * ix, stepZ * iz, 0, 0);
                    rectangle.Fill = brush;
                    rectangle.HorizontalAlignment = System.Windows.HorizontalAlignment.Left;
                    rectangle.VerticalAlignment = System.Windows.VerticalAlignment.Top;
                    gridXZ.Children.Add(rectangle);
                }
            }
        }

        void PaintYZ(double x)
        {
            gridYZ.Children.Clear();

            double[] section = trajectoryBox.GetSectionYZ(x);
            sectionYZ = section;

            double height = gridYZ.ActualHeight;
            double width = gridYZ.ActualWidth;
            double stepY = width / trajectoryBox.area.partitionNumber.y;
            double stepZ = height / trajectoryBox.area.partitionNumber.z;

            for (int iy = 0; iy < trajectoryBox.area.partitionNumber.y; ++iy)
            {
                for (int iz = 0; iz < trajectoryBox.area.partitionNumber.z; ++iz)
                {
                    int index = iy * trajectoryBox.area.partitionNumber.z + iz;
                    double weight = section[index];
                    if (weight < 1.0)
                        weight = 1.0;

                    SolidColorBrush brush = new SolidColorBrush(
                        GetColor(Math.Log10(weight) / Math.Log10(scaledCoefficient)));
                    Rectangle rectangle = new Rectangle();
                    rectangle.Height = stepZ;
                    rectangle.Width = stepY;
                    rectangle.Margin = new Thickness(stepY * iy, stepZ * iz, 0, 0);
                    rectangle.Fill = brush;
                    rectangle.HorizontalAlignment = System.Windows.HorizontalAlignment.Left;
                    rectangle.VerticalAlignment = System.Windows.VerticalAlignment.Top;
                    gridYZ.Children.Add(rectangle);
                }
            }
        }

        Color GetColor(double weight)
        {
            if (weight < 0) weight = 0;
            if (weight > 1) weight = 1;

            if (weight < 1.0 / 3.0)
            {
                weight *= 3;
                return Color.FromRgb((byte)(weight * 255), 0, 0);
            }
            else if (weight < 2.0 / 3.0)
            {
                weight = 3 * weight - 1.0;
                return Color.FromRgb(255, (byte)(255 * weight), 0);
            }
            else
            {
                weight = 3 * weight - 2.0;
                return Color.FromRgb(255, 255, (byte)(255 * weight));
            }
        }

        private void sliderZ_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (trajectoryBox != null)
            {
                sliderZ.ToolTip = sliderZ.Value.ToString();
                PaintXY(sliderZ.Value);
            }
        }

        private void sliderY_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (trajectoryBox != null)
            {
                sliderY.ToolTip = sliderY.Value.ToString();
                PaintXZ(sliderY.Value);
            }
        }

        private void sliderX_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (trajectoryBox != null)
            {
                sliderX.ToolTip = sliderX.Value.ToString();
                PaintYZ(sliderX.Value);
            }
        }

        private void MenuItem_XY_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Trajectory maps TXT file|*.txt";

            if (sfd.ShowDialog() == true)
            {
                StreamWriter stream = new StreamWriter(sfd.FileName);
                for (int ix = 0; ix < trajectoryBox.area.partitionNumber.x; ++ix)
                {
                    for (int iy = 0; iy < trajectoryBox.area.partitionNumber.y; ++iy)
                    {
                        int index = ix * trajectoryBox.area.partitionNumber.y + iy;
                        stream.Write(sectionXY[index]);
                        stream.Write('\t');
                    }
                    stream.WriteLine();
                }
                stream.Flush();
                stream.Close();
            }
        }

        private void MenuItem_XZ_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Trajectory maps TXT file|*.txt";

            if (sfd.ShowDialog() == true)
            {
                StreamWriter stream = new StreamWriter(sfd.FileName);
                for (int iz = 0; iz < trajectoryBox.area.partitionNumber.z; ++iz)
                {
                    for (int ix = 0; ix < trajectoryBox.area.partitionNumber.x; ++ix)
                    {
                        int index = ix * trajectoryBox.area.partitionNumber.z + iz;
                        stream.Write(sectionXZ[index]);
                        stream.Write('\t');
                    }
                    stream.WriteLine();
                }
                stream.Flush();
                stream.Close();
            }
        }

        private void MenuItem_YZ_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Trajectory maps TXT file|*.txt";

            if (sfd.ShowDialog() == true)
            {
                StreamWriter stream = new StreamWriter(sfd.FileName);
                for (int iz = 0; iz < trajectoryBox.area.partitionNumber.z; ++iz)
                {
                    for (int iy = 0; iy < trajectoryBox.area.partitionNumber.y; ++iy)
                    {
                        int index = iy * trajectoryBox.area.partitionNumber.z + iz;
                        stream.Write(sectionYZ[index]);
                        stream.Write('\t');
                    }
                    stream.WriteLine();
                }
                stream.Flush();
                stream.Close();
            }
        }

        private void CompareFiles_Click(object sender, RoutedEventArgs e)
        {
            const double EPS = 1.0E-12;
            double error;

            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Multiselect = true;
            dialog.Filter = "MCML Output files (*.mcml.out, *.mcml.bk)|*.mcml.out;*.mcml.bk";
            dialog.Title = "Choose two files for comparing";
            if (dialog.ShowDialog().Value && dialog.FileNames.Length > 1)
            {
                List<string> result = new List<string>();
                Parser parser1 = new Parser(dialog.FileNames[0]);
                Parser parser2 = new Parser(dialog.FileNames[1]);

                result.Add("Number of photons:");
                ulong numberOfPhotons1 = parser1.GetNumberOfPhotons();
                ulong numberOfPhotons2 = parser2.GetNumberOfPhotons();
                if (numberOfPhotons1 != numberOfPhotons2)
                {
                    result.Add(string.Format("{0} - {1}", numberOfPhotons1, numberOfPhotons2));
                }

                result.Add("Specular reflectance:");
                double specularReflectance1 = parser1.GetSpecularReflectance();
                double specularReflectance2 = parser2.GetSpecularReflectance();
                if (specularReflectance1 != specularReflectance2)
                {
                    result.Add(string.Format("{0} - {1} ({2})", specularReflectance1,
                        specularReflectance2, Math.Abs(specularReflectance1 - specularReflectance2)));
                }

                result.Add("Absorption:");
                double[] absorption1 = parser1.GetTrajectories();
                double[] absorption2 = parser2.GetTrajectories();
                if (absorption1.Length != absorption2.Length)
                {
                    result.Add(string.Format("Length: {0} - {1}", absorption1.Length, absorption2.Length));
                }
                else
                {
                    double maxError = 0;
                    for (int i = 0; i < absorption1.Length; ++i)
                    {
                        error = Math.Abs((absorption1[i] - absorption2[i]) / 
                            Math.Min(absorption1[i], absorption2[i]));
                        if (error > maxError)
                        {
                            maxError = error;
                        }
                        if (error > EPS)
                        {
                            result.Add(string.Format("[{0}]: {1} - {2} ({3})", i, absorption1[i],
                                absorption2[i], error));
                        }
                    }
                    result.Add("Max error " + maxError.ToString());
                }

                result.Add("Detectors:");

                int numberOfDetectors1 = parser1.GetNumberOfDetectors();
                int numberOfDetectors2 = parser2.GetNumberOfDetectors();
                if (numberOfDetectors1 != numberOfDetectors2)
                {
                    result.Add(string.Format("Number of detectors: {0} - {1}", numberOfDetectors1,
                        numberOfDetectors2));
                }
                else
                {
                    double[] weights1 = parser1.GetDetectorWeights();
                    double[] weights2 = parser2.GetDetectorWeights();

                    double[] otherRanges1 = parser1.GetDetectorOtherRanges();
                    double[] otherRanges2 = parser2.GetDetectorOtherRanges();

                    double[] targetRanges1 = parser1.GetDetectorTargetRanges();
                    double[] targetRanges2 = parser2.GetDetectorTargetRanges();

                    for (int i = 0; i < numberOfDetectors1; ++i)
                    {
                        result.Add(string.Format("Detector {0}:", i));
                        
                        ulong photonsInDetector1 = parser1.GetNumberOfPhotonsInDetector(i);
                        ulong photonsInDetector2 = parser2.GetNumberOfPhotonsInDetector(i);
                        if (photonsInDetector1 != photonsInDetector2)
                        {
                            result.Add(string.Format("Number of photons: {0} - {1}", photonsInDetector1,
                                photonsInDetector2));
                        }
                        else
                        {                            
                            error = Math.Abs((weights1[i] - weights2[i])/Math.Min(weights1[i], weights2[i]));
                            if (error > EPS)
                            {
                                result.Add(string.Format("Weight: {0} - {1} ({2})", weights1[i],
                                    weights2[i], error));
                            }
                            else
                            {
                                error = Math.Abs((otherRanges1[i] - otherRanges2[i])/Math.Min(otherRanges1[i], otherRanges2[i]));
                                if (error > EPS)
                                {
                                    result.Add(string.Format("Other range: {0} - {1} ({2})", otherRanges1[i], otherRanges2[i], error));
                                }

                                error = Math.Abs((targetRanges1[i] - targetRanges2[i]) / Math.Min(targetRanges1[i], targetRanges2[i]));
                                if (error > EPS)
                                {
                                    result.Add(string.Format("Target range: {0} - {1} ({2})", targetRanges1[i], targetRanges2[i], error));
                                }
                                

                                ulong[] trajectory1 = parser1.GetDetectorTrajectories(i);
                                ulong[] trajectory2 = parser2.GetDetectorTrajectories(i);
                                if (trajectory1.Length != trajectory2.Length)
                                {
                                    result.Add(string.Format("Length: {0} - {1}",
                                        trajectory1.Length, trajectory2.Length));
                                }
                                else
                                {
                                    for (int j = 0; j < trajectory1.Length; ++j)
                                    {
                                        if (trajectory1[j] != trajectory2[j])
                                        {
                                            result.Add(string.Format("[{0}]: {1} - {2}", i, trajectory1[j],
                                                trajectory2[j]));
                                        }
                                    }
                                }
                            }
                        }

                        TimeInfo[] timeScale1 = parser1.GetDetectorTimeScale(i);
                        TimeInfo[] timeScale2 = parser2.GetDetectorTimeScale(i);
                        result.Add(string.Format("Time scale:"));
                        if (timeScale1.Length != timeScale2.Length)
                        {
                            result.Add(string.Format("Time scale length: {0} - {1}",
                                timeScale1.Length, timeScale2.Length));
                        }
                        else
                        {
                            double maxError = 0;
                            for (int j = 0; j < timeScale1.Length; ++j)
                            {
                                if (timeScale1[j].timeStart != timeScale2[j].timeStart)
                                {
                                    result.Add(string.Format("[{0}] time start: {0} - {1}",
                                        j, timeScale1[j].timeStart, timeScale2[j].timeStart));
                                }
                                if (timeScale1[j].timeFinish != timeScale2[j].timeFinish)
                                {
                                    result.Add(string.Format("[{0}] time finish: {0} - {1}",
                                        j, timeScale1[j].timeFinish, timeScale2[j].timeFinish));
                                }
                                if (timeScale1[j].numberOfPhotons != timeScale2[j].numberOfPhotons)
                                {
                                    result.Add(string.Format("[{0}] number of photons: {0} - {1}",
                                        j, timeScale1[j].numberOfPhotons, timeScale2[j].numberOfPhotons));
                                }
                                error = timeScale1[j].weight - timeScale2[j].weight;
                                if (error > maxError)
                                {
                                    maxError = error;
                                }
                                if (error > EPS)
                                {
                                    result.Add(string.Format("[{0}] weight: {1} - {2} ({3})",
                                        j, timeScale1[j].weight, timeScale2[j].weight, error));
                                }
                            }
                            result.Add("Max error " + maxError.ToString());
                        }
                    }
                }

                CompareWindow compareWindow = new CompareWindow(result);
                compareWindow.ShowDialog();
            }
        }

        private void MenuItem_SaveDetectors_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Weight in detectors TXT file|*.txt";

            if (sfd.ShowDialog() == true && parser != null)
            {
                double[] weights = parser.GetDetectorWeights();
                UInt64 numberOfPhotons = parser.GetNumberOfPhotons();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                foreach (double weight in weights)
                {
                    writer.WriteLine(weight / numberOfPhotons);
                }
                writer.Flush();
                writer.Close();    
            }
        }

        private void MenuItem_SaveTimeScales_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Time scales TXT file|*.txt";

            if (sfd.ShowDialog() == true && parser != null)
            {
                ulong numberOfPhotons = parser.GetNumberOfPhotons();
                int numberOfDetectors = parser.GetNumberOfDetectors();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                
                writer.Write("Detector/Time");
                TimeInfo[] timeInfo = parser.GetDetectorTimeScale(0);
                foreach (TimeInfo element in timeInfo)
                {
                    writer.Write('\t');
                    writer.Write(element.timeStart);
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
            }
        }

        private void PaintBoundaries_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Title = "Open boundary file";
            ofd.DefaultExt = "txt";
            ofd.Filter = "Boundary file|*.txt";
            if (ofd.ShowDialog() == true)
            {
                StreamReader reader = new StreamReader(ofd.FileName);
                reader.ReadLine();
                double x1 = double.Parse(reader.ReadLine());
                double x2 = double.Parse(reader.ReadLine());
                double z1 = double.Parse(reader.ReadLine());
                double z2 = double.Parse(reader.ReadLine());
                reader.ReadLine();
                int numberOfSurfaces = int.Parse(reader.ReadLine());
                List<Double3>[] points = new List<Double3>[numberOfSurfaces];
                for (int i = 0; i < numberOfSurfaces; ++i)
                {
                    reader.ReadLine();
                    int numberOfPoints = int.Parse(reader.ReadLine());
                    points[i] = new List<Double3>();
                    for (int j = 0; j < numberOfPoints; ++j)
                    {
                        Double3 point = new Double3(
                            double.Parse(reader.ReadLine()), 
                            0,
                            double.Parse(reader.ReadLine()));
                        points[i].Add(point);
                    }
                }

                double stepX = gridXZ.ActualWidth / (x2 - x1);
                double stepZ = gridXZ.ActualHeight / (z2 - z1);
                for (int i = 0; i < points.Length; ++i)
                {
                    for (int j = 0; j < points[i].Count; ++j)
                    {
                        SolidColorBrush brush = new SolidColorBrush(GetColorByID(i));
                        Rectangle rectangle = new Rectangle();
                        rectangle.Height = 1;
                        rectangle.Width = 1;
                        rectangle.Margin = new Thickness((points[i][j].x - x1) * stepX, (points[i][j].z - z1) * stepZ, 0, 0);
                        rectangle.HorizontalAlignment = System.Windows.HorizontalAlignment.Left;
                        rectangle.VerticalAlignment = System.Windows.VerticalAlignment.Top;
                        rectangle.Fill = brush;
                        gridXZ.Children.Add(rectangle);
                    }
                }
            }
        }

        private Color GetColorByID(int id)
        {
            if (id == 0) return Colors.Red;
            if (id == 1) return Colors.Green;
            if (id == 2) return Colors.Yellow;
            if (id == 3) return Colors.Blue;
            if (id == 4) return Colors.Orange;
            if (id == 5) return Colors.Purple;
            return Colors.Black;
        }

        private void MenuItem_SaveDetectorsWithoutNorma_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Weight in detectors TXT file|*.txt";

            if (sfd.ShowDialog() == true && parser != null)
            {
                double[] weights = parser.GetDetectorWeights();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                foreach (double weight in weights)
                {
                    writer.WriteLine(weight);
                }
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveNumberOfPhotonsPerDetector_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Number of photons per detector TXT file|*.txt";

            if (sfd.ShowDialog() == true && parser != null)
            {
                int numberOfDetectors = parser.GetNumberOfDetectors();
                UInt64[] numberOfPhotonsPerDetector = parser.GetNumberOfPhotonsInDetectorAsArray();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(numberOfPhotonsPerDetector[i]);
                }
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveOtherRangesForDetectors_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Other ranges for detectors TXT file|*.txt";
            if (sfd.ShowDialog() == true && parser != null)
            {
                int numberOfDetectors = parser.GetNumberOfDetectors();
                double[] otherRanges = parser.GetDetectorOtherRanges();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(otherRanges[i]);
                }
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveTargetRangesForDetectors_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Target ranges for detectors TXT file|*.txt";
            if (sfd.ShowDialog() == true && parser != null)
            {
                int numberOfDetectors = parser.GetNumberOfDetectors();
                double[] targetRanges = parser.GetDetectorTargetRanges();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(targetRanges[i]);
                }
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveOtherRangesForDetectorsNormalized_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Other ranges for detectors TXT file|*.txt";
            if (sfd.ShowDialog() == true && parser != null)
            {
                int numberOfDetectors = parser.GetNumberOfDetectors();
                double[] otherRanges = parser.GetDetectorOtherRanges();
                UInt64 numberOfPhotons = parser.GetNumberOfPhotons();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(otherRanges[i] / numberOfPhotons);
                }
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveTargetRangesForDetectorsNormalized_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Target ranges for detectors TXT file|*.txt";
            if (sfd.ShowDialog() == true && parser != null)
            {
                int numberOfDetectors = parser.GetNumberOfDetectors();
                double[] targetRanges = parser.GetDetectorTargetRanges();
                UInt64 numberOfPhotons = parser.GetNumberOfPhotons();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(targetRanges[i] / numberOfPhotons);
                }
                writer.Flush();
                writer.Close();
            }
        }
    }
}
