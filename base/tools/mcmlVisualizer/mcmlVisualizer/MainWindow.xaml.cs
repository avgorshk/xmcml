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
                parser = Parser.getInstance(openFileDialog.FileName);
                if (parser == null)
                {
                    throw new Exception("File is incorrect");
                }
                
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
            double[] targetRanges = parser.GetDetectorTargetRanges();

            UInt64[] numberOfPhotonsPerDetector = parser.GetNumberOfPhotonsInDetectorAsArray();
            UInt64 totalNumberOfPhotonsInDetectors = 0;
            for (int i = 0; i < numberOfPhotonsPerDetector.Length; ++i)
            {
                totalNumberOfPhotonsInDetectors += numberOfPhotonsPerDetector[i];
            }

            textBoxInformation.Text += "Detectors: (" + totalNumberOfPhotonsInDetectors.ToString() +")\n";
            for (int i = 0; i < weights.Length; ++i)
            {
                textBoxInformation.Text += "\tDetector " + i.ToString() + ": " + weights[i] / numberOfPhotons;
                if (i < targetRanges.Length)
                {
                    textBoxInformation.Text += " | " + targetRanges[i] / numberOfPhotons;
                }
                textBoxInformation.Text += " (" + numberOfPhotonsPerDetector[i] + ")" + "\n";
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
            sfd.Filter = "Text file|*.txt";

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
            sfd.Filter = "Text file|*.txt";

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
            sfd.Filter = "Text file|*.txt";

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

        private void MenuItem_SaveDetectors_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Text file|*.txt";

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
            sfd.Filter = "Text file|*.txt";

            if (sfd.ShowDialog() == true && parser != null)
            {
                ulong numberOfPhotons = parser.GetNumberOfPhotons();
                int numberOfDetectors = parser.GetNumberOfDetectors();
                StreamWriter writer = new StreamWriter(sfd.FileName);

                TimeInfo[] timeInfo = parser.GetDetectorTimeScale(0);
                double[,] data = new double[timeInfo.Length, numberOfDetectors];

                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    TimeInfo[] info = parser.GetDetectorTimeScale(i);
                    if (info.Length != timeInfo.Length) throw new Exception();
                    for (int j = 0; j < info.Length; ++j)
                    {
                        data[j, i] = info[j].weight / numberOfPhotons;
                    }
                }

                writer.Write("Time / Detector");
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.Write("\t");
                    writer.Write(i);
                }
                writer.WriteLine();

                for (int i = 0; i < timeInfo.Length; ++i)
                {
                    writer.Write(timeInfo[i].timeFinish);
                    for (int j = 0; j < numberOfDetectors; ++j)
                    {
                        writer.Write("\t");
                        writer.Write(data[i, j]);
                    }
                    writer.WriteLine();
                }

                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveDetectorsWithoutNorma_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Text file|*.txt";

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
            sfd.Filter = "Text file|*.txt";

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
            sfd.Filter = "Text file|*.txt";
            if (sfd.ShowDialog() == true && parser != null)
            {
                int numberOfDetectors = parser.GetNumberOfDetectors();
                double[] weights = parser.GetDetectorWeights();
                double[] targetRanges = parser.GetDetectorTargetRanges();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine(weights[i] - targetRanges[i]);
                }
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveTargetRangesForDetectors_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Text file|*.txt";
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
            sfd.Filter = "Text file|*.txt";
            if (sfd.ShowDialog() == true && parser != null)
            {
                int numberOfDetectors = parser.GetNumberOfDetectors();
                double[] weights = parser.GetDetectorWeights();
                double[] targetRanges = parser.GetDetectorTargetRanges();
                UInt64 numberOfPhotons = parser.GetNumberOfPhotons();
                StreamWriter writer = new StreamWriter(sfd.FileName);
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.WriteLine((weights[i] - targetRanges[i]) / numberOfPhotons);
                }
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveTargetRangesForDetectorsNormalized_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Text file|*.txt";
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

        private void MenuItem_SaveAsText_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Text file|*.txt";
            if (sfd.ShowDialog() == true && parser != null)
            {
                StreamWriter writer = new StreamWriter(sfd.FileName);
                writer.Write(parser.GetText().ToString());
                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveTargetTimeScales_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Text file|*.txt";

            if (sfd.ShowDialog() == true && parser != null)
            {
                ulong numberOfPhotons = parser.GetNumberOfPhotons();
                int numberOfDetectors = parser.GetNumberOfDetectors();
                StreamWriter writer = new StreamWriter(sfd.FileName);

                TimeInfo[] timeInfo = parser.GetDetectorTimeScale(0);
                double[,] data = new double[timeInfo.Length, numberOfDetectors];

                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    TimeInfo[] info = parser.GetDetectorTimeScale(i);
                    if (info.Length != timeInfo.Length) throw new Exception();
                    for (int j = 0; j < info.Length; ++j)
                    {
                        data[j, i] = info[j].targetWeight / numberOfPhotons;
                    }
                }

                writer.Write("Time / Detector");
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.Write("\t");
                    writer.Write(i);
                }
                writer.WriteLine();

                for (int i = 0; i < timeInfo.Length; ++i)
                {
                    writer.Write(timeInfo[i].timeFinish);
                    for (int j = 0; j < numberOfDetectors; ++j)
                    {
                        writer.Write("\t");
                        writer.Write(data[i, j]);
                    }
                    writer.WriteLine();
                }

                writer.Flush();
                writer.Close();
            }
        }

        private void MenuItem_SaveOtherTimeScales_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.DefaultExt = "txt";
            sfd.Filter = "Text file|*.txt";

            if (sfd.ShowDialog() == true && parser != null)
            {
                ulong numberOfPhotons = parser.GetNumberOfPhotons();
                int numberOfDetectors = parser.GetNumberOfDetectors();
                StreamWriter writer = new StreamWriter(sfd.FileName);

                TimeInfo[] timeInfo = parser.GetDetectorTimeScale(0);
                double[,] data = new double[timeInfo.Length, numberOfDetectors];

                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    TimeInfo[] info = parser.GetDetectorTimeScale(i);
                    if (info.Length != timeInfo.Length) throw new Exception();
                    for (int j = 0; j < info.Length; ++j)
                    {
                        data[j, i] = (info[j].weight - info[j].targetWeight) / numberOfPhotons;
                    }
                }

                writer.Write("Time / Detector");
                for (int i = 0; i < numberOfDetectors; ++i)
                {
                    writer.Write("\t");
                    writer.Write(i);
                }
                writer.WriteLine();

                for (int i = 0; i < timeInfo.Length; ++i)
                {
                    writer.Write(timeInfo[i].timeFinish);
                    for (int j = 0; j < numberOfDetectors; ++j)
                    {
                        writer.Write("\t");
                        writer.Write(data[i, j]);
                    }
                    writer.WriteLine();
                }

                writer.Flush();
                writer.Close();
            }
        }
    }
}
