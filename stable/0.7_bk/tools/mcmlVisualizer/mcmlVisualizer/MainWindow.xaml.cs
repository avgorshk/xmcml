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

                int numberOfDetectors = parser.GetDetectors().Length;
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

            textBoxInformation.Text += "Detectors:\n";
            Detector[] detectors = parser.GetDetectors();
            for (int i = 0; i < detectors.Length; ++i)
            {
                ulong numberOfPhotonsPerDetector = parser.GetNumberOfPhotonsInDetector(i);
                textBoxInformation.Text += "\tDetector " + i.ToString() + ": " + 
                    detectors[i].weight / numberOfPhotons + " (" + numberOfPhotonsPerDetector + ")" + "\n";
            }

            StreamWriter writer = new StreamWriter("detectors.txt");
            foreach (Detector detector in detectors)
            {
                writer.WriteLine(detector.weight / numberOfPhotons);
            }
            writer.Flush();
            writer.Close();
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
            double width = height;
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
            double width = height;
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
            double width = height;
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
                        stream.Write(sectionXZ[index]);
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
                    for (int i = 0; i < absorption1.Length; ++i)
                    {
                        error = Math.Abs((absorption1[i] - absorption2[i]) / 
                            Math.Min(absorption1[i], absorption2[i]));
                        if (error > EPS)
                        {
                            result.Add(string.Format("[{0}]: {1} - {2} ({3})", i, absorption1[i],
                                absorption2[i], error));
                        }
                    }
                }

                result.Add("Detectors:");

                Detector[] detectors1 = parser1.GetDetectors();
                Detector[] detectors2 = parser2.GetDetectors();
                if (detectors1.Length != detectors2.Length)
                {
                    result.Add(string.Format("Number of detectors: {0} - {1}", detectors1.Length, 
                        detectors2.Length));
                }
                else
                {
                    for (int i = 0; i < detectors1.Length; ++i)
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
                            error = Math.Abs((detectors1[i].weight - detectors2[i].weight) /
                                Math.Min(detectors1[i].weight, detectors2[i].weight));
                            if (error > EPS)
                            {
                                result.Add(string.Format("Weight: {0} - {1} ({2})", detectors1[i].weight,
                                    detectors2[i].weight, error));
                            }
                            else
                            {
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
                    }
                }

                CompareWindow compareWindow = new CompareWindow(result);
                compareWindow.ShowDialog();
            }
        }
    }
}
