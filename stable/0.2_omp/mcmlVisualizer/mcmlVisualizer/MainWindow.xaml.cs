﻿using System;
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
        const double MAX_VALUE = 1.0E+12;

        private TrajectoryBox trajectoryBox = null;
        private double scaledCoefficient = 0.0;

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
            openFileDialog.DefaultExt = ".mcml.out";
            openFileDialog.Filter = "MCML Output files (.mcml.out)|*.mcml.out";
            openFileDialog.Title = "Open MCML Output files...";
            openFileDialog.ShowDialog();
        }

        void openFileDialog_FileOk(object sender, System.ComponentModel.CancelEventArgs e)
        {
            try
            {
                OpenFileDialog openFileDialog = sender as OpenFileDialog;
                this.Title = "mcmlVisualizer: " + openFileDialog.FileName;
                Parser parser = new Parser(openFileDialog.FileName);
                
                SetInformation(parser);
                
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

        void SetInformation(Parser parser)
        {
            textBoxInformation.Text = "";

            int numberOfPhotons = parser.GetNumberOfPhotons();
            textBoxInformation.Text += "Number of photons: " + numberOfPhotons + "\n";
            textBoxInformation.Text += "Specular reflectance: " + parser.GetSpecularReflectance() + "\n";

            textBoxInformation.Text += "Detectors:\n";
            Detector[] detectors = parser.GetDetectors();
            for (int i = 0; i < detectors.Length; ++i)
            {
                textBoxInformation.Text += "\tDetector " + i.ToString() + ": " + 
                    detectors[i].weight / numberOfPhotons + "\n";
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
            
            double height = gridXY.ActualHeight;
            double width = height;
            double stepX = width / trajectoryBox.area.partitionNumber.x;
            double stepY = height / trajectoryBox.area.partitionNumber.y;

            for (int ix = 0; ix < trajectoryBox.area.partitionNumber.x; ++ix)
            {
                for (int iy = 0; iy < trajectoryBox.area.partitionNumber.y; ++iy)
                {
                    int index = ix * trajectoryBox.area.partitionNumber.y + iy;
                    double weight = (section[index] / scaledCoefficient) * MAX_VALUE;
                    if (weight < 1.0) 
                        weight = 1.0;
                    
                    SolidColorBrush brush = new SolidColorBrush(
                        GetColor(Math.Log10(weight) / Math.Log10(MAX_VALUE)));
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

            double height = gridXZ.ActualHeight;
            double width = height;
            double stepX = width / trajectoryBox.area.partitionNumber.x;
            double stepZ = height / trajectoryBox.area.partitionNumber.z;

            for (int ix = 0; ix < trajectoryBox.area.partitionNumber.x; ++ix)
            {
                for (int iz = 0; iz < trajectoryBox.area.partitionNumber.z; ++iz)
                {
                    int index = ix * trajectoryBox.area.partitionNumber.z + iz;
                    double weight = (section[index] / scaledCoefficient) * MAX_VALUE;
                    if (weight < 1.0)
                        weight = 1.0;

                    SolidColorBrush brush = new SolidColorBrush(
                        GetColor(Math.Log10(weight) / Math.Log10(MAX_VALUE)));
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

            double height = gridYZ.ActualHeight;
            double width = height;
            double stepY = width / trajectoryBox.area.partitionNumber.y;
            double stepZ = height / trajectoryBox.area.partitionNumber.z;

            for (int iy = 0; iy < trajectoryBox.area.partitionNumber.y; ++iy)
            {
                for (int iz = 0; iz < trajectoryBox.area.partitionNumber.z; ++iz)
                {
                    int index = iy * trajectoryBox.area.partitionNumber.z + iz;
                    double weight = (section[index] / scaledCoefficient) * MAX_VALUE;
                    if (weight < 1.0)
                        weight = 1.0;

                    SolidColorBrush brush = new SolidColorBrush(
                        GetColor(Math.Log10(weight) / Math.Log10(MAX_VALUE)));
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
                return Color.FromRgb(0, (byte)(255 * weight),
                    (byte)(Colors.DarkBlue.B + weight * (Colors.Cyan.B - Colors.DarkBlue.B)));
            }
            else if (weight < 2.0 / 3.0)
            {
                weight = 3 * weight - 1.0;
                return Color.FromRgb((byte)(255 * weight), 255,
                    (byte)(255 - 255 * weight));
            }
            else
            {
                weight = 3 * weight - 2.0;
                return Color.FromRgb(
                    (byte)(Colors.Yellow.R - weight * (Colors.Yellow.R - Colors.DarkRed.R)),
                    (byte)(255 - 255 * weight), 0);
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
    }
}
