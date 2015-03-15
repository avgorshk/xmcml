using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Tao.OpenGl;
using Tao.Platform.Windows;

namespace surfaceVisualizer
{
    public partial class MainForm : Form
    {
        private Surface surface = null;

        private float angleY = 0.0f;
        private float angleX = 0.0f;
        private float zoom = 6.0f;

        //Инициализация библитеки OpenGL
        private void InitGL()
        {
            //Инициализация GL 
            Gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            Gl.glShadeModel(Gl.GL_SMOOTH);

            Gl.glEnable(Gl.GL_DEPTH_TEST);
            Gl.glEnable(Gl.GL_LIGHTING);
            Gl.glEnable(Gl.GL_LIGHT0);
            Gl.glEnable(Gl.GL_NORMALIZE);

            //Настройка источника света по умолчанию
            float[] light0DiffuseSpecular = { 1.0f, 1.0f, 1.0f };
            float[] lightModelAmbient = { 0.2f, 0.2f, 0.2f, 1.0f };
            float[] light0Position = { 50.0f, 50.0f, 50.0f, 0.0f }; //
            Gl.glLightfv(Gl.GL_LIGHT0, Gl.GL_DIFFUSE, light0DiffuseSpecular);
            Gl.glLightfv(Gl.GL_LIGHT0, Gl.GL_SPECULAR, light0DiffuseSpecular);
            Gl.glLightModelfv(Gl.GL_LIGHT_MODEL_AMBIENT, lightModelAmbient);
            Gl.glLightModeli(Gl.GL_LIGHT_MODEL_LOCAL_VIEWER, Gl.GL_TRUE);
            Gl.glLightModeli(Gl.GL_LIGHT_MODEL_TWO_SIDE, Gl.GL_TRUE);
            Gl.glLightfv(Gl.GL_LIGHT0, Gl.GL_POSITION, light0Position); //

        }

        //Функция, вызываемая при изменении размеров окна отрисовки
        private void ResizeGL()
        {
            Gl.glViewport(0, 0, OGL.Width, OGL.Height);
            Gl.glMatrixMode(Gl.GL_PROJECTION);
            Gl.glLoadIdentity();
            Glu.gluPerspective(60.0, (double)OGL.Width / (double)OGL.Height,
                0.001, 1000.0);
            Gl.glMatrixMode(Gl.GL_MODELVIEW);
            Gl.glLoadIdentity();
        }

        //Функция для отрисовки сцены
        private void DrawGL()
        {
            //Очистка цвета и буфера глубины 
            Gl.glClear(Gl.GL_COLOR_BUFFER_BIT | Gl.GL_DEPTH_BUFFER_BIT);

            //Управление положением камеры
            Gl.glLoadIdentity();
            Glu.gluLookAt(0.0, 0.0, zoom, 0.0, 0.0, -100.0, 0.0, 1.0, 0.0);
            Gl.glRotatef(angleY, 0.0f, 1.0f, 0.0f);
            Gl.glRotatef(angleX, 1.0f, 0.0f, 0.0f);

            //Основная отрисовка
            if (surface != null)
                surface.Draw();

            Gl.glFlush();
        }

        public MainForm()
        {
            InitializeComponent();
            OGL.InitializeContexts();
        }

        private void closeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            InitGL();
            ResizeGL();
        }

        private void OGL_Paint(object sender, PaintEventArgs e)
        {
            DrawGL();
        }

        private void OGL_Resize(object sender, EventArgs e)
        {
            ResizeGL();
        }

        private void opentxtFileToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.DefaultExt = "txt";
            dialog.Filter = "Surface text file (*.txt)|*.txt";
            DialogResult dialogResult = dialog.ShowDialog();
            if (dialogResult == System.Windows.Forms.DialogResult.OK)
            {
                TxtReader txtReader = new TxtReader(dialog.FileName);
                surface = new Surface(txtReader.triangle);
                OGL.Refresh();
            }
        }

        private void OGL_KeyPress(object sender, KeyPressEventArgs e)
        {
            switch (e.KeyChar)
            {
                case 'd':
                    angleY -= 5.0f;
                    OGL.Refresh();
                    break;
                case 'a':
                    angleY += 5.0f;
                    OGL.Refresh();
                    break;
                case 's':
                    angleX -= 5.0f;
                    OGL.Refresh();
                    break;
                case 'w':
                    angleX += 5.0f;
                    OGL.Refresh();
                    break;
                case 'e':
                    zoom -= 0.5f;
                    OGL.Refresh();
                    break;
                case 'q':
                    zoom += 0.5f;
                    OGL.Refresh();
                    break;
                default:
                    break;
            }
        }

        private void opensurfaceFileToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.DefaultExt = "surface";
            dialog.Filter = "Surface binary file (*.surface)|*.surface";
            DialogResult dialogResult = dialog.ShowDialog();
            if (dialogResult == System.Windows.Forms.DialogResult.OK)
            {
                SurfaceReader reader = new SurfaceReader(dialog.FileName);
                surface = reader.surface[0];
                OGL.Refresh();
            }
        }
    }
}
