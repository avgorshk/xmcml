using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tao.Platform.Windows;
using Tao.OpenGl;

namespace surfaceVisualizer
{
    class Surface
    {
        private Triangle[] triangle = null;

        public Surface(Triangle[] triangle)
        {
            this.triangle = triangle;
        }

        public void Draw()
        {
            Gl.glPushMatrix();

            Translate();

            float[] material = { 1.0f, 1.0f, 1.0f, 1.0f };
            float[] shininess = { 50.0f };
            Gl.glMaterialfv(Gl.GL_FRONT_AND_BACK, Gl.GL_SPECULAR, material);
            Gl.glMaterialfv(Gl.GL_FRONT_AND_BACK, Gl.GL_DIFFUSE, material);
            Gl.glMaterialfv(Gl.GL_FRONT_AND_BACK, Gl.GL_AMBIENT, material);
            Gl.glMaterialfv(Gl.GL_FRONT_AND_BACK, Gl.GL_SHININESS, shininess);
            
            Gl.glBegin(Gl.GL_TRIANGLES);

            if (triangle != null)
            {
                foreach (Triangle t in triangle)
                {
                    float3 normal = t.GetNormal();
                    Gl.glVertex3f(t.a.x, t.a.y, t.a.z);
                    Gl.glNormal3f(normal.x, normal.y, normal.z);
                    Gl.glVertex3f(t.b.x, t.b.y, t.b.z);
                    Gl.glNormal3f(normal.x, normal.y, normal.z);
                    Gl.glVertex3f(t.c.x, t.c.y, t.c.z);
                    Gl.glNormal3f(normal.x, normal.y, normal.z);
                }
            }

            Gl.glEnd();

            Gl.glPopMatrix();
        }

        private void Translate()
        {
            float3 min = triangle[0].a;
            float3 max = triangle[0].a;

            foreach (Triangle t in triangle)
            {
                min = t.GetMin(min);
                max = t.GetMax(max);
            }

            float3 center = new float3();
            center.x = (min.x + max.x) / 2.0f;
            center.y = (min.y + max.y) / 2.0f;
            center.z = (min.z + max.z) / 2.0f;

            float scale = Math.Abs(max.x - min.x);
            if (scale < Math.Abs(max.y - min.y))
                scale = Math.Abs(max.y - min.y);
            if (scale < Math.Abs(max.z - min.z))
                scale = Math.Abs(max.z - min.z);
            scale = 3.0f / scale;

            Gl.glScalef(scale, scale, scale);

            Gl.glTranslatef(-center.x, -center.y, -center.z);
        }
    }
}
