#ifndef __XMCML_SURFACE_H
#define __XMCML_SURFACE_H

#include "..\base\example\math\math.h"

#include "..\base\include\GL\glew.h"
#include "..\base\include\GL\glut.h"

class surface
{
    private:
        float* vertices;
        int number_of_vertices;
		mpoint prev_view_point;

    public:
        surface();
        ~surface();

    public:
        void set_vertices(float* vertices, int number_of_vertices);
        float get_max_abs_vertex();
        void scale(float coeff);
        void translate(float x, float y, float z);
        void draw(mpoint view_point, float alpha);
        mpoint get_triangle_mass_center(float* vertex);

	private:
		void sort(mpoint view_point);
		static void qsort(float* weights, int* indices, int low, int high);
		static int shell_increment(int inc[], int size);
		static void shell_sort(float* weights, int* indices, int size);
};

#endif //__XMCML_SURFACE_H
