#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <omp.h>

#include "xmcml_surface.h"

surface::surface()
{
    number_of_vertices = 0;
    vertices = NULL;
}

surface::~surface()
{
    if (vertices != NULL)
    {
        delete[] vertices;
    }
}

void surface::set_vertices(float* vertices, int number_of_vertices)
{
    this->vertices = new float[3 * number_of_vertices];
    this->number_of_vertices = number_of_vertices;
    for (int i = 0; i < number_of_vertices; ++i)
    {
        this->vertices[3 * i + 0] = vertices[3 * i + 0];
        this->vertices[3 * i + 1] = vertices[3 * i + 1];
        this->vertices[3 * i + 2] = -vertices[3 * i + 2];
    }
}

float surface::get_max_abs_vertex()
{
    float max = 0.0f;
    for (int i = 0; i < 3 * number_of_vertices; ++i)
    {
        if (fabs(vertices[i]) > max)
        {
            max = fabs(vertices[i]);
        }
    }
    return max;
}

void surface::scale(float coeff)
{
    for (int i = 0; i < 3 * number_of_vertices; ++i)
    {
        vertices[i] /= coeff;
    }
}

void surface::translate(float x, float y, float z)
{
    for (int i = 0; i < number_of_vertices; ++i)
    {
        vertices[3 * i] += x;
        vertices[3 * i + 1] += y;
        vertices[3 * i + 2] += z;
    }
}

void surface::draw(mpoint view_point, float alpha)
{
    mcolor color = color.kWhite;
	color.a = 0.1f;

	if (prev_view_point != view_point)
	{
		this->sort(view_point);
		prev_view_point = view_point;
	}

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor4fv(color.v);

	glVertexPointer(3, GL_FLOAT, 0, vertices);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_TRIANGLES, 0, number_of_vertices);
	glDisableClientState(GL_VERTEX_ARRAY);
}

mpoint surface::get_triangle_mass_center(float* vertex)
{
    mpoint res((vertex[0] + vertex[3] + vertex[6]) / 3.0f, 
        (vertex[1] + vertex[4] + vertex[7]) / 3.0f, 
        (vertex[2] + vertex[5] + vertex[8]) / 3.0f);

    return res;
}

void surface::sort(mpoint view_point)
{
	int number_of_triangles = number_of_vertices / 3;
	float* weights = new float[number_of_triangles];
	int* indices = new int[number_of_triangles];
	int* merged_indices = new int[number_of_triangles];

	int num_threads = 2;
	int portion_size = number_of_triangles / num_threads;
	int remainder = number_of_triangles - portion_size * num_threads;
	
	int* portion_sizes = new int[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		portion_sizes[i] = portion_size;
		if (i < remainder) ++portion_sizes[i];
	}

	omp_set_num_threads(num_threads);
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int start = 0;
		for (int i = 0; i < tid; ++i)
			start += portion_sizes[i];

		for (int i = start; i < start + portion_sizes[tid]; ++i)
		{
			weights[i] = view_point.distance(get_triangle_mass_center(vertices + 9 * i));
		}

		for (int i = start; i < start + portion_sizes[tid]; ++i)
		{
			indices[i] = i;
		}

		surface::qsort(weights, indices, start, start + portion_sizes[tid] - 1);
	}	

	int pos1 = 0;
	int pos2 = portion_sizes[0];
	int pos3 = 0;
	while (pos1 < portion_sizes[0] && pos2 < number_of_triangles)
	{
		if (weights[pos1] > weights[pos2])
		{
			merged_indices[pos3] = indices[pos1];
			++pos3; ++pos1;
		}
		else
		{
			merged_indices[pos3] = indices[pos2];
			++pos3; ++pos2;
		}
	}

	while (pos1 < portion_sizes[0])
	{
		merged_indices[pos3] = indices[pos1];
		++pos1; ++pos3;
	}

	while (pos2 < number_of_triangles)
	{
		merged_indices[pos3] = indices[pos2];
		++pos2; ++pos3;
	}

	float* sorted_vertices = new float[3 * number_of_vertices];
	#pragma omp parallel for
	for (int i = 0; i < number_of_triangles; ++i)
	{
		memcpy(sorted_vertices + 9 * i, vertices + 9 * merged_indices[i], 
			9 * sizeof(float));
	}

	delete[] vertices;
	vertices = sorted_vertices;

	delete[] weights;
	delete[] indices;
	delete[] merged_indices;
}

void surface::qsort(float* weights, int* indices, int low, int high)
{
    float tmp_weight;
    int tmp_index;

    int l = low;
    int h = high;
    float x = weights[(low + high) / 2];
    do
    {
        while (weights[l] > x) ++l;
        while (weights[h] < x) --h;
        if (l <= h)
        {
            tmp_weight = weights[l];
            weights[l] = weights[h];
            weights[h] = tmp_weight;
            tmp_index = indices[l];
            indices[l] = indices[h];
            indices[h] = tmp_index;
            ++l;
            --h;
        }
    } while (l <= h);

    if (low < h) qsort(weights, indices, low, h);
    if (l < high) qsort(weights, indices, l, high);
}

int surface::shell_increment(int inc[], int size)
{
	int p1, p2, p3, s;

	p1 = p2 = p3 = 1;
	s = -1;
	do {
		if (++s % 2) {
			inc[s] = 8 * p1 - 6 * p2 + 1;
		} else {
			inc[s] = 9 * p1 - 9 * p3 + 1;
			p2 *= 2;
			p3 *= 2;
		}
		p1 *= 2;
	} while(3 * inc[s] < size);

	return s > 0 ? --s : 0;
}

void surface::shell_sort(float* weights, int* indices, int size)
{
	int inc, i, j, seq[40];
	int s;
	float tmp_weight;
	int tmp_index;

	s = surface::shell_increment(seq, size);
	while (s >= 0) 
	{
		inc = seq[s--];
		for (i = inc; i < size; i++) 
		{
			float tmp_weight = weights[i];
			float tmp_index = indices[i];
			for (j = i - inc; (j >= 0) && (weights[j] < tmp_weight); j -= inc)
			{
				weights[j + inc] = weights[j];
				indices[j + inc] = indices[j];
			}
			weights[j + inc] = tmp_weight;
			indices[j + inc] = tmp_index;
		}
	}
}