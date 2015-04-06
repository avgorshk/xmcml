#include "xmcml_trajectory_map_3d.h"

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

trajectory_map_3d::trajectory_map_3d(area* area_info, trajectory_map* trajectory_map_info)
{
    this->area_info = new area;
    this->area_info->corner = area_info->corner;
    this->area_info->length = area_info->length;
    this->area_info->size = area_info->size;
    this->area_info->corner.z = -this->area_info->corner.z;
    this->area_info->length.z = -this->area_info->length.z;

    this->trajectory_map_info = new trajectory_map(*trajectory_map_info);
    this->set_map_index();
}

trajectory_map_3d::~trajectory_map_3d()
{
    delete area_info;
    delete trajectory_map_info;
}

void trajectory_map_3d::draw(mpoint view_point)
{
	mcolor color;
	float* colors = new float[map_index_size * 24 * 4];
	float* vertices = new float[map_index_size * 24 * 3];
    
    sort_map_index(view_point);

	//Set colors and vertices arrays
	#pragma omp parallel for private(color)
	for (int i = 0; i < map_index_size; ++i)
    {
        color = get_color(map_index[i].x, map_index[i].y, map_index[i].z);
		color.a = 0.1f;
        for (int j = 0; j < 24; ++j)
		{
			memcpy(colors + 24 * 4 * i + 4 * j, color.v, 4 * sizeof(float));
		}

        set_cube(map_index[i].x, map_index[i].y, map_index[i].z, vertices + i * 24 * 3);
    }

    //Draw fill cubes
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glVertexPointer(3, GL_FLOAT, 0, vertices);
	glColorPointer(4, GL_FLOAT, 0, colors);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_QUADS, 0, map_index_size * 24);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	delete[] colors;
	delete[] vertices;
}

void trajectory_map_3d::update(trajectory_map* trajectory_map_info)
{
    delete this->trajectory_map_info;
    this->trajectory_map_info = new trajectory_map(*trajectory_map_info);
    this->set_map_index();
}

mcolor trajectory_map_3d::get_color(int x, int y, int z)
{
    mcolor color;
    float intensity = trajectory_map_info->get(x, y, z);
    
    if (intensity < 0.0f) intensity = 0.0f;
    if (intensity > 1.0f) intensity = 1.0f;

    if (intensity < 1.0f / 3.0f)
    {
        intensity *= 3.0f;
        color.r = intensity;
        color.g = 0.0f;
        color.b = 0.0f;
    }
    else if (intensity < 2.0f / 3.0f)
    {
        intensity = 3.0f * intensity - 1.0f;
        color.r = 1.0f;
        color.g = intensity;
        color.b = 0.0f;
    }
    else
    {
        intensity = 3.0f * intensity - 2.0f;
        color.r = 1.0f;
        color.g = 1.0f;
        color.b = intensity;
    }

    return color;
}

void trajectory_map_3d::set_map_index()
{
    map_index_size = 0;
    int map_size = trajectory_map_info->get_full_size();
    int3 map_size3 = trajectory_map_info->get_size();
    float intensity;

    for (int i = 0; i < map_size; ++i)
    {
        intensity = trajectory_map_info->get(i);
        if (intensity > 0.0f) ++map_index_size;
    }

    if (map_index_size > 0)
    {
        map_index = new int3[map_index_size];

        int j = 0;
        for (int x = 0; x < map_size3.x; ++x)
        {
            for (int y = 0; y < map_size3.y; ++y)
            {
                for (int z = 0; z < map_size3.z; ++z)
                {
                    intensity = trajectory_map_info->get(x, y, z);
                    if (intensity > 0.0f)
                    {
                        map_index[j].x = x;
                        map_index[j].y = y;
                        map_index[j].z = z;
                        ++j;
                    }
                }
            }
        }
    }
    else 
    {
        map_index = NULL;
    }
}

void trajectory_map_3d::sort_map_index(mpoint view_point)
{
    float* distances = new float[map_index_size];
    mpoint center;

    for (int i = 0; i < map_index_size; ++i)
    {
        center = get_cell_center(map_index[i].x, map_index[i].y, map_index[i].z);
        distances[i] = (view_point - center).length();
    }

    qsort(distances, map_index, 0, map_index_size - 1);

    delete[] distances;
}

mpoint trajectory_map_3d::get_cell_center(int x, int y, int z)
{
    float cx, cy, cz;

    cx = area_info->corner.x + x * (area_info->length.x / area_info->size.x) + 
        0.5f * (area_info->length.x / area_info->size.x);
    cy = area_info->corner.y + y * (area_info->length.y / area_info->size.y) + 
        0.5f * (area_info->length.y / area_info->size.y);
    cz = area_info->corner.z + z * (area_info->length.z / area_info->size.z) + 
        0.5f * (area_info->length.z / area_info->size.z);

    return mpoint(cx, cy, cz);
}

void trajectory_map_3d::qsort(float* weights, int3* indices, int low, int high)
{
    float tmp_weight;
    int3 tmp_index;

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

void trajectory_map_3d::set_cube(int x, int y, int z, float* vertices)
{
    mpoint vertex[8];
    float sx = area_info->length.x / area_info->size.x;
    float sy = area_info->length.y / area_info->size.y;
    float sz = area_info->length.z / area_info->size.z;

    vertex[0].x = area_info->corner.x + x * sx;
    vertex[0].y = area_info->corner.y + y * sy;
    vertex[0].z = area_info->corner.z + z * sz;

    vertex[1].x = area_info->corner.x + x * sx + sx;
    vertex[1].y = area_info->corner.y + y * sy;
    vertex[1].z = area_info->corner.z + z * sz;

    vertex[2].x = area_info->corner.x + x * sx + sx;
    vertex[2].y = area_info->corner.y + y * sy + sy;
    vertex[2].z = area_info->corner.z + z * sz;

    vertex[3].x = area_info->corner.x + x * sx;
    vertex[3].y = area_info->corner.y + y * sy + sy;
    vertex[3].z = area_info->corner.z + z * sz;

    vertex[4].x = area_info->corner.x + x * sx;
    vertex[4].y = area_info->corner.y + y * sy;
    vertex[4].z = area_info->corner.z + z * sz + sz;

    vertex[5].x = area_info->corner.x + x * sx + sx;
    vertex[5].y = area_info->corner.y + y * sy;
    vertex[5].z = area_info->corner.z + z * sz + sz;

    vertex[6].x = area_info->corner.x + x * sx + sx;
    vertex[6].y = area_info->corner.y + y * sy + sy;
    vertex[6].z = area_info->corner.z + z * sz + sz;

    vertex[7].x = area_info->corner.x + x * sx;
    vertex[7].y = area_info->corner.y + y * sy + sy;
    vertex[7].z = area_info->corner.z + z * sz + sz;

	memcpy(vertices + 0 * 3, vertex[0].v, 3 * sizeof(float));
	memcpy(vertices + 1 * 3, vertex[1].v, 3 * sizeof(float));
	memcpy(vertices + 2 * 3, vertex[2].v, 3 * sizeof(float));
	memcpy(vertices + 3 * 3, vertex[3].v, 3 * sizeof(float));

	memcpy(vertices + 4 * 3, vertex[1].v, 3 * sizeof(float));
	memcpy(vertices + 5 * 3, vertex[5].v, 3 * sizeof(float));
	memcpy(vertices + 6 * 3, vertex[6].v, 3 * sizeof(float));
	memcpy(vertices + 7 * 3, vertex[2].v, 3 * sizeof(float));

	memcpy(vertices +  8 * 3, vertex[5].v, 3 * sizeof(float));
	memcpy(vertices +  9 * 3, vertex[4].v, 3 * sizeof(float));
	memcpy(vertices + 10 * 3, vertex[7].v, 3 * sizeof(float));
	memcpy(vertices + 11 * 3, vertex[6].v, 3 * sizeof(float));

	memcpy(vertices + 12 * 3, vertex[4].v, 3 * sizeof(float));
	memcpy(vertices + 13 * 3, vertex[0].v, 3 * sizeof(float));
	memcpy(vertices + 14 * 3, vertex[3].v, 3 * sizeof(float));
	memcpy(vertices + 15 * 3, vertex[7].v, 3 * sizeof(float));

	memcpy(vertices + 16 * 3, vertex[0].v, 3 * sizeof(float));
	memcpy(vertices + 17 * 3, vertex[1].v, 3 * sizeof(float));
	memcpy(vertices + 18 * 3, vertex[5].v, 3 * sizeof(float));
	memcpy(vertices + 19 * 3, vertex[4].v, 3 * sizeof(float));

	memcpy(vertices + 20 * 3, vertex[3].v, 3 * sizeof(float));
	memcpy(vertices + 21 * 3, vertex[2].v, 3 * sizeof(float));
	memcpy(vertices + 22 * 3, vertex[6].v, 3 * sizeof(float));
	memcpy(vertices + 23 * 3, vertex[7].v, 3 * sizeof(float));
}