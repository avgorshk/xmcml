#include "xmcml_trajectory_map.h"

trajectory_map::trajectory_map() {}

trajectory_map::trajectory_map(int3 size, float* data)
{
    this->size = size;
    int total_size = size.x * size.y * size.z;
    this->data = new float[total_size];
    for (int i = 0; i < total_size; ++i)
    {
        this->data[i] = data[i];
    }
}

trajectory_map::trajectory_map(const trajectory_map& copy)
{
    size = copy.size;
    int total_size = size.x * size.y * size.z;
    data = new float[total_size];
    for (int i = 0; i < total_size; ++i)
    {
        data[i] = copy.data[i];
    }
}

trajectory_map::~trajectory_map()
{
    delete[] data;
}

float trajectory_map::get(int x, int y, int z)
{
    if ((x < 0 || x >= size.x) || (y < 0 || y >= size.y) || (z < 0 || z >= size.z))
    {
        throw;
    }

    return data[x * size.y * size.z + y * size.z + z];
}

float trajectory_map::get(int data_index)
{
    if (data_index < 0 || data_index >= size.x * size.y * size.z)
    {
        throw;
    }

    return data[data_index];
}

void trajectory_map::set(int3 size, float* data)
{
    this->size = size;
    int total_size = size.x * size.y * size.z;
    this->data = new float[total_size];
    for (int i = 0; i < total_size; ++i)
    {
        this->data[i] = data[i];
    }
}

int trajectory_map::get_full_size()
{
    return size.x * size.y * size.z;
}

int3 trajectory_map::get_size()
{
    return size;
}