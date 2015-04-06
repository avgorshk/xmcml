#ifndef __XMCML_TRAJECTORY_MAP_H
#define __XMCML_TRAJECTORY_MAP_H

#include "xmcml_demo_types.h"

class trajectory_map
{
    private:
        int3 size;
        float* data;

    public:
        trajectory_map();
        trajectory_map(int3 size, float* data);
        trajectory_map(const trajectory_map& copy);
        ~trajectory_map();

        float get(int x, int y, int z);
        float get(int data_index);
        void set(int3 size, float* data);
        int get_full_size();
        int3 get_size();
};

#endif //__XMCML_TRAJECTORY_MAP_H