#ifndef __XMCML_TRAJECTORY_MAP_3D_H
#define __XMCML_TRAJECTORY_MAP_3D_H

#include "xmcml_trajectory_map.h"
#include "xmcml_demo_types.h"
#include "..\base\example\math\math.h"

#include "..\base\include\GL\glew.h"
#include "..\base\include\GL\glut.h"

class trajectory_map_3d
{
    private:
        area* area_info;
        trajectory_map* trajectory_map_info;
        int3* map_index;
        int map_index_size;

    public:
        trajectory_map_3d(area* area_info, trajectory_map* trajectory_map_info);
        ~trajectory_map_3d();
        void update(trajectory_map* trajectory_map_info);

    public:
        void draw(mpoint view_piont);

    private:
        mcolor get_color(int x, int y, int z);
        void set_map_index();
        void sort_map_index(mpoint view_point);
        mpoint get_cell_center(int x, int y, int z);
        void set_cube(int x, int y, int z, float* vertices);

    private:    
        static void qsort(float* weights, int3* indices, int low, int high);
};

#endif //__XMCML_TRAJECTORY_MAP_3D_H