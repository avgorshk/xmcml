#ifndef __XMCML_DEMO_H
#define __XMCML_DEMO_H

#include "..\base\example\base_demo.h"

#include "xmcml_detector.h"
#include "xmcml_surface.h"
#include "xmcml_trajectory_map_3d.h"

class xmcml_demo : public base_demo
{
    private:
        char* data_file_name_message;
        char* geometry_file_name_message;
        char* detector_id_message;
        char* probing_power_message;
        char* time_message;

        bool is_no_errors;
        bool is_geometry_enabled;

        area area_info;
        detector* detector_info;
        int number_of_detectors;

        surface* surfaces;
        int number_of_surfaces;

        int detector_id;
        int map_id;

        ui_int_control* detector_id_control;
        ui_float_control_lin* animation_speed_control;

        float total_time;
        float animation_speed;

        trajectory_map_3d* map_3d;

    public:
        xmcml_demo(int argc, char** argv, const char* demo_name);
        virtual ~xmcml_demo();

        virtual	void	init();
		virtual void	update( float dt );
		virtual void	draw3d( float dt );
		virtual void	draw2d( float dt );
};

#endif //__XMCML_DEMO_H