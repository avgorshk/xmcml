#include "xmcml_demo.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "xmcml_reader.h"

#define MAX_STR_LENGTH 256

xmcml_demo::xmcml_demo(int argc, char** argv, const char* demo_name) : 
    base_demo(argc, argv, demo_name) 
{
    is_no_errors = true;
    is_geometry_enabled = false;

    data_file_name_message = new char[MAX_STR_LENGTH];
    geometry_file_name_message = new char[MAX_STR_LENGTH];
    detector_id_message = new char[MAX_STR_LENGTH];
    probing_power_message = new char[MAX_STR_LENGTH];
    time_message = new char[MAX_STR_LENGTH];

    detector_info = NULL;
    number_of_detectors = 0;

    surfaces = NULL;
    number_of_surfaces = 0;

    detector_id = 0;
    map_id = 0;

    total_time = 0.0f;
    animation_speed = 0.1f;

    map_3d = NULL;

    detector_id_control = NULL;
    animation_speed_control = NULL;
}

xmcml_demo::~xmcml_demo() 
{
    if (data_file_name_message != NULL)
        delete[] data_file_name_message;

    if (geometry_file_name_message != NULL)
        delete[] geometry_file_name_message;

    if (detector_id_message != NULL)
        delete[] detector_id_message;

    if (probing_power_message != NULL)
        delete[] probing_power_message;

    if (time_message != NULL)
        delete[] time_message;

    if (detector_id_control != NULL)
        delete detector_id_control;

    if (animation_speed_control != NULL)
        delete animation_speed_control;
    
    if (detector_info != NULL)
    {
        for (int i = 0; i < number_of_detectors; ++i)
        {
            if (detector_info[i].map != NULL)
            {
                delete[] detector_info[i].map;
            }
        }
        delete[] detector_info;
    }

    if (surfaces != NULL)
    {
        delete[] surfaces;
    }

    if (map_3d != NULL)
    {
        delete map_3d;
    }
}

void xmcml_demo::init()
{
    //Parse command line
    static const char* data_file_name = arg_get_value("-data", "file not found");
    sprintf(data_file_name_message, "Data file: %s", data_file_name);
    if (strcmp(data_file_name, "file not found") == 0)
    {
        is_no_errors = false;
    }

    static const char* geometry_file_name = arg_get_value("-geometry", "file not found");
    sprintf(geometry_file_name_message, "Geometry file: %s", geometry_file_name);
    if (strcmp(geometry_file_name, "file not found") != 0)
    {
        is_geometry_enabled = true;
    }

    static const char* detector_id_cmd_str = arg_get_value("-detector", "0");
    int detector_id_cmd = atoi(detector_id_cmd_str);
    
    //Read *.mcml.out file
    if (is_no_errors)
    {
        if (ReadArea(data_file_name, &area_info) != 0)
        {
            is_no_errors = false;
            sprintf(data_file_name_message, "Data file: %s", "error while reading");
            return;
        }

        if (ReadNumberOfDetectors(data_file_name, &number_of_detectors) != 0)
        {
            is_no_errors = false;
            sprintf(data_file_name_message, "Data file: %s", "error while reading");
            return;
        }

        detector_info = new detector[number_of_detectors];
        if (ReadDetectors(data_file_name, &area_info, detector_info, number_of_detectors) != 0)
        {
            is_no_errors = false;
            sprintf(data_file_name_message, "Data file: %s", "error while reading");
            return;
        }

        if (detector_id_cmd >= 0 && detector_id_cmd < number_of_detectors)
        {
            detector_id = detector_id_cmd;
        }
    }

    //Process data
    if (is_no_errors)
    {
        int size = area_info.size.x * area_info.size.y * area_info.size.z;
        float* buffer = new float[size];
        for (int i = 0; i < number_of_detectors; ++i)
        {
            int number_of_maps = detector_info[i].number_of_maps;
            
            for (int k = 0; k < size; ++k)
            {
                buffer[k] = detector_info[i].map[0].get(k);
            }

            for (int j = 1; j < number_of_maps; ++j)
            {
                for (int k = 0; k < size; ++k)
                {
                    buffer[k] += detector_info[i].map[j].get(k);
                }
                detector_info[i].map[j].set(area_info.size, buffer);
            }
        }

        for (int i = 0; i < number_of_detectors; ++i)
        {
            int number_of_maps = detector_info[i].number_of_maps;
            float number_of_photons = (float)(detector_info[i].number_of_photons);
            for (int j = 0; j < number_of_maps; ++j)
            {
                for (int k = 0; k < size; ++k)
                {
                    buffer[k] = detector_info[i].map[j].get(k);
                    if (buffer[k] < 1.0f) buffer[k] = 1.0f;
                    buffer[k] = log10(buffer[k]) / log10(number_of_photons);
                }
                detector_info[i].map[j].set(area_info.size, buffer);
            }
        }

        delete[] buffer;
    }

    //Read *.surface file
    if (is_geometry_enabled)
    {
        if (ReadNumberOfSurfaces(geometry_file_name, &number_of_surfaces) != 0)
        {
            is_geometry_enabled = false;
			sprintf(geometry_file_name_message, "Geometry file: %s", "error while reading");
        }

        surfaces = new surface[number_of_surfaces];
        if (ReadSurfaces(geometry_file_name, surfaces, number_of_surfaces) != 0)
        {
            is_geometry_enabled = false;
            sprintf(geometry_file_name_message, "Geometry file: %s", "error while reading");
        }
    }

    //Set controls
    if (is_no_errors)
    {
        detector_id_control = new ui_int_control("Detector ID", &detector_id, 1, 0, number_of_detectors - 1);
        add_control(detector_id_control);
        animation_speed_control = new ui_float_control_lin("Animation speed", &animation_speed, 0.1f, 0.1f, 0.9f);
        add_control(animation_speed_control);
    }

    //Init 3D trajectory map
    if (is_no_errors)
    {
        map_3d = new trajectory_map_3d(&area_info, &(detector_info[detector_id].map[map_id]));
    }

    //Set camera position
    base_demo::g_view_dist = 220.0f;
	base_demo::g_view_phi = -40.0f;
	base_demo::g_view_theta = -20.0f;

    //Set OpenGL options
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SMOOTH);
}

void xmcml_demo::update(float dt)
{
    if (is_no_errors)
    {
        total_time += dt;
        if (total_time > (1.0f - animation_speed))
        {
            ++map_id;
            if (map_id >= detector_info[detector_id].number_of_maps)
                map_id = 0;
            total_time = 0.0f;
            map_3d->update(&(detector_info[detector_id].map[map_id]));
        } 
    }
}

void xmcml_demo::draw2d(float dt)
{
    draw_string(char_width * 2, char_height * 12, data_file_name_message);
    draw_string(char_width * 2, char_height * 13, geometry_file_name_message);

    if (is_no_errors)
    {
        sprintf(detector_id_message, "Detector: %d", detector_id);
        draw_string(char_width * 2, char_height * 14, detector_id_message, 0, 1, 0);

        sprintf(probing_power_message, "Probing power: %.5e", detector_info[detector_id].probing_power);
        draw_string(char_width * 2, char_height * 15, probing_power_message, 0, 1, 0);

        float time = detector_info[detector_id].time_start + map_id * 
            (detector_info[detector_id].time_finish - detector_info[detector_id].time_start) / 
            detector_info[detector_id].number_of_maps;
        time /= 300.0f;
        sprintf(time_message, "Time: %.2f ps", time, 1, 1, 0);

        draw_string(char_width * 2, char_height * 10, time_message, 1, 1, 0);
    }
}

void xmcml_demo::draw3d(float dt)
{
	//Draw 3D trajectory map
    if (is_no_errors)
    {
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
        map_3d->draw(base_demo::g_view_pos);
    }

	//Draw geometry
    if (is_geometry_enabled)
    {
		for (int i = 0; i < number_of_surfaces; ++i)
        {
            surfaces[i].draw(base_demo::g_view_pos, 0.1f * i + 0.01f);
        }
    }    
}