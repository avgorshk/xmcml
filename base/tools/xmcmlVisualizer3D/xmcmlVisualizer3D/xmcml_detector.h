#ifndef __XMCML_DETECTOR_H
#define __XMCML_DETECTOR_H

#include "xmcml_trajectory_map.h"

typedef struct __detector
{
    float distance;
    float time_start;
    float time_finish;
    float probing_power;
    uint64 number_of_photons;
    int number_of_maps;
    trajectory_map* map;
} detector;

#endif //__XMCML_DETECTOR_H