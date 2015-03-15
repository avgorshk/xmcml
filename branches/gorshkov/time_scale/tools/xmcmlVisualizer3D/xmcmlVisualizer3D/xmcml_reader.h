#ifndef __XMCML_READER_H
#define __XMCML_READER_H

#include "xmcml_detector.h"
#include "xmcml_surface.h"

int ReadArea(const char* file_name, area* area_info);
int ReadNumberOfDetectors(const char* file_name, int* number_of_detectors);
int ReadDetectors(const char* file_name, area* area_info, detector* detector_info, int number_of_detectors);
int ReadNumberOfSurfaces(const char* file_name, int* number_of_surfaces);
int ReadSurfaces(const char* file_name, surface* surfaces, int number_of_surfaces);

#endif //__XMCML_READER_H
