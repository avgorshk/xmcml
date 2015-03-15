#include "xmcml_reader.h"

#include <stdio.h>
#include <math.h>

#include "xmcml_sections.h"

int SkipCurrentSection(FILE* file)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    if (fseek(file, section_length, SEEK_CUR) != 0)
        return -1;

    return 0;
}

int ReadArea(const char* file_name, area* area_info)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    double double_buffer[6];
    int int_buffer[3];

    FILE* file = NULL;
    fopen_s(&file, file_name, "rb");
    if (file == NULL)
        return -1;

    SkipCurrentSection(file); //MCML_SECTION_NUMBER_OF_PHOTONS

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_AREA)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(double_buffer, sizeof(double), 6, file);
    if (reading_items < 6)
        return -1;

    area_info->corner.x = (float)(double_buffer[0]);
    area_info->corner.y = (float)(double_buffer[1]);
    area_info->corner.z = (float)(double_buffer[2]);
    area_info->length.x = (float)(double_buffer[3]);
    area_info->length.y = (float)(double_buffer[4]);
    area_info->length.z = (float)(double_buffer[5]);

    reading_items = fread(int_buffer, sizeof(int), 3, file);
    if (reading_items < 3)
        return -1;

    area_info->size.x = int_buffer[0];
    area_info->size.y = int_buffer[1];
    area_info->size.z = int_buffer[2];

    fclose(file);

    return 0;
}

int ReadNumberOfDetectors(const char* file_name, int* number_of_detectors)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;

    FILE* file = NULL;
    fopen_s(&file, file_name, "rb");
    if (file == NULL)
        return -1;

    SkipCurrentSection(file); //MCML_SECTION_NUMBER_OF_PHOTONS
    SkipCurrentSection(file); //MCML_SECTION_AREA

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTORS)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(number_of_detectors, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    fclose(file);

    return 0;
}

int ReadDetectors(const char* file_name, area* area_info, detector* detector_info, int number_of_detectors)
{
    unsigned long long int reading_items;
    int section;
    unsigned int section_length;
    int int_buffer;
    uint64 uint64_buffer;
    double double_buffer[6];
    int total_map_size = area_info->size.x * area_info->size.y * area_info->size.z;
    uint64 total_number_of_photons;
    float float_number_of_photons;

    uint64* uint64_map = new uint64[total_map_size];
    float* float_map = new float[total_map_size];

    FILE* file = NULL;
    fopen_s(&file, file_name, "rb");
    if (file == NULL)
        return -1;

    //MCML_SECTION_NUMBER_OF_PHOTONS
    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_NUMBER_OF_PHOTONS)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&total_number_of_photons, sizeof(uint64), 1, file);
    if (reading_items < 1)
        return -1;
    
    SkipCurrentSection(file); //MCML_SECTION_AREA

    //MCML_SECTION_DETECTORS

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTORS)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&int_buffer, sizeof(int), 1, file);
    if (reading_items < 1 || int_buffer != number_of_detectors)
        return -1;

    for (int i = 0; i < number_of_detectors; ++i)
    {
        reading_items = fread(double_buffer, sizeof(double), 6, file);
        if (reading_items < 6)
            return -1;

        detector_info[i].distance = (float)(double_buffer[0]);
    }

    SkipCurrentSection(file); //MCML_SECTION_SPECULAR_REFLECTANCE
    SkipCurrentSection(file); //MCML_SECTION_COMMON_TRAJECTORIES
    
    //MCML_SECTION_DETECTOR_WEIGHTS

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTOR_WEIGHTS)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&int_buffer, sizeof(int), 1, file);
    if (reading_items < 1 || int_buffer != number_of_detectors)
        return -1;

    for (int i = 0; i < number_of_detectors; ++i)
    {
        reading_items = fread(double_buffer, sizeof(double), 1, file);
        if (reading_items < 1)
            return -1;

        detector_info[i].probing_power = (float)((double)(double_buffer[0]) / total_number_of_photons);
    }

    //MCML_SECTION_DETECTOR_TRAJECTORIES

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTOR_TRAJECTORIES)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&int_buffer, sizeof(int), 1, file);
    if (reading_items < 1 || int_buffer != number_of_detectors)
        return -1;

    for (int i = 0; i < number_of_detectors; ++i)
    {
        reading_items = fread(&uint64_buffer, sizeof(uint64), 1, file);
        if (reading_items < 1)
            return -1;

        detector_info[i].number_of_photons = uint64_buffer;

        reading_items = fread(&int_buffer, sizeof(int), 1, file);
        if (reading_items < 1 || int_buffer != total_map_size)
            return -1;

        reading_items = fread(uint64_map, sizeof(uint64), total_map_size, file);
        if (reading_items < total_map_size)
            return -1;
    }

    //MCML_SECTION_DETECTOR_TIME_SCALE

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_DETECTOR_TIME_SCALE)
        return -1;

    reading_items = fread(&section_length, sizeof(unsigned int), 1, file);
    if (reading_items < 1)
        return -1;

    reading_items = fread(&int_buffer, sizeof(int), 1, file);
    if (reading_items < 1 || int_buffer != number_of_detectors)
        return -1;

    for (int i = 0; i < number_of_detectors; ++i)
    {
        reading_items = fread(&int_buffer, sizeof(int), 1, file);
        if (reading_items < 1)
            return -1;

        detector_info[i].number_of_maps = int_buffer;
        detector_info[i].map = new trajectory_map[detector_info[i].number_of_maps];

        float_number_of_photons = (float)detector_info[i].number_of_photons;

        for (int j = 0; j < detector_info[i].number_of_maps; ++j)
        {
            reading_items = fread(double_buffer, sizeof(double), 1, file);
            if (reading_items < 1)
                return -1;

            if (j == 0) 
                detector_info[i].time_start = (float)(double_buffer[0]);

            reading_items = fread(double_buffer, sizeof(double), 1, file);
            if (reading_items < 1)
                return -1;

            if (j == detector_info[i].number_of_maps - 1) 
                detector_info[i].time_finish = (float)(double_buffer[0]);

            reading_items = fread(&uint64_buffer, sizeof(uint64), 1, file);
            if (reading_items < 1)
                return -1;

            reading_items = fread(&int_buffer, sizeof(int), 1, file);
            if (reading_items < 1 || int_buffer != total_map_size)
                return -1;

            reading_items = fread(uint64_map, sizeof(uint64), total_map_size, file);
            if (reading_items < total_map_size)
                return -1;

            for (int k = 0; k < total_map_size; ++k)
            {
                float_map[k] = (float)uint64_map[k];
            }

            detector_info[i].map[j].set(area_info->size, float_map);
        }
    }

    fclose(file);

    delete[] uint64_map;
    delete[] float_map;

    return 0;
}

int ReadNumberOfSurfaces(const char* file_name, int* number_of_surfaces)
{
    unsigned long long int reading_items;
    int section;

    FILE* file = NULL;
    fopen_s(&file, file_name, "rb");
    if (file == NULL)
        return -1;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_SURFACES)
        return -1;

    reading_items = fread(number_of_surfaces, sizeof(int), 1, file);
    if (reading_items < 1)
        return -1;

    fclose(file);

    return 0;
}

int ReadSurfaces(const char* file_name, surface* surfaces, int number_of_surfaces)
{
    unsigned long long int reading_items;
    int section;
    int int_buffer;
    int number_of_vertices;
    int number_of_triangles;
    double* double_buffer;
    float* vertices;
    int* triangles;

    FILE* file = NULL;
    fopen_s(&file, file_name, "rb");
    if (file == NULL)
        return -1;

    reading_items = fread(&section, sizeof(int), 1, file);
    if (reading_items < 1 || section != MCML_SECTION_SURFACES)
        return -1;

    reading_items = fread(&int_buffer, sizeof(int), 1, file);
    if (reading_items < 1 || int_buffer != number_of_surfaces)
        return -1;

    for (int i = 0; i < number_of_surfaces; ++i)
    {
        reading_items = fread(&number_of_vertices, sizeof(int), 1, file);
        if (reading_items < 1)
            return -1;

        double_buffer = new double[3 * number_of_vertices];
        reading_items = fread(double_buffer, sizeof(double), 3 * number_of_vertices, file);
        if (reading_items < 3 * number_of_vertices)
            return -1;

        reading_items = fread(&number_of_triangles, sizeof(int), 1, file);
        if (reading_items < 1)
            return -1;

        triangles = new int[3 * number_of_triangles];
        reading_items = fread(triangles, sizeof(int), 3 * number_of_triangles, file);
        if (reading_items < 3 * number_of_triangles)
            return -1;

        vertices = new float[3 * 3 * number_of_triangles];
        for (int j = 0; j < number_of_triangles; ++j)
        {
            vertices[9 * j + 0] = (float)(double_buffer[3 * triangles[3 * j + 0] + 0]);
            vertices[9 * j + 1] = (float)(double_buffer[3 * triangles[3 * j + 0] + 1]);
            vertices[9 * j + 2] = (float)(double_buffer[3 * triangles[3 * j + 0] + 2]);
            vertices[9 * j + 3] = (float)(double_buffer[3 * triangles[3 * j + 1] + 0]);
            vertices[9 * j + 4] = (float)(double_buffer[3 * triangles[3 * j + 1] + 1]);
            vertices[9 * j + 5] = (float)(double_buffer[3 * triangles[3 * j + 1] + 2]);
            vertices[9 * j + 6] = (float)(double_buffer[3 * triangles[3 * j + 2] + 0]);
            vertices[9 * j + 7] = (float)(double_buffer[3 * triangles[3 * j + 2] + 1]);
            vertices[9 * j + 8] = (float)(double_buffer[3 * triangles[3 * j + 2] + 2]);
        }

        surfaces[i].set_vertices(vertices, 3 * number_of_triangles);

        delete[] double_buffer;
        delete[] vertices;
        delete[] triangles;
    }


    fclose(file);

    return 0;
}