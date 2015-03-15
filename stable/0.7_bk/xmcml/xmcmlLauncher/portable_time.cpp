#include "portable_time.h"

#ifdef _WIN32

#include <Windows.h>

double PortableGetTime()
{
    return GetTickCount() * 1.0e-3;
}

#else

#include <time.h>
#include <sys/time.h>

double PortableGetTime()
{
    struct timeval t;
    gettimeofday(&t, 0);
    return (t.tv_sec * 1000000ULL + t.tv_usec) * 1.0e-6;
}

#endif
