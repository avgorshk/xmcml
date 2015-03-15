#ifndef __QUASISPHERE_H
#define __QUASISPHERE_H

#include "..\..\..\xmcml\xmcml\xmcml\mcml_kernel_types.h"

void GenerateBottomHalfQuasiSphere(double3 center, double radius, int numberOfSubdividings, 
    double delta, Surface* quasiSphere);

#endif //__QUASISPHERE_H
