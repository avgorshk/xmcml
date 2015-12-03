#include "..\..\..\xmcml\xmcml\mcml_kernel_types.h"

void GenerateSurfaceFtomFunction(double3 center, double lengthX, double lengthY, int scaleSizeX, int scaleSizeY, double (*inputFunction)(double , double), Surface* fSurface);