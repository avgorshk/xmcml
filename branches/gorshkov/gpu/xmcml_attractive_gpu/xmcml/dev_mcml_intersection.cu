#include <math.h>
#include <stdio.h>

#include "mcml_intersection.h"

#include "dev_mcml_math.cu"

#define STACK_SIZE 32

__device__ double gpuIntersectAABB(AABB &Box, const Double3 &origin, const Double3 &direction)
{
	double tmin = - MAX_DISTANCE, tmax = MAX_DISTANCE;
    double t_near = MAX_DISTANCE;
	double t1, t2;
	for ( short i = 0; i < 3; i++ ) 
    {
		if ( fabs(direction.cell[i]) > EPSILON ) 
        {
			t1 = ( Box.Ver1.cell[i] - origin.cell[i] ) / direction.cell[i];
			t2 = ( Box.Ver2.cell[i] - origin.cell[i] ) / direction.cell[i];
			tmin = max(tmin, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
			if ( tmin > tmax || tmax < 0) 
			{
				return MAX_DISTANCE;
			}
		}
		else 
        {
			if ((Box.Ver1.cell[i] - origin.cell[i] > 0) || (Box.Ver2.cell[i] - origin.cell[i] < 0)) 
			{
				return MAX_DISTANCE;
			}
		}
	}
	t_near = tmin;
    return t_near;
}

__device__ double gpuGetTriangleIntersectionDistance(Double3 &origin, Double3 &dir, Double3 &a, Double3 &b, Double3 &c)
{
	Double3 edge1 = gpuSubVector(b, a);
	Double3 edge2 = gpuSubVector(c, a);
	
	Double3 pvec = gpuCrossVector(dir, edge2);

	double inv_det = edge1.x*pvec.x + edge1.y*pvec.y + edge1.z*pvec.z;

	if (inv_det < EPSILON && inv_det > -EPSILON)
	{
		return MAX_DISTANCE;
	}
	inv_det = 1.f / inv_det;

	Double3 tvec = gpuSubVector(origin, a);

	double u = (tvec.x*pvec.x + tvec.y*pvec.y + tvec.z*pvec.z) * inv_det;
	
	if (u < 0.f || u > 1.f)
	{
		return MAX_DISTANCE;
	}

	pvec = gpuCrossVector(tvec, edge1);
	double v = (dir.x*pvec.x + dir.y*pvec.y + dir.z*pvec.z) * inv_det;

	if (v < 0.f || u + v > 1.f)
	{
		return MAX_DISTANCE;
	}

	return (edge2.x*pvec.x + edge2.y*pvec.y + edge2.z*pvec.z) * inv_det;
}

__device__ IntersectionInfo GpuComputeBVHIntersection(Double3 origin, Double3 direction, double step, BVHTree* tree, Surface* surfaces)
{
    IntersectionInfo result;
	int curr, left, right;
	double tnear1, tnear2, t, distance;
	short top, tr_index;
	Triangle currTr;
	Double3* ver;
	Int3 Tr;
    int offset;
    bool useStack;
	
	BVHBoxIntersectionInfo bvhStack[STACK_SIZE];
		
	result.isFindIntersection = 0;
	result.distance = MAX_DISTANCE;
        
	distance = MAX_DISTANCE;
	top = -1;                
	curr = tree->root;
		
	tnear1 = gpuIntersectAABB(tree->nodes[curr].Box, origin, direction);
	if (tnear1 <= step)
	{	                
	    while (true)
	    {                
            useStack = true;                
            if (tree->nodes[curr].offset >= 0)
		    {
                offset = tree->nodes[curr].offset;
                tr_index = -1;
			    for (short i = 0; i < tree->nodes[curr].numOfTriangles; i++)
			    {
                    currTr = tree->triangles[offset + i];
                    ver = surfaces[currTr.surfId].vertices;
				    Tr = surfaces[currTr.surfId].triangles[currTr.trIndex];	
				    t = gpuGetTriangleIntersectionDistance(origin, direction, 
                                                            ver[Tr.x], ver[Tr.y], ver[Tr.z]);
				    if (t >= 0.f && t <= step && t < distance)
				    {
					    distance = t;
					    tr_index = i;
				    }
			    }
			    if (tr_index >= 0)
			    {
                    currTr = tree->triangles[offset + tr_index];
                    ver = surfaces[currTr.surfId].vertices;
                    Tr = surfaces[currTr.surfId].triangles[currTr.trIndex];
                    result.surfaceId = currTr.surfId;				                     
				    result.isFindIntersection = 1;
                    result.distance = distance;
				    result.normal = gpuGetPlaneNormal(ver[Tr.x], ver[Tr.y], ver[Tr.z]);
				    result.normal = gpuNormalizeVector(result.normal);
			    }			        
		    }
		    else 
            {
			    left = tree->nodes[curr].leftNode;
			    right = tree->nodes[curr].rightNode;
			    tnear1 = gpuIntersectAABB(tree->nodes[left].Box, origin, direction);
			    tnear2 = gpuIntersectAABB(tree->nodes[right].Box, origin, direction);
                if (min(tnear1, tnear2) <= step)
                {
                    useStack = false;
                    curr = (tnear1 > tnear2) ? right : left;
					if (max(tnear1, tnear2) <= step)
					{
						top++;
						bvhStack[top].tnear = max(tnear1, tnear2);
						bvhStack[top].node = (tnear1 > tnear2) ? left : right;
					}
                }				      
		    }

		    if (useStack) 
            {
			    if (top < 0)
                {
				    break;
                }
                curr = bvhStack[top].node;
                top--;
				while (top >= 0 && result.distance < bvhStack[top+1].tnear)
				{
					top--;
				}
				if (result.distance < bvhStack[top+1].tnear || step < bvhStack[top+1].tnear)
                {
                    break;
                }
				else curr = bvhStack[top+1].node;
		    }
	    }
    }

    return result;
}
