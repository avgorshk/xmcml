#include "test_cuda.h"

using namespace std;

#define THREADS 256
#define BLOCKS 512
#define PORTION_SIZE (32*THREADS*BLOCKS)

#define STACK_SIZE 32

__constant__ GpuBVH TREE;

__device__ floatVec3 subVector(floatVec3 &a, floatVec3 &b)
{
    floatVec3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

__device__ floatVec3 crossVector(floatVec3 &a, floatVec3 &b)
{
    floatVec3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

__device__ floatVec3 normalizeVector(floatVec3 &a)
{
    floatVec3 result;
    float inv_length = 1.0 / sqrt(a.x*a.x + a.y*a.y + a.z*a.z);    
    result.x = a.x * inv_length;
    result.y = a.y * inv_length;
    result.z = a.z * inv_length;
    return result;
}

__device__ floatVec3 getPlaneNormal(floatVec3 &a, floatVec3 &b, floatVec3 &c)
{
    floatVec3 vec1 = subVector(a, b);
    floatVec3 vec2 = subVector(a, c);
	return crossVector(vec1, vec2);
}

__device__ float gpuIntersectAABB(AABB &Box, const floatVec3 &origin, const floatVec3 &direction)
{
	float tmin = - MAX_DISTANCE, tmax = MAX_DISTANCE;
    float t_near = MAX_DISTANCE;
	float t1, t2;
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

__device__ float gpuGetTriangleIntersectionDistance(floatVec3 &origin, floatVec3 &dir, floatVec3 &a, floatVec3 &b, floatVec3 &c)
{
	floatVec3 edge1 = subVector(b, a);
	floatVec3 edge2 = subVector(c, a);
	
	floatVec3 pvec = crossVector(dir, edge2);

	float inv_det = edge1.x*pvec.x + edge1.y*pvec.y + edge1.z*pvec.z;

	if (inv_det < EPSILON && inv_det > -EPSILON)
	{
		return MAX_DISTANCE;
	}
	inv_det = 1.f / inv_det;

	floatVec3 tvec = subVector(origin, a);

	float u = (tvec.x*pvec.x + tvec.y*pvec.y + tvec.z*pvec.z) * inv_det;
	
	if (u < 0.f || u > 1.f)
	{
		return MAX_DISTANCE;
	}

	pvec = crossVector(tvec, edge1);
	float v = (dir.x*pvec.x + dir.y*pvec.y + dir.z*pvec.z) * inv_det;

	if (v < 0.f || u + v > 1.f)
	{
		return MAX_DISTANCE;
	}

	return (edge2.x*pvec.x + edge2.y*pvec.y + edge2.z*pvec.z) * inv_det;
}

__global__ void GpuBVHTraverse(floatVec3 *origin, floatVec3 *direction, float *step, int numRays, GPUSurface *surfaces, IntersectionInfo *results)
{
	int th_ind = blockIdx.x*blockDim.x + threadIdx.x;

	int curr, left, right;
	float tnear1, tnear2, t, distance;
	floatVec3 _origin, _direction;
	float _step;
	short top, tr_index;
	Triangle currTr;
	floatVec3* ver;
	int3 Tr;
    int offset;
    bool useStack;
	
	BVHBoxIntersectionInfo bvhStack[STACK_SIZE];
		
    while (th_ind < numRays)
    {
	    results[th_ind].isFindIntersection = 0;
	    results[th_ind].distance = MAX_DISTANCE;
        
		distance = MAX_DISTANCE;
		_origin = origin[th_ind];
		_direction = direction[th_ind];
		_step = step[th_ind];
		top = -1;                
	    curr = TREE.root;
		
	    tnear1 = gpuIntersectAABB(TREE.nodes[curr].Box, _origin, _direction);
	    if (tnear1 <= _step)
		{	                
	        while (true)
	        {                
                useStack = true;                
                if (TREE.nodes[curr].offset >= 0)
		        {
                    offset = TREE.nodes[curr].offset;
                    tr_index = -1;
			        for (short i = 0; i < TREE.nodes[curr].numOfTriangles; i++)
			        {
                        currTr = TREE.triangles[offset + i];
                        ver = surfaces[currTr.surfId].vertices;
				        Tr = surfaces[currTr.surfId].triangles[currTr.trIndex];	
				        t = gpuGetTriangleIntersectionDistance(_origin, _direction, 
                                                               ver[Tr.x], ver[Tr.y], ver[Tr.z]);
				        if (t >= 0.f && t <= _step && t < distance)
				        {
					        distance = t;
					        tr_index = i;
				        }
			        }
			        if (tr_index >= 0)
			        {
                        currTr = TREE.triangles[offset + tr_index];
                        ver = surfaces[currTr.surfId].vertices;
                        Tr = surfaces[currTr.surfId].triangles[currTr.trIndex];
                        results[th_ind].surfaceId = currTr.surfId;				                     
				        results[th_ind].isFindIntersection = 1;
                        results[th_ind].distance = distance;
				        results[th_ind].normal = getPlaneNormal(ver[Tr.x], ver[Tr.y], ver[Tr.z]);
				        results[th_ind].normal = normalizeVector(results[th_ind].normal);
			        }			        
		        }
		        else 
                {
			        left = TREE.nodes[curr].leftNode;
			        right = TREE.nodes[curr].rightNode;
			        tnear1 = gpuIntersectAABB(TREE.nodes[left].Box, _origin, _direction);
			        tnear2 = gpuIntersectAABB(TREE.nodes[right].Box, _origin, _direction);
                    if (min(tnear1, tnear2) <= _step)
                    {
                        useStack = false;
                        curr = (tnear1 > tnear2) ? right : left;
						if (max(tnear1, tnear2) <= _step)
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
				    while (top >= 0 && results[th_ind].distance < bvhStack[top+1].tnear)
				    {
					    top--;
				    }
				    if (results[th_ind].distance < bvhStack[top+1].tnear || _step < bvhStack[top+1].tnear)
                    {
                        break;
                    }
				    else curr = bvhStack[top+1].node;
		        }
	        }
        }
        th_ind += gridDim.x * blockDim.x;
    }
}

void GpuBVHIntersections(floatVec3* origin, floatVec3* direction, float* step, int numRays, BVHTree* tree, Surface* surfaces, int numSurfaces, IntersectionInfo *results, float &computeTime, float &fullTime)
{
    cudaEvent_t start, stop;    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEvent_t computeStart, computeStop;
    cudaEventCreate(&computeStart);
    cudaEventCreate(&computeStop);

    cudaEventRecord(start, 0);    

    GPUSurface *gpuSurfaces;                    
    gpuSurfaces = copySurfacesToGPU(surfaces, numSurfaces);  
    GpuBVH gpuTree = createGpuBVH(tree);      
	cudaMemcpyToSymbol(TREE, &gpuTree, sizeof(GpuBVH));
    
    int portionCount = numRays / PORTION_SIZE;    
    if (numRays % PORTION_SIZE > 0)
    {
        portionCount++;
    }
	
	floatVec3 *gpuOrigin, *gpuDirection;
    float *gpuSteps;  
    IntersectionInfo *gpuResults;

    if (portionCount == 1)
    {        
        int nBytesForRays = numRays * sizeof(floatVec3);
        int nBytesForRes = numRays * sizeof(IntersectionInfo);
        int nBytesForSteps = numRays * sizeof(float); 
        cudaMalloc((void**) &gpuOrigin, nBytesForRays);
        cudaMalloc((void**) &gpuDirection, nBytesForRays);
        cudaMalloc((void**) &gpuSteps, nBytesForSteps);
        cudaMalloc((void**) &gpuResults, nBytesForRes);     
    
        cudaMemcpy(gpuOrigin, origin, nBytesForRays, cudaMemcpyHostToDevice);    
        cudaMemcpy(gpuDirection, direction, nBytesForRays, cudaMemcpyHostToDevice);    
        cudaMemcpy(gpuSteps, step, nBytesForSteps, cudaMemcpyHostToDevice);    
      
        cudaEventRecord(computeStart, 0);
	    GpuBVHTraverse<<<BLOCKS, THREADS>>>(gpuOrigin, gpuDirection, gpuSteps, numRays, gpuSurfaces, gpuResults);
        cudaEventRecord(computeStop, 0);
        cudaEventSynchronize(computeStop);
        cudaEventElapsedTime(&computeTime, computeStart, computeStop); 
                   
        cudaMemcpy(results, gpuResults, nBytesForRes, cudaMemcpyDeviceToHost);        
    }
    else
    {
        int nBytesForRays = PORTION_SIZE * sizeof(floatVec3);
        int nBytesForRes = PORTION_SIZE * sizeof(IntersectionInfo);
        int nBytesForSteps = PORTION_SIZE * sizeof(float); 
        cudaMalloc((void**) &gpuOrigin, nBytesForRays);
        cudaMalloc((void**) &gpuDirection, nBytesForRays);
        cudaMalloc((void**) &gpuSteps, nBytesForSteps);
        cudaMalloc((void**) &gpuResults, nBytesForRes);  

        int n1 = 0, n_rest = numRays;  
        float t;      
        computeTime = 0;
        for (int i = 0; i < portionCount; i++)
        {
            n1 = min(PORTION_SIZE, n_rest);
            n_rest -= n1;

            cudaMemcpy(gpuOrigin, origin + i*PORTION_SIZE, n1*sizeof(floatVec3), cudaMemcpyHostToDevice);
            cudaMemcpy(gpuDirection, direction + i*PORTION_SIZE, n1*sizeof(floatVec3), cudaMemcpyHostToDevice);
            cudaMemcpy(gpuSteps, step + i*PORTION_SIZE, n1*sizeof(float), cudaMemcpyHostToDevice);               
       
            cudaEventRecord(computeStart, 0);

	        GpuBVHTraverse<<<BLOCKS, THREADS>>>(gpuOrigin, gpuDirection, gpuSteps, n1, gpuSurfaces, gpuResults);

            cudaEventRecord(computeStop, 0);
            cudaEventSynchronize(computeStop);
            cudaEventElapsedTime(&t, computeStart, computeStop);   
            computeTime += t;
            cudaMemcpy(results + i*PORTION_SIZE, gpuResults, n1*sizeof(IntersectionInfo), cudaMemcpyDeviceToHost);
        }         
    }
	
	cudaFree(gpuOrigin);
    cudaFree(gpuDirection);
    cudaFree(gpuSteps);
    cudaFree(gpuResults);

    releaseGPUSurfaces(gpuSurfaces, numSurfaces);
    releaseGpuBVH(gpuTree);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fullTime, start, stop);    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    cudaEventDestroy(computeStart);
    cudaEventDestroy(computeStop);  
}
