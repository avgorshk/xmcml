#include "mcml_intersection.h"
#include "mcml_math.h"

#include <stdlib.h>
#include <vector>
#include <stack>
#include <list>

using namespace std;

bool IntersectAABB(AABB Box, double3 origin, double3 direction, double& t_near, double& t_far)
{
	double tmin = - MAX_DISTANCE, tmax = MAX_DISTANCE;
	double t1, t2, t;

	for ( int i = 0; i < 3; i++ ) {
		if ( abs(direction.cell[i]) > EPSILON ) {
			t1 = ( Box.Ver1.cell[i] - origin.cell[i] ) / direction.cell[i];
			t2 = ( Box.Ver2.cell[i] - origin.cell[i] ) / direction.cell[i];
			if ( t2 < t1 ) { t = t1; t1 = t2; t2 = t; }
			if ( t1 > tmin ) tmin = t1;
			if ( t2 < tmax ) tmax = t2;
			if ( tmin > tmax || tmax < 0) 
			{
				t_near = MAX_DISTANCE;
				t_far = MAX_DISTANCE;
				return false;
			}
		}
		else {
			if ((Box.Ver1.cell[i] - origin.cell[i] > 0) || (Box.Ver2.cell[i] - origin.cell[i] < 0)) 
			{
				t_near = MAX_DISTANCE;
				t_far = MAX_DISTANCE;
				return false;
			}
		}
	}
	t_near = tmin; t_far = tmax; 
	return true;
}

double GetTriangleIntersectionDistance(double3 origin, double3 direction, double3 a, double3 b, double3 c)
{
    double3 edge1;
	double3 edge2;
	double3 tvec; 
	double3 pvec;
	double3 qvec;

	double det, inv_det;
    double u, v;

	edge1 = SubVector(b, a);
	edge2 = SubVector(c, a);
	
	pvec = CrossVector(direction, edge2);

	det = DotVector(edge1, pvec);

	if (det < EPSILON && det > -EPSILON)
	{
		return MAX_DISTANCE;
	}
	inv_det = 1.0 / det;

	tvec = SubVector(origin, a);

	u = DotVector(tvec, pvec) * inv_det;
	
	if (u < 0.0 || u > 1.0)
	{
		return MAX_DISTANCE;
	}

	qvec = CrossVector(tvec, edge1);
	v = DotVector(direction, qvec) * inv_det;

	if (v < 0.0 || u + v > 1.0)
	{
		return MAX_DISTANCE;
	}

	return DotVector(edge2, qvec) * inv_det;
}

AABB createBox(Triangle* tris, int numOfTriangles, Surface* surfaces)
{
	AABB box;
	double3* V = surfaces[tris[0].surfId].vertices;
	int3 Tr = surfaces[tris[0].surfId].triangles[tris[0].trIndex];
	box.Ver1.x = box.Ver1.y = box.Ver1.z = MAX_DISTANCE;
	box.Ver2.x = box.Ver2.y = box.Ver2.z = -MAX_DISTANCE;
	
	for(int i = 0; i < numOfTriangles; i++)
	{
		V = surfaces[tris[i].surfId].vertices;
		Tr = surfaces[tris[i].surfId].triangles[tris[i].trIndex];

		for (int axis = 0; axis < 3; axis++)
		{
			if (V[Tr.x].cell[axis] < box.Ver1.cell[axis]) 
				box.Ver1.cell[axis] = V[Tr.x].cell[axis];
			if (V[Tr.y].cell[axis] < box.Ver1.cell[axis]) 
				box.Ver1.cell[axis] = V[Tr.y].cell[axis];
			if (V[Tr.z].cell[axis] < box.Ver1.cell[axis]) 
				box.Ver1.cell[axis] = V[Tr.z].cell[axis];
        }

        for (int axis = 0; axis < 3; axis++)
		{
			if (V[Tr.x].cell[axis] > box.Ver2.cell[axis]) 
				box.Ver2.cell[axis] = V[Tr.x].cell[axis];
			if (V[Tr.y].cell[axis] > box.Ver2.cell[axis]) 
				box.Ver2.cell[axis] = V[Tr.y].cell[axis];
			if (V[Tr.z].cell[axis] > box.Ver2.cell[axis]) 
				box.Ver2.cell[axis] = V[Tr.z].cell[axis];
		}
	}

	return box;
}

void FindBVHSplit(AABB Box, Triangle *triangles, int numberOfTriangles, Surface* surfaces, double &min_sah, int &split, int &splitAxis)
{
	//корзины, содержат число треугольников в каждой из них
	int Bins[3][BIN_NUMBER];
	for (int i = 0; i < BIN_NUMBER; i++)
		Bins[0][i] = Bins[1][i] = Bins[2][i] = 0;		
	double c; // центр треугольника
	int binIndex;
	double3 *ver;
	int3 Tr;
	double size[3], d[3];

	size[0] = Box.Ver2.x - Box.Ver1.x;
	size[1] = Box.Ver2.y - Box.Ver1.y;
	size[2] = Box.Ver2.z - Box.Ver1.z;
	d[0] = size[0] / BIN_NUMBER;
	d[1] = size[1] / BIN_NUMBER;
	d[2] = size[2] / BIN_NUMBER;
	
	//распределение по корзинам
	for(int i = 0; i < numberOfTriangles; i++)
	{
		ver = surfaces[triangles[i].surfId].vertices;
		Tr = surfaces[triangles[i].surfId].triangles[triangles[i].trIndex];

		for (int axis = 0; axis < 3; axis++)
		{
			if (d[axis] > EPSILON)
			{
				c = (ver[Tr.x].cell[axis] + ver[Tr.y].cell[axis] + ver[Tr.z].cell[axis]) / 3.0;
				binIndex = (int) ((c - Box.Ver1.cell[axis]) / d[axis]);
				if (binIndex == BIN_NUMBER) binIndex--;
			} else binIndex = 0;
			Bins[axis][binIndex]++;
		}
	}
	
	//подсчет SAH на границах корзин
	double sah;
	double leftArea, rightArea, tmpSize;
	int leftCount, rightCount;
	double invArea = 0.5 / (size[0]*size[1] + size[1]*size[2] + size[0]*size[2]); 

	min_sah = MAX_DISTANCE;
	
	for (int axis = 0; axis < 3; axis++)
    {
		if (d[axis] > EPSILON)
		{
			leftCount = 0;
			rightCount = numberOfTriangles;
			tmpSize = size[axis];

			for (int i = 1; i < BIN_NUMBER; i++)
			{
				leftCount += Bins[axis][i-1];
				rightCount -= Bins[axis][i-1];
				if (leftCount > 0 && rightCount > 0)
				{
					size[axis] = i*d[axis];
					leftArea = 2.0 * (size[0]*size[1] + size[1]*size[2] + size[2]*size[0]);
					size[axis] = tmpSize - size[axis];
					rightArea = 2.0 * (size[0]*size[1] + size[1]*size[2] + size[2]*size[0]);
					sah = invArea*(leftCount*leftArea + rightArea*rightCount);
					if (sah < min_sah)
					{
						min_sah = sah;
						splitAxis = axis;
						split = i;
					}
				}
			}
			size[axis] = tmpSize;
		}
    }
}

int GenerateBVHNode(vector<BVHNode> &nodes, Triangle *triangles, int offset, int numberOfTriangles, Surface* surfaces)
{
    Triangle* nodeTris = (triangles + offset);
	AABB Box = createBox(nodeTris, numberOfTriangles, surfaces);

	//завершение построения, если мало треугольников
	if (numberOfTriangles <= MAX_TRIANGLES_IN_LEAVES)
	{
        BVHNode node;
		node.Box = Box;
		node.rightNode = -1;
		node.leftNode = -1;
		node.numOfTriangles = numberOfTriangles;
        node.offset = offset;
        nodes.push_back(node);
		return (int)(nodes.size() - 1);
	}

	int split, splitAxis;
	double sah, min_sah;
	
	FindBVHSplit(Box, nodeTris, numberOfTriangles, surfaces, min_sah, split, splitAxis);

	//выбор разбиения
	sah = numberOfTriangles;
	if (sah <= min_sah)
	{
        BVHNode node;
		node.Box = Box;
		node.leftNode = -1;
		node.rightNode = -1;
		node.numOfTriangles = numberOfTriangles;
        node.offset = offset;
        nodes.push_back(node);
		return (int)(nodes.size() - 1);
	}
	int leftCount = 0;
	Triangle tempTr;
	double c1, c2;
	int i1, i2;
	double d = (Box.Ver2.cell[splitAxis] - Box.Ver1.cell[splitAxis]) / BIN_NUMBER;

	int k = 0;
	int j = numberOfTriangles-1;
	double3 *ver = surfaces[nodeTris[k].surfId].vertices;
	int3 Tr = surfaces[nodeTris[k].surfId].triangles[nodeTris[k].trIndex];
	c1 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.0;
	i1 = (int) ((c1 - Box.Ver1.cell[splitAxis]) / d);
	if (i1 == BIN_NUMBER) i1--;

	ver = surfaces[nodeTris[j].surfId].vertices;
	Tr = surfaces[nodeTris[j].surfId].triangles[nodeTris[j].trIndex];
	c2 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.0;
	i2 = (int) ((c2 - Box.Ver1.cell[splitAxis]) / d);
	if (i2 == BIN_NUMBER) i2--;

	while (k < j) {
		if (i1 < split) {
			k++;
			leftCount++;
			if (k < numberOfTriangles) 
			{
				ver = surfaces[nodeTris[k].surfId].vertices;
				Tr = surfaces[nodeTris[k].surfId].triangles[nodeTris[k].trIndex];
				c1 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] +ver[Tr.z].cell[splitAxis]) / 3.0;
				i1 = (int) ((c1 - Box.Ver1.cell[splitAxis]) / d);
				if (i1 == BIN_NUMBER) i1--;
			}
		}
		if (i2 >= split) {
			j--;
			if (j >= 0)
			{
				ver = surfaces[nodeTris[j].surfId].vertices;
				Tr = surfaces[nodeTris[j].surfId].triangles[nodeTris[j].trIndex];
				c2 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.0;
				i2 = (int) ((c2 - Box.Ver1.cell[splitAxis]) / d);
				if (i2 == BIN_NUMBER) i2--;
			}
		}
		if (i1 >= split && i2 < split) {
			tempTr = nodeTris[k];
			nodeTris[k] = nodeTris[j];
			nodeTris[j] = tempTr;
			k++; j--;
			leftCount++;
			if (k < numberOfTriangles)
			{
				ver = surfaces[nodeTris[k].surfId].vertices;
				Tr = surfaces[nodeTris[k].surfId].triangles[nodeTris[k].trIndex];
				c1 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] +ver[Tr.z].cell[splitAxis]) / 3.0;
				i1 = (int) ((c1 - Box.Ver1.cell[splitAxis]) / d);
				if (i1 == BIN_NUMBER) i1--;
			}
			if (j >= 0)
			{
				ver = surfaces[nodeTris[j].surfId].vertices;
				Tr = surfaces[nodeTris[j].surfId].triangles[nodeTris[j].trIndex];
				c2 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.0;
				i2 = (int) ((c2 - Box.Ver1.cell[splitAxis]) / d);
				if (i2 == BIN_NUMBER) i2--;
			}
		}	
	}
    BVHNode node;
	node.Box = Box;
	node.numOfTriangles = 0;
    node.offset = -1;
	node.leftNode = GenerateBVHNode(nodes, triangles, offset, leftCount, surfaces);
	node.rightNode = GenerateBVHNode(nodes, triangles, offset + leftCount, numberOfTriangles - leftCount, surfaces);
    nodes.push_back(node);
	return (int)(nodes.size() - 1);
}

BVHTree* GenerateBVHTree(Surface* surfaces, int numberOfSurfaces)
{
	BVHTree* tree = new BVHTree();
	int numOfTriangles = 0;
	int i = 0, j = 0;
	for(i = 0; i < numberOfSurfaces; i++)
		numOfTriangles += surfaces[i].numberOfTriangles;
    tree->numOfTriangles = numOfTriangles;
	tree->triangles = new Triangle[numOfTriangles];
	int index = 0;
	for (i = 0; i < numberOfSurfaces; i++)
	{
		for (j = 0; j < surfaces[i].numberOfTriangles; j++)
		{
			tree->triangles[index].surfId = i;
			tree->triangles[index].trIndex = j;
			index++;
		}
	}
    vector<BVHNode> nodes;
	tree->root = GenerateBVHNode(nodes, tree->triangles, 0, numOfTriangles, surfaces);
    tree->numOfNodes = (int)(nodes.size());
    tree->nodes = new BVHNode[tree->numOfNodes];
    for (int i = 0; i < tree->numOfNodes; i++)
    {
        tree->nodes[i] = nodes[i];
    }
	return tree;
}

BVHBoxIntersectionInfo createBVHBoxIntersectionInfo(int node, double tnear)
{
    BVHBoxIntersectionInfo info;
    info.node = node;
    info.tnear = tnear;
    return info;
}

IntersectionInfo ComputeBVHIntersection(double3 origin, double3 direction, double step, BVHTree* tree, Surface* surfaces)
{
	IntersectionInfo result;
	result.isFindIntersection = 0;
	result.distance = MAX_DISTANCE;

	double t;
	int curr = tree->root;
	double tnear1, tnear2, tfar;
	bool isIntersectBox = IntersectAABB(tree->nodes[curr].Box, origin, direction, tnear1, tfar);
	if(!isIntersectBox || step < tnear1)
		return result;

	stack<BVHBoxIntersectionInfo, list<BVHBoxIntersectionInfo> > bvhStack;
	BVHBoxIntersectionInfo info = createBVHBoxIntersectionInfo(curr, tnear1);
	int tr_index;
	int left, right;
	double3 *ver;
	int3 Tr;
	bool f1, f2, useStack;
	while (true)
	{
        if (tree->nodes[curr].offset >= 0)
		{
            Triangle* currTris = tree->triangles + tree->nodes[curr].offset;
			tr_index = -1;
			for (int i = 0; i < tree->nodes[curr].numOfTriangles; i++)
			{
                ver = surfaces[currTris[i].surfId].vertices;
				Tr = surfaces[currTris[i].surfId].triangles[currTris[i].trIndex];	
				t = GetTriangleIntersectionDistance(origin, direction, ver[Tr.x], ver[Tr.y], ver[Tr.z]);
				if (t >= 0.0 && t < result.distance)
				{
					result.distance = t;
					tr_index = i;
				}
			}
			if (tr_index >= 0 && result.distance <= step)
			{
                result.surfaceId = currTris[tr_index].surfId;
				ver = surfaces[result.surfaceId ].vertices;
                Tr = surfaces[result.surfaceId].triangles[currTris[tr_index].trIndex];
				result.isFindIntersection = 1;
				result.normal = GetPlaneNormal(ver[Tr.x], ver[Tr.y], ver[Tr.z]);
				result.normal = NormalizeVector(result.normal);
			}
			useStack = true;
		}
		else {
			left = tree->nodes[curr].leftNode;
			right = tree->nodes[curr].rightNode;
			f1 = IntersectAABB(tree->nodes[left].Box, origin, direction, tnear1, tfar);
			f2 = IntersectAABB(tree->nodes[right].Box, origin, direction, tnear2, tfar);
			useStack = false;
			if (f1 && f2)
            {
				if (tnear1 <= tnear2) 
				{
					if (tnear1 <= step) 
                    {
						curr = left;
						if (tnear2 <= step) 
                            bvhStack.push(createBVHBoxIntersectionInfo(right, tnear2));
					} 
                    else useStack = true;
				} 
                else 
                {
					if (tnear2 <= step) {
						curr = right;
						if (tnear1 <= step) 
                            bvhStack.push(createBVHBoxIntersectionInfo(left, tnear1));
					} 
                    else useStack = true;
				}
            }
			else 
            {
                if (f1) 
                {
					if (tnear1 <= step) 
						curr = left;
					else useStack = true;
				 } 
                else 
                {
                    if (f2 && tnear2 <= step)
				        curr = right;				
				    else useStack = true;
                }
            }
		}
		if (useStack) 
        {
			if (bvhStack.empty())
				return result;
			info = bvhStack.top();
			curr = info.node;
			bvhStack.pop();
			if (result.distance < info.tnear) 
            {
				while (!bvhStack.empty() && result.distance < info.tnear)
				{
					info = bvhStack.top();
					bvhStack.pop();
				}
				if (result.distance < info.tnear || step < info.tnear)
					return result;
				else curr = info.node;
			}
		}
	}
}

IntersectionInfo ComputeBVHIntersectionWithoutStep(double3 origin, double3 direction, BVHTree* tree, Surface* surfaces)
{
	IntersectionInfo result;
	result.isFindIntersection = 0;
	result.distance = MAX_DISTANCE;

	double t;
	int curr = tree->root;
	double tnear1, tnear2, tfar;
	bool isIntersectBox = IntersectAABB(tree->nodes[curr].Box, origin, direction, tnear1, tfar);
	if(!isIntersectBox)
		return result;

	stack<BVHBoxIntersectionInfo, list<BVHBoxIntersectionInfo> > bvhStack;
	BVHBoxIntersectionInfo info = createBVHBoxIntersectionInfo(curr, tnear1);
	int tr_index;
	int left, right;
	double3 *ver;
	int3 *Tr;
	int ind;
	bool f1, f2;
	while (true)
	{        
        if (tree->nodes[curr].offset >= 0)
		{
            Triangle* currTris = tree->triangles + tree->nodes[curr].offset;
			tr_index = -1;
			for (int i = 0; i < tree->nodes[curr].numOfTriangles; i++)
			{
                ver = surfaces[currTris[i].surfId].vertices;
				ind = currTris[i].trIndex;
				Tr = surfaces[currTris[i].surfId].triangles;
				t = GetTriangleIntersectionDistance(origin, direction, ver[Tr[ind].x], ver[Tr[ind].y], ver[Tr[ind].z]);
				if (t >= 0.0 && t < result.distance)
				{
					result.distance = t;
					tr_index = i;
				}
			}
			if (tr_index >= 0)
			{
                result.surfaceId = currTris[tr_index].surfId;
				ind = currTris[tr_index].trIndex;
				ver = surfaces[result.surfaceId ].vertices;
				Tr = surfaces[result.surfaceId].triangles;
				result.isFindIntersection = 1;
				result.normal = GetPlaneNormal(ver[Tr[ind].x], ver[Tr[ind].y], ver[Tr[ind].z]);
				result.normal = NormalizeVector(result.normal);
			}
			if (bvhStack.empty())
				return result;
			info = bvhStack.top();
			curr = info.node;
			bvhStack.pop();
			if (result.distance < info.tnear) 
            {
				while (!bvhStack.empty() && result.distance < info.tnear)
				{
					info = bvhStack.top();
					bvhStack.pop();
				}
				if (result.distance < info.tnear)
					return result;
				else curr = info.node;
			}	
		}
		else 
        {
			left = tree->nodes[curr].leftNode;
			right = tree->nodes[curr].rightNode;
			f1 = IntersectAABB(tree->nodes[left].Box, origin, direction, tnear1, tfar);
			f2 = IntersectAABB(tree->nodes[right].Box, origin, direction, tnear2, tfar);
			if (f1 && f2)
			{
				if (tnear1 <= tnear2) 
				{
					curr = left;
					bvhStack.push(createBVHBoxIntersectionInfo(right, tnear2));
				} 
                else 
                {
					curr = right;
					bvhStack.push(createBVHBoxIntersectionInfo(left, tnear1));
				}
			} 
            else 
            {
				if (f1)	
                    curr = left;
				else 
                {
					if (f2)	
                        curr = right;
					else
                    {
						if (bvhStack.empty())
							return result;
						info = bvhStack.top();
						curr = info.node;
						bvhStack.pop();
						if (result.distance < info.tnear) 
                        {
							while (!bvhStack.empty() && result.distance < info.tnear)
							{
								info = bvhStack.top();
								bvhStack.pop();
							}
							if (result.distance < info.tnear)
								return result;
							else curr = info.node;
						}
					}
                }
			}
		}
	}
}
