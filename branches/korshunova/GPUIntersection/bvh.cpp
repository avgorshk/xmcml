#include "mcml_intersection.h"
#include "mcml_math.h"
#include <stack>
#include <list>
using namespace std;

AABB createBox(Triangle* tris, int numOfTriangles, Surface* surfaces)
{
	AABB box;
	floatVec3* V = surfaces[tris[0].surfId].vertices;
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

void FindBVHSplit(AABB Box, Triangle *triangles, int numberOfTriangles, Surface* surfaces, float &min_sah, int &split, int &splitAxis)
{
	//корзины, содержат число треугольников в каждой из них
	int Bins[3][BIN_NUMBER];
	for (int i = 0; i < BIN_NUMBER; i++)
		Bins[0][i] = Bins[1][i] = Bins[2][i] = 0;		
	float c; // центр треугольника
	int binIndex;
	floatVec3 *ver;
	int3 Tr;
	float size[3], d[3];

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
				c = (ver[Tr.x].cell[axis] + ver[Tr.y].cell[axis] + ver[Tr.z].cell[axis]) / 3.f;
				binIndex = (int) ((c - Box.Ver1.cell[axis]) / d[axis]);
				if (binIndex == BIN_NUMBER) binIndex--;
			} else binIndex = 0;
			Bins[axis][binIndex]++;
		}
	}
	
	//подсчет SAH на границах корзин
	float sah;
	float leftArea, rightArea, tmpSize;
	int leftCount, rightCount;
	float invArea = 0.5f / (size[0]*size[1] + size[1]*size[2] + size[0]*size[2]); 

	min_sah = MAX_DISTANCE;
	
	for (int axis = 0; axis < 3; axis++)
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
					leftArea = 2.f * (size[0]*size[1] + size[1]*size[2] + size[2]*size[0]);
					size[axis] = tmpSize - size[axis];
					rightArea = 2.f * (size[0]*size[1] + size[1]*size[2] + size[2]*size[0]);
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
		return nodes.size() - 1;
	}

	int split, splitAxis;
	float sah, min_sah;
	
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
		return nodes.size() - 1;
	}
	int leftCount = 0;
	Triangle tempTr;
	float c1, c2;
	int i1, i2;
	float d = (Box.Ver2.cell[splitAxis] - Box.Ver1.cell[splitAxis]) / BIN_NUMBER;

	int k = 0;
	int j = numberOfTriangles-1;
	floatVec3 *ver = surfaces[nodeTris[k].surfId].vertices;
	int3 Tr = surfaces[nodeTris[k].surfId].triangles[nodeTris[k].trIndex];
	c1 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.f;
	i1 = (int) ((c1 - Box.Ver1.cell[splitAxis]) / d);
	if (i1 == BIN_NUMBER) i1--;

	ver = surfaces[nodeTris[j].surfId].vertices;
	Tr = surfaces[nodeTris[j].surfId].triangles[nodeTris[j].trIndex];
	c2 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.f;
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
				c1 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] +ver[Tr.z].cell[splitAxis]) / 3.f;
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
				c2 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.f;
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
				c1 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] +ver[Tr.z].cell[splitAxis]) / 3.f;
				i1 = (int) ((c1 - Box.Ver1.cell[splitAxis]) / d);
				if (i1 == BIN_NUMBER) i1--;
			}
			if (j >= 0)
			{
				ver = surfaces[nodeTris[j].surfId].vertices;
				Tr = surfaces[nodeTris[j].surfId].triangles[nodeTris[j].trIndex];
				c2 = (ver[Tr.x].cell[splitAxis] + ver[Tr.y].cell[splitAxis] + ver[Tr.z].cell[splitAxis]) / 3.f;
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
	return nodes.size() - 1;
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
    tree->numOfNodes = nodes.size();
    tree->nodes = new BVHNode[tree->numOfNodes];
    for (int i = 0; i < tree->numOfNodes; i++)
    {
        tree->nodes[i] = nodes[i];
    }
	return tree;
}

BVHBoxIntersectionInfo createBVHBoxIntersectionInfo(int node, float tnear)
{
    BVHBoxIntersectionInfo info;
    info.node = node;
    info.tnear = tnear;
    return info;
}

IntersectionInfo ComputeBVHIntersection(floatVec3 origin, floatVec3 direction, float step, BVHTree* tree, Surface* surfaces)
{
	IntersectionInfo result;
	result.isFindIntersection = 0;
	result.distance = MAX_DISTANCE;

	float t;
	int curr = tree->root;
	float tnear1, tnear2, tfar;
	bool isIntersectBox = IntersectAABB(tree->nodes[curr].Box, origin, direction, tnear1, tfar);
	if(!isIntersectBox || step < tnear1)
		return result;

	stack<BVHBoxIntersectionInfo, list<BVHBoxIntersectionInfo>> bvhStack;
	BVHBoxIntersectionInfo info = createBVHBoxIntersectionInfo(curr, tnear1);
	int tr_index;
	int left, right;
	floatVec3 *ver;
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

IntersectionInfo ComputeBVHIntersectionWithoutStep(floatVec3 origin, floatVec3 direction, BVHTree* tree, Surface* surfaces)
{
	IntersectionInfo result;
	result.isFindIntersection = 0;
	result.distance = MAX_DISTANCE;

	float t;
	int curr = tree->root;
	float tnear1, tnear2, tfar;
	bool isIntersectBox = IntersectAABB(tree->nodes[curr].Box, origin, direction, tnear1, tfar);
	if(!isIntersectBox)
		return result;

	stack<BVHBoxIntersectionInfo, list<BVHBoxIntersectionInfo>> bvhStack;
	BVHBoxIntersectionInfo info = createBVHBoxIntersectionInfo(curr, tnear1);
	int tr_index;
	int left, right;
	floatVec3 *ver;
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