#include "mcml_intersection.h"
#include "mcml_math.h"
#include <stack>
#include <list>
#include <algorithm>
using namespace std;

#define BIN_COUNT 32
#define MAX_DEPTH 20
#define MAX_TRIANGLES 5

void FindKdSplit_bins(AABB Box, Triangle *triangles, int numberOfTriangles, Surface* surfaces, double &min_sah, double &split, int &splitAxis, int& leftCount, int &rightCount)
{
	//корзины, содержат число треугольников в каждой из них
	int Bins[3][BIN_COUNT];
	for (int i = 0; i < BIN_COUNT; i++)
		Bins[0][i] = Bins[1][i] = Bins[2][i] = 0;		
	double c; // центр треугольника
	int binIndex;
	floatVec3 *ver;
	int3 Tr;
	double size[3], d[3];

	size[0] = Box.Ver2.x - Box.Ver1.x;
	size[1] = Box.Ver2.y - Box.Ver1.y;
	size[2] = Box.Ver2.z - Box.Ver1.z;
	d[0] = size[0] / BIN_COUNT;
	d[1] = size[1] / BIN_COUNT;
	d[2] = size[2] / BIN_COUNT;
	
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
				if (c < Box.Ver1.cell[axis])
					binIndex = 0;
				else {
					binIndex = (int) ((c - Box.Ver1.cell[axis]) / d[axis]);
					if (binIndex >= BIN_COUNT) binIndex = BIN_COUNT-1;
				}
			} else binIndex = 0;
			Bins[axis][binIndex]++;
		}
	}
	
	//подсчет SAH на границах корзин
	double sah;
	double leftArea, rightArea, tmpSize;
	double invArea = 0.5 / (size[0]*size[1] + size[1]*size[2] + size[0]*size[2]); 

	min_sah = MAX_DISTANCE;
	
	for (int axis = 0; axis < 3; axis++)
		if (d[axis] > EPSILON)
		{
			leftCount = 0;
			rightCount = numberOfTriangles;
			tmpSize = size[axis];
			for (int i = 1; i < BIN_COUNT; i++)
			{
				leftCount += Bins[axis][i-1];
				rightCount -= Bins[axis][i-1];
				size[axis] = i*d[axis];
				leftArea = 2.0 * (size[0]*size[1] + size[1]*size[2] + size[2]*size[0]);
				size[axis] = tmpSize - size[axis];
				rightArea = 2.0 * (size[0]*size[1] + size[1]*size[2] + size[2]*size[0]);
				sah = invArea*(leftCount*leftArea + rightArea*rightCount);
				if (sah < min_sah)
				{
					min_sah = sah;
					splitAxis = axis;
					split = Box.Ver1.cell[axis] + i*d[axis];
				}
			}
			size[axis] = tmpSize;
		}
	
	leftCount = 0;
	rightCount = 0;
	if (min_sah < MAX_DISTANCE)
		for (int i = 0; i < numberOfTriangles; i++)
		{
			ver = surfaces[triangles[i].surfId].vertices;
			Tr = surfaces[triangles[i].surfId].triangles[triangles[i].trIndex];
			if (ver[Tr.x].cell[splitAxis] < split || ver[Tr.y].cell[splitAxis] < split || ver[Tr.z].cell[splitAxis] < split)
				leftCount++;
			if (ver[Tr.x].cell[splitAxis] > split || ver[Tr.y].cell[splitAxis] > split || ver[Tr.z].cell[splitAxis] > split)
				rightCount++;
			if (ver[Tr.x].cell[splitAxis] == split && ver[Tr.y].cell[splitAxis] == split && ver[Tr.z].cell[splitAxis] == split)
				leftCount++;
		}
}

KdTree* GenerateKdNode_bins(KdTree* tree, Surface* surfaces, int depth)
{
	if (tree->numOfTriangles <= MAX_TRIANGLES || depth == MAX_DEPTH)
	{
		tree->leftNode = 0;
		tree->rightNode = 0;
		return tree;
	}
	int splitAxis, leftCount = 0, rightCount = 0;
	double split, sah, min_sah;
	FindKdSplit_bins(tree->Box, tree->triangles, tree->numOfTriangles, surfaces, min_sah, split, splitAxis, leftCount, rightCount);
	sah = tree->numOfTriangles;
	if (sah <= min_sah || (leftCount == tree->numOfTriangles && rightCount == tree->numOfTriangles))
	{
		tree->leftNode = 0;
		tree->rightNode = 0;
		return tree;
	}
	floatVec3 *ver;
	int3 tr;

	KdTree *left = new KdTree();
	left->numOfTriangles = leftCount;
	left->Box = tree->Box;
	left->Box.Ver2.cell[splitAxis] = split;
	left->triangles = new Triangle[leftCount];

	KdTree *right = new KdTree();
	right->numOfTriangles = rightCount;
	right->Box = tree->Box;
	right->Box.Ver1.cell[splitAxis] = split;
	right->triangles = new Triangle[rightCount];

	int j = 0, k = 0;
	for (int i = 0; i < tree->numOfTriangles; i++)
	{
		ver = surfaces[tree->triangles[i].surfId].vertices;
		tr = surfaces[tree->triangles[i].surfId].triangles[tree->triangles[i].trIndex];
		if (ver[tr.x].cell[splitAxis] < split || ver[tr.y].cell[splitAxis] < split || ver[tr.z].cell[splitAxis] < split)
			left->triangles[j++] = tree->triangles[i];
		if (ver[tr.x].cell[splitAxis] > split || ver[tr.y].cell[splitAxis] > split || ver[tr.z].cell[splitAxis] > split)
			right->triangles[k++] = tree->triangles[i];
		if (ver[tr.x].cell[splitAxis] == split && ver[tr.y].cell[splitAxis] == split && ver[tr.z].cell[splitAxis] == split) 
			left->triangles[j++] = tree->triangles[i];
	}
	delete[] tree->triangles;
	tree->triangles = 0;

	tree->splitAxis = splitAxis;
	tree->splitPos = split;
	GenerateKdNode_bins(left, surfaces, depth+1);
	GenerateKdNode_bins(right, surfaces, depth+1);	
	tree->leftNode = left;
	tree->rightNode = right;
	return tree;
}


void FindKdSplit_sort(AABB Box, Triangle *triangles, int numberOfTriangles, Surface* surfaces, double &min_sah, double &split, int &splitAxis)
{
	floatVec3 size;
	for (int axis = 0; axis < 3; axis++)
		size.cell[axis] = Box.Ver2.cell[axis] - Box.Ver1.cell[axis];
	double* Maxs = new double[numberOfTriangles];
	double* Mins = new double[numberOfTriangles];
	floatVec3 *ver;
	int3 tr;
	double sah, leftArea, rightArea, tmpSize, lastMax, lastMin;
	int leftCount, rightCount;
	double invArea = 0.5 / (size.x*size.y + size.y*size.z + size.x*size.z); 

	min_sah = MAX_DISTANCE;
	for (int axis = 0; axis < 3; axis++)
		if (size.cell[axis] > EPSILON)
		{
			for (int i = 0; i < numberOfTriangles; i++)
			{
				ver = surfaces[triangles[i].surfId].vertices;
				tr = surfaces[triangles[i].surfId].triangles[triangles[i].trIndex];
				Mins[i] = Maxs[i] = ver[tr.x].cell[axis];
				if (ver[tr.y].cell[axis] > Maxs[i])
					Maxs[i] = ver[tr.y].cell[axis];
				if (ver[tr.z].cell[axis] > Maxs[i])
					Maxs[i] = ver[tr.z].cell[axis];
				if (ver[tr.y].cell[axis] < Mins[i])
					Mins[i] = ver[tr.y].cell[axis];
				if (ver[tr.z].cell[axis] < Mins[i])
					Mins[i] = ver[tr.z].cell[axis];
			}
			sort(Maxs, Maxs+numberOfTriangles-1);
			leftCount = numberOfTriangles;
			rightCount = 0;
			tmpSize = size.cell[axis];
			lastMax = Maxs[numberOfTriangles-1];
			for (int j = numberOfTriangles-2; j >= 0 && Maxs[j] > Box.Ver1.cell[axis]; j--)
			{
				if (Maxs[j] < lastMax && Maxs[j] < Box.Ver2.cell[axis])
				{
					lastMax = Maxs[j];
					leftCount = j+1;
					rightCount = numberOfTriangles - leftCount;
					size.cell[axis] = Maxs[j] - Box.Ver1.cell[axis];
					leftArea = 2.0 * (size.x*size.y + size.y*size.z + size.x*size.z);
					size.cell[axis] = Box.Ver2.cell[axis] - Maxs[j];
					rightArea = 2.0 * (size.x*size.y + size.y*size.z + size.x*size.z);
					sah = invArea*(leftCount*leftArea + rightArea*rightCount);
					if (sah < min_sah)
					{
						min_sah = sah;
						split = Maxs[j];
						splitAxis = axis;
					}
				}
			}
			sort(Mins, Mins+numberOfTriangles-1);
			leftCount = 0; 
			rightCount = numberOfTriangles;
			lastMin = Mins[0];
			for (int j = 1; j < numberOfTriangles && Mins[j] < Box.Ver2.cell[axis]; j++)
			{
				if (Mins[j] > lastMin && Mins[j] > Box.Ver1.cell[axis])
				{
					lastMin = Mins[j];
					rightCount = numberOfTriangles - j;
					leftCount = j;
					size.cell[axis] = Mins[j] - Box.Ver1.cell[axis];
					leftArea = 2.0 * (size.x*size.y + size.y*size.z + size.x*size.z);
					size.cell[axis] = Box.Ver2.cell[axis] - Mins[j];
					rightArea = 2.0 * (size.x*size.y + size.y*size.z + size.x*size.z);
					sah = invArea*(leftCount*leftArea + rightArea*rightCount);
					if (sah < min_sah)
					{
						min_sah = sah;
						split = Mins[j];
						splitAxis = axis;
					}
				}
			}
			size.cell[axis] = tmpSize;
		}
	delete[] Maxs;
	delete[] Mins;
}

KdTree* GenerateKdNode_sort(KdTree* tree, Surface* surfaces, int depth)
{
	if (tree->numOfTriangles <= MAX_TRIANGLES || depth == MAX_DEPTH)
	{
		tree->leftNode = 0;
		tree->rightNode = 0;
		return tree;
	}
	int splitAxis;
	double split, sah, min_sah;
	FindKdSplit_sort(tree->Box, tree->triangles, tree->numOfTriangles, surfaces, min_sah, split, splitAxis);
	sah = tree->numOfTriangles;
	if (sah <= min_sah)
	{
		tree->leftNode = 0;
		tree->rightNode = 0;
		return tree;
	}
	int leftCount = 0, rightCount = 0;
	floatVec3 *ver;
	int3 tr;
	char* splitArray = new char[tree->numOfTriangles];
	for (int i = 0; i < tree->numOfTriangles; i++)
	{
		ver = surfaces[tree->triangles[i].surfId].vertices;
		tr = surfaces[tree->triangles[i].surfId].triangles[tree->triangles[i].trIndex];
		splitArray[i] = 0;
		if (ver[tr.x].cell[splitAxis] < split || ver[tr.y].cell[splitAxis] < split || ver[tr.z].cell[splitAxis] < split)
		{
			leftCount++;
			splitArray[i]++;
		}
		if (ver[tr.x].cell[splitAxis] > split || ver[tr.y].cell[splitAxis] > split || ver[tr.z].cell[splitAxis] > split)
		{
			rightCount++;
			splitArray[i] += 2;
		}
		if (ver[tr.x].cell[splitAxis] == split && ver[tr.y].cell[splitAxis] == split && ver[tr.z].cell[splitAxis] == split)
		{
			leftCount++; 
			splitArray[i] = 1;
		}
	}
	Triangle *leftTris = new Triangle[leftCount];
	Triangle *rightTris = new Triangle[rightCount];
	int j = 0, k = 0;
	for (int i = 0; i < tree->numOfTriangles; i++)
	{
		if (splitArray[i] == 1 || splitArray[i] == 3)
			leftTris[j++] = tree->triangles[i];
		if (splitArray[i] >= 2)
			rightTris[k++] = tree->triangles[i];
	}
	delete[] splitArray;
	delete[] tree->triangles;
	tree->triangles = 0;

	tree->splitAxis = splitAxis;
	tree->splitPos = split;

	KdTree *left = new KdTree();
	KdTree *right = new KdTree();
	left->numOfTriangles = leftCount;
	left->triangles = leftTris;
	left->Box = tree->Box;
	left->Box.Ver2.cell[splitAxis] = split;
	GenerateKdNode_sort(left, surfaces, depth+1);

	right->numOfTriangles = rightCount;
	right->triangles = rightTris;
	right->Box = tree->Box;
	right->Box.Ver1.cell[splitAxis] = split;
	GenerateKdNode_sort(right, surfaces, depth+1);
	
	tree->leftNode = left;
	tree->rightNode = right;
	return tree;
}

KdTree* GenerateKdTree(Surface* surfaces, int numberOfSurfaces)
{
	KdTree* tree = new KdTree();
	int i = 0, j = 0;
	for(i = 0; i < numberOfSurfaces; i++)
		tree->numOfTriangles += surfaces[i].numberOfTriangles;
	tree->triangles = new Triangle[tree->numOfTriangles];
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
	//общий ограничивающий бокс
	tree->Box = createBox(tree->triangles, tree->numOfTriangles, surfaces);
	//GenerateKdNode_bins(tree, surfaces, 0);
	GenerateKdNode_sort(tree, surfaces, 0);
	return tree;
}

IntersectionInfo ComputeKDIntersectionWithoutStep(floatVec3 origin, floatVec3 direction, KdTree* tree, Surface* surfaces)
{
	IntersectionInfo result;
	result.isFindIntersection = 0;
	result.distance = MAX_DISTANCE;

	KdTree* curr = tree;
	double t_near, t_far, t_split, t;
	bool isIntersectBox = IntersectAABB(curr->Box, origin, direction, t_near, t_far);
	if(!isIntersectBox)
		return result;

	stack<KdBoxIntersectionInfo, list<KdBoxIntersectionInfo>> kdStack;
	KdBoxIntersectionInfo info(curr, t_far);
	floatVec3 *ver;
	int3 tr;
	int tr_index;
	const int left_or_right[3] = { (direction.x >= 0)?1:0,  
								   (direction.y >= 0)?1:0,  
								   (direction.z >= 0)?1:0};

	while(true)
	{
		while(curr->triangles == 0)
		{
			int axis = curr->splitAxis;
			t_split = (curr->splitPos - origin.cell[axis]) / direction.cell[axis];
			KdTree* nearNode = (left_or_right[axis] == 1) ? curr->leftNode : curr->rightNode;
			KdTree* farNode = (left_or_right[axis] == 1) ? curr->rightNode : curr->leftNode;
	
			if(t_split <= t_near)
			{
				curr = farNode;
			}
			else if(t_split >= t_far)
			{
				curr = nearNode;
			}
			else
			{
				kdStack.push(KdBoxIntersectionInfo(farNode,t_far));
				curr = nearNode;
				t_far = t_split;
			}
		}

		tr_index = -1;
		for (int i = 0; i < curr->numOfTriangles; i++)
		{
			ver = surfaces[curr->triangles[i].surfId].vertices;
			tr = surfaces[curr->triangles[i].surfId].triangles[curr->triangles[i].trIndex];
			t = GetTriangleIntersectionDistance(origin, direction, ver[tr.x], ver[tr.y], ver[tr.z]);
			if (t >= 0.0 && t < result.distance)
			{
				result.distance = t;
				tr_index = i;
			}
		}
		if (tr_index >= 0)
		{
			result.surfaceId = curr->triangles[tr_index].surfId;
			ver = surfaces[result.surfaceId ].vertices;
			tr = surfaces[result.surfaceId].triangles[curr->triangles[tr_index].trIndex];
			result.isFindIntersection = 1;
			result.normal = GetPlaneNormal(ver[tr.x], ver[tr.y], ver[tr.z]);
			result.normal = NormalizeVector(result.normal);

			if (result.distance <= t_far)
				return result; 
		}
		if (kdStack.empty())
			return result;
		t_near = t_far;
		curr = kdStack.top().node;
		t_far = kdStack.top().tfar;
		kdStack.pop();
	}

	return result;
}

IntersectionInfo ComputeKDIntersection(floatVec3 origin, floatVec3 direction, double step, KdTree* tree, Surface* surfaces)
{
	IntersectionInfo result;
	result.isFindIntersection = 0;
	result.distance = MAX_DISTANCE;

	KdTree* curr = tree;
	double t_near, t_far, t_split, t;
	bool isIntersectBox = IntersectAABB(curr->Box, origin, direction, t_near, t_far);
	if(!isIntersectBox || step < t_near)
		return result;

	stack<KdBoxIntersectionInfo, list<KdBoxIntersectionInfo>> kdStack;
	KdBoxIntersectionInfo info(curr, t_far);
	floatVec3 *ver;
	int3 tr;
	int tr_index;
	const int left_or_right[3] = { (direction.x >= 0)?1:0,  
								   (direction.y >= 0)?1:0,  
								   (direction.z >= 0)?1:0};

	while(true)
	{
		while(curr->triangles == 0)
		{
			int axis = curr->splitAxis;
			t_split = (curr->splitPos - origin.cell[axis]) / direction.cell[axis];
			KdTree* nearNode = (left_or_right[axis] == 1) ? curr->leftNode : curr->rightNode;
			KdTree* farNode = (left_or_right[axis] == 1) ? curr->rightNode : curr->leftNode;
	
			if(t_split <= t_near)
			{
				curr = farNode;
			}
			else if(t_split >= t_far)
			{
				curr = nearNode;
			}
			else
			{
				if (t_split <=  step)
					kdStack.push(KdBoxIntersectionInfo(farNode,t_far));
				curr = nearNode;
				t_far = t_split;
			}
		}

		tr_index = -1;
		for (int i = 0; i < curr->numOfTriangles; i++)
		{
			ver = surfaces[curr->triangles[i].surfId].vertices;
			tr = surfaces[curr->triangles[i].surfId].triangles[curr->triangles[i].trIndex];
			t = GetTriangleIntersectionDistance(origin, direction, ver[tr.x], ver[tr.y], ver[tr.z]);
			if (t >= 0.0 && t < result.distance)
			{
				result.distance = t;
				tr_index = i;
			}
		}
		if (tr_index >= 0 && result.distance <= step)
		{
			result.surfaceId = curr->triangles[tr_index].surfId;
			ver = surfaces[result.surfaceId ].vertices;
			tr = surfaces[result.surfaceId].triangles[curr->triangles[tr_index].trIndex];
			result.isFindIntersection = 1;
			result.normal = GetPlaneNormal(ver[tr.x], ver[tr.y], ver[tr.z]);
			result.normal = NormalizeVector(result.normal);

			if (result.distance <= t_far)
				return result; 
		}
		if (kdStack.empty())
			return result;
		t_near = t_far;
		if (step < t_near)
			return result;
		curr = kdStack.top().node;
		t_far = kdStack.top().tfar;
		kdStack.pop();
	}

	return result;
}