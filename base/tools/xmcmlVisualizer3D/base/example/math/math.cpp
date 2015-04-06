
#include "math.h"

/*-----------------------------------------------------------------------------
	MATH :
-----------------------------------------------------------------------------*/

//
//	Math_RayTriangleIntersection
//	
bool Math_RayTriangleIntersection(mpoint &result, float &fraction, const mpoint &origin, const mvector &dir, 
												const mpoint &v0,  const mpoint &v1,  const mpoint &v2 )
{
	const float EPS1	=	0.0001f;
	const float EPS		=	0;

	mvector	edge1, edge2;
	mvector	tvec, pvec, qvec;
	
	float	det;
					   
	//	find vectors for two sharing v0	:
	edge1	=	mvector( v0, v1 );
	edge2	=	mvector( v0, v2 );

	//	begin calculating determinant - also used for calculate U param :	
	pvec	=	mmath::cross( dir, edge2 );
	
	det		=	mmath::dot( edge1, pvec );

	//	if det is near zero - ray lies in plane of triangle :	
	if ( abs(det) < EPS1 ) {
		return false;
	}
	
	//	calculate distance from v0 to ray origin :
	tvec	=	mvector( v0, origin );

	float u	=	mmath::dot( tvec, pvec ) / det;
	if (u<0-EPS || u>1+EPS) {
		return false;
	}
	
	qvec	=	mmath::cross( tvec, edge1 );
	
	float v	=	mmath::dot( dir, qvec ) / det;
	if (v<0-EPS || u+v>1+EPS) {
		return false;
	}
	
	float t =	mmath::dot(edge2, qvec) / det;
	
	fraction	=	t;
	
	mvector	result_vector	=	mvector( v0 ) * (1-u-v) 
							+	mvector( v1 ) * u 
							+	mvector( v2 ) * v;
	
	result	=	mpoint( result_vector );
	
	return 1;

#if 0
	EVec3	v0v1	=	v1 - v0;
	EVec3	v0v2	=	v2 - v0;
	EVec3	v2v1	=	v1 - v2;
	EVec3	N		=	Vec3Normalize( Vec3Cross(v0v1, v0v2) );
	float	d		=	- Vec3Dot(N, v0);
	
	/*result = start + dir;
	fraction = 1;
	return true;*/

	
	float NdotD		=	Vec3Dot(N, dir);

	fraction = 0;	
	if (fabs(NdotD)<0.00001) return false;	// triangle and ray are parallel

	fraction = - ((d + Vec3Dot(N, start)) / NdotD);
	
	if ( fraction<0.0f ) return false;				// triangle is behind the ray
	if ( fraction>1.0f ) return false;				// ray can`t hit the triangle

	EVec3	P	=	start + dir * fraction;
	
	EVec3	v0P	=	P - v0;
	EVec3	v2P	=	P - v2;
	
	// stupid algorithm using cross products :
	float sign_v0v1 = Vec3Dot(N, Vec3Cross(v0v1, v0P));	// should be positive! (+)	(-)
	float sign_v0v2 = Vec3Dot(N, Vec3Cross(v0v2, v0P));	// should be negative! (-)
	float sign_v2v1 = Vec3Dot(N, Vec3Cross(v2v1, v2P));	// should be negative! (-)

	if (sign_v0v1>=0 && sign_v0v2<=0 && sign_v2v1<=0) {
		result = P;
		return true;
	} else {
	//if (sign_v0v1<=0 && sign_v0v2>=0 && sign_v2v1>=0) {
	//	result = P;
	//	return true;
	//} else {
		return false;
	}
#endif	
}

