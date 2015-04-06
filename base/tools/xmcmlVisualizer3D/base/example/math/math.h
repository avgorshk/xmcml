
#pragma once

#include <math.h>
#include <memory.h>
#include <stdlib.h>

class	mvector;
class	mpoint;
class	mtex_coord;
class	mmatrix;
class	mquaternion;

#include "point.h"
#include "vector.h"
#include "quaternion.h"
#include "color.h"
#include "texcoord.h"
#include "matrix.h"

#include "point_imp.h"
#include "vector_imp.h"
#include "texcoord_imp.h"
#include "color_imp.h"

#include "transform.h"


/*-----------------------------------------------------------------------------
	MATH :
-----------------------------------------------------------------------------*/

class mmath {
	public:
		static float		rad		( float a )	{ return a/57.295779513260928129089516840402f; }
		static float		deg		( float a )	{ return a*57.295779513260928129089516840402f; }
		static float		sqr		( float a ) { return a*a; }
		static float		sign	( float a ) { return (a >= 0) ? 1.0f : -1.0f; }
		static float		frac	( float a );
		static float		randf	( float a=0, float b=1 );
		static void			sincos	( float a, float &s, float &c ) { s = sin(a); c = cos(a); }
		
		static float		dot		( const mvector &a, const mvector &b ) { return a.dot( b ); }
		static mvector		cross	( const mvector &a, const mvector &b ) { return a.cross( b ); }
		static bool			compare	( const mvector	  &a, const mvector	  &b, float eps ) { return a.isEqual(b, eps); }
		static bool			compare	( const mpoint	  &a, const mpoint	  &b, float eps ) { return a.isEqual(b, eps); }
		static bool			compare	( const mcolor	  &a, const mcolor	  &b, float eps ) { return a.isEqual(b, eps); }
		static bool			compare	( const mtex_coord &a, const mtex_coord &b, float eps ) { return a.isEqual(b, eps); }
		static float		lerp	( float a, float b, float factor ) { return a * (1-factor) + b * factor; }
		static mvector		lerp	( const mvector   &a, const mvector   &b, float factor ) { return a.lerp(b, factor); }
		static mpoint		lerp	( const mpoint	  &a, const mpoint	  &b, float factor ) { return a.lerpTo(b, factor); }
		static mcolor		lerp	( const mcolor	  &a, const mcolor	  &b, float factor ) { return a.lerpTo(b, factor); }
		static mquaternion	slerp	( const mquaternion &a, const mquaternion &b, float factor ) { return a.SLerp(b, factor); }
		static float		clamp	( float a, float min, float max ) { if (a<min) return min; if (a>max) return max; return a; }
	};
	
	
inline mvector	operator - ( const mpoint &a, const mpoint &b ) { return mvector(b, a); }
	
/*-----------------------------------------------------------------------------
	MATH :
-----------------------------------------------------------------------------*/

//
//	EMath::Randf
//
inline float mmath::randf( float a/* =0 */, float b/* =1 */ )
{
	float fr = rand() / (float)RAND_MAX;
	return a + fr * (b-a);
}


inline float mmath::frac( float a )
{
	double dummy;
	return modf(a, &dummy);
}	
	

template<class Type> Type clamp(Type a, Type low, Type high)
{
	if (a > high) a = high;
	if (a < low) a = low;
	return a;
}


#undef PI
const float PI	=	3.14159265358f;

const float MAX_FLOAT_VALUE	=	3.40E38f;
const float MIN_FLOAT_VALUE	=	1.18E-38f;
const float	FLOAT_EPSILON	=	1.192092896e-07f;	//	smallest positive number such that 1.0+FLT_EPSILON != 1.0


template<class Type> void Swap(Type &a, Type &b)
{
	Type t = a;
	a = b;
	b = t;
}

bool Math_RayTriangleIntersection( mpoint &result, float &fraction, const mpoint &origin, const mvector &dir, 
												const mpoint &v0,  const mpoint &v1,  const mpoint &v2 );



