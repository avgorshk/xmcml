
#include "math.h"
#include <stdio.h>
	
/*-----------------------------------------------------------------------------
	VECTOR.CPP
-----------------------------------------------------------------------------*/

const mvector	mvector::kZero	=	mvector( 0,  0,  0 );
const mvector	mvector::kOne	=	mvector( 1,  1,  1 );
const mvector	mvector::kX		=	mvector( 1,  0,  0 );
const mvector	mvector::kY		=	mvector( 0,  1,  0 );
const mvector	mvector::kZ		=	mvector( 0,  0,  1 );
const mvector	mvector::kNegX	=	mvector(-1,  0,  0 );
const mvector	mvector::kNegY	=	mvector( 0, -1,  0 );
const mvector	mvector::kNegZ	=	mvector( 0,  0, -1 );


bool mvector::isEqual( const mvector &other, float eps ) const
{
	if ( fabs(x - other.x) > eps ) {
		return false;
	}
	if ( fabs(y - other.y) > eps ) {
		return false;
	}
	if ( fabs(z - other.z) > eps ) {
		return false;
	}
	return true;
}


float mvector::length( void ) const
{
	return sqrt(x*x + y*y + z*z);
}


float mvector::lengthSqr( void ) const
{
	return x*x + y*y + z*z;
}


mvector mvector::normalize( void ) const
{
	float len	=	length();
	if (len==0) return *this;
	return *this * (1.0f/len);
}


mvector& mvector::normalizeSelf( void )
{
	float len	=	length();
	if (len==0) return *this;
	*this = *this * (1.0f/len);
	return *this;
}


mvector mvector::transform( const mmatrix &xform ) const
{
	mvector	temp;
	temp.v[0] =  xform(0,0)*v[0]  +  xform(1,0)*v[1]  +  xform(2,0)*v[2];
	temp.v[1] =  xform(0,1)*v[0]  +  xform(1,1)*v[1]  +  xform(2,1)*v[2];
	temp.v[2] =  xform(0,2)*v[0]  +  xform(1,2)*v[1]  +  xform(2,2)*v[2];
	temp.v[3] =  0;
	return temp;
}


mvector mvector::rotate( const mquaternion &quat ) const
{
	mquaternion qi = quat.Inverse();
	mquaternion qv ( x, y, z, 0 );

	mquaternion res = (quat*qv )*qi;

	return mvector( res.x, res.y, res.z );
}


mvector mvector::lerp( const mvector &target, float factor ) const
{
	return mvector(	x*(1-factor) +	target.x*factor,
					y*(1-factor) +	target.y*factor,
					z*(1-factor) +	target.z*factor );
}


float mvector::dot( const mvector &other ) const
{
	return x * other.x  +  y * other.y  +  z * other.z;
}


mvector mvector::cross( const mvector &other ) const
{
	return mvector ( y*other.z - z*other.y,	
					 z*other.x - x*other.z,	
					 x*other.y - y*other.x );	
}


mvector mvector::fromString( const char *str )
{
	mvector v(0,0,0);
	sscanf(str, "%f%f%f", &v.x, &v.y, &v.z);
	return v;
}
