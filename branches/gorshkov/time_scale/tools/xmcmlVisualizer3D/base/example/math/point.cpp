
#include "math.h"
#include <stdio.h>
	
/*-----------------------------------------------------------------------------
	POINT.CPP
-----------------------------------------------------------------------------*/

const mpoint mpoint::kOrigin(0,0,0);


bool mpoint::isEqual( const mpoint &other, float eps ) const
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

float mpoint::distanceSqr( const mpoint &other ) const
{
	float dx = x-other.x;
	float dy = y-other.y;
	float dz = z-other.z;
	return dx*dx + dy*dy + dz*dz;
}


float mpoint::distance( const mpoint &other ) const
{
	return sqrt( distanceSqr(other) );
}


mpoint	mpoint::rotate( const mquaternion &quat ) const
{
	mquaternion qi = quat.Inverse();
	mquaternion qv ( x, y, z, 0 );

	mquaternion res = (quat*qv )*qi;

	return mpoint( res.x, res.y, res.z );
}


mpoint mpoint::transform( const mmatrix &xform ) const
{
	mpoint	temp;
	temp.v[0] =  xform(0,0)*v[0]  +  xform(1,0)*v[1]  +  xform(2,0)*v[2]  +  xform(3,0)*v[3];
	temp.v[1] =  xform(0,1)*v[0]  +  xform(1,1)*v[1]  +  xform(2,1)*v[2]  +  xform(3,1)*v[3];
	temp.v[2] =  xform(0,2)*v[0]  +  xform(1,2)*v[1]  +  xform(2,2)*v[2]  +  xform(3,2)*v[3];
	temp.v[3] =  xform(0,3)*v[0]  +  xform(1,3)*v[1]  +  xform(2,3)*v[2]  +  xform(3,3)*v[3];
	temp.v[0] /= temp.v[3];
	temp.v[1] /= temp.v[3];
	temp.v[2] /= temp.v[3];
	temp.v[3] = 1;
	return temp;
}


mpoint mpoint::lerpTo( const mpoint &target, float factor ) const
{
	return mpoint (	x*(1-factor) +	target.x*factor,
					y*(1-factor) +	target.y*factor,
					z*(1-factor) +	target.z*factor );
}


mpoint mpoint::lerpAlong( const mvector &direction, float factor ) const
{
	return mpoint( x + direction.x*factor, y + direction.y*factor, z + direction.z*factor );
}


mpoint mpoint::FromString( const char *str )
{
	mpoint p(0,0,0);
	sscanf(str, "%f%f%f", &p.x, &p.y, &p.z);
	return p;
}