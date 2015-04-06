
#pragma once

/*-----------------------------------------------------------------------------
	VECTOR IMPLEMENTATION :
-----------------------------------------------------------------------------*/

inline mvector::mvector( void ) : x(0), y(0), z(0), w(0)
{

}


inline mvector::mvector( float xx, float yy, float zz ) : x(xx), y(yy), z(zz), w(0)
{

}


inline mvector::mvector( const mvector &other ) : x(other.x), y(other.y), z(other.z), w(0)
{

}


inline mvector::mvector( const mpoint &point ) : x(point.x), y(point.y), z(point.z), w(0)
{

}


inline mvector::mvector( const float *floatptr ) : x(floatptr[0]), y(floatptr[1]), z(floatptr[2]), w(0)
{

}


inline mvector::mvector( const mpoint &a, const mpoint &b ) : x(b.x-a.x), y(b.y-a.y), z(b.z-a.z), w(0)
{

}


inline mvector::mvector( const mtex_coord &a, const mtex_coord &b ) : x(b.x-a.x), y(b.y-a.y), z(0), w(0)
{

}


inline float* mvector::Ptr( void )
{
	return v;
}


inline const float* mvector::Ptr( void ) const
{
	return v;
}


inline mvector& mvector::operator=( const mvector &src )
{
	x = src.x;
	y = src.y;
	z = src.z;
	return *this;
}


inline mvector mvector::operator-( void ) const
{
	return mvector(-x, -y, -z);
}


inline mvector mvector::operator+( const mvector &other ) const
{
	return mvector( x + other.x, y + other.y, z + other.z );
}


inline mvector mvector::operator-( const mvector &other ) const
{
	return mvector( x - other.x, y - other.y, z - other.z );
}


inline mvector mvector::operator*( const float scale ) const
{
	return mvector( x*scale, y*scale, z*scale );
}


inline mvector mvector::operator/( const float scale ) const
{
	return mvector( x/scale, y/scale, z/scale );
}


inline bool mvector::operator==( const mvector &other ) const
{
	return (x==other.x && 
			y==other.y &&
			z==other.z);
}


inline bool mvector::operator!=( const mvector &other ) const
{
	return !(*this==other);
}
