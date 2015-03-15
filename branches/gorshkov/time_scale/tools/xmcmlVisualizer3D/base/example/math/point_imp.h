
#pragma once

/*-----------------------------------------------------------------------------
	POINT IMPLEMENTATION :
-----------------------------------------------------------------------------*/

inline mpoint::mpoint( void ) : x(0), y(0), z(0), w(1)
{

}


inline mpoint::mpoint( float xx, float yy, float zz, float ww ) : x(xx/ww), y(yy/ww), z(zz/ww), w(1)
{

}


inline mpoint::mpoint( const mpoint &other ) : x(other.x), y(other.y), z(other.z), w(1)
{

}


inline mpoint::mpoint( const mvector &vector ) : x(vector.x), y(vector.y), z(vector.z), w(1)
{

}


inline mpoint::mpoint( const float *floatptr ) : x(floatptr[0]), y(floatptr[1]), z(floatptr[2]), w(1)
{

}


inline float* mpoint::ptr( void )
{
	return v;
}


inline const float* mpoint::ptr( void ) const
{
	return v;
}


inline mpoint& mpoint::operator=( const mpoint &src )
{
	x = src.x;
	y = src.y;
	z = src.z;
	return *this;
}


inline mpoint mpoint::operator+( const mvector &other ) const
{
	return mpoint( x + other.x, y + other.y, z + other.z );
}


inline mpoint mpoint::operator-( const mvector &other ) const
{
	return mpoint( x - other.x, y - other.y, z - other.z );
}


inline mpoint& mpoint::operator+=( const mvector &other )
{
	x += other.x;
	y += other.y;
	z += other.z;
	return *this;
}


inline mpoint& mpoint::operator-=( const mvector &other )
{
	x -= other.x;
	y -= other.y;
	z -= other.z;
	return *this;
}


inline mpoint mpoint::operator*( const float scale ) const
{
	return mpoint( x*scale, y*scale, z*scale );
}


inline mpoint mpoint::operator/( const float scale ) const
{
	return mpoint( x/scale, y/scale, z/scale );
}


inline bool mpoint::operator==( const mpoint &other ) const
{
	return (x==other.x && 
			y==other.y &&
			z==other.z);
}


inline bool mpoint::operator!=( const mpoint &other ) const
{
	return !(*this==other);
}
