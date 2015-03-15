
#pragma once

/*-----------------------------------------------------------------------------
	TEXCOORD_IMP.H :
-----------------------------------------------------------------------------*/

inline mtex_coord::mtex_coord( void )
{
	u=0; v=0;
}


inline mtex_coord::mtex_coord( float uu, float vv )
{
	u=uu; v=vv;
}


inline mtex_coord::mtex_coord( const mtex_coord &other )
{
	u	=	other.u;
	v	=	other.v;
}


inline mtex_coord::mtex_coord( const float *floatptr )
{
	u	=	floatptr[0];
	v	=	floatptr[1];
}


inline float * mtex_coord::ptr( void )
{
	return &u;
}


inline const float	* mtex_coord::ptr( void ) const
{
	return &u;
}


inline mtex_coord& mtex_coord::operator=( const mtex_coord &src )
{
	u	=	src.u;
	v	=	src.v;
	return *this;
}


inline bool mtex_coord::isEqual( const mtex_coord &other, float eps ) const
{
	if ( fabs(u - other.u) > eps ) {
		return false;
	}
	if ( fabs(v - other.v) > eps ) {
		return false;
	}
	return true;
}


inline mtex_coord mtex_coord::operator+( const mvector &other ) const
{
	return mtex_coord( u + other.x, v + other.y );
}


inline mtex_coord mtex_coord::operator-( const mvector &other ) const
{
	return mtex_coord( u - other.x, v - other.y );
}


inline mtex_coord mtex_coord::operator*( const float scale ) const
{
	return mtex_coord( u * scale, v * scale );
}


inline mtex_coord mtex_coord::operator/( const float scale ) const
{
	return mtex_coord( u / scale, v / scale );
}


inline bool mtex_coord::operator==( const mtex_coord &other ) const
{
	return ( u==other.u && v==other.v );
}


inline bool mtex_coord::operator!=( const mtex_coord &other ) const
{
	return !(*this==other);
}