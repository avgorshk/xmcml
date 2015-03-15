
#include "math.h"
#include <stdio.h>

/*-----------------------------------------------------------------------------
	TEXCOORD.CPP :
-----------------------------------------------------------------------------*/

mtex_coord mtex_coord::fromString( const char *text )
{
	float u, v;
	sscanf( text, "%f%f", &u, &v );
	return mtex_coord( u, v );
}


mtex_coord mtex_coord::lerpTo( const mtex_coord &target, float factor ) const
{
	return mtex_coord (	x*(1-factor) +	target.x*factor,
						y*(1-factor) +	target.y*factor );
}


mtex_coord mtex_coord::lerp( const mtex_coord &a, const mtex_coord &b, float factor )
{
	return a.lerpTo( b, factor );
}