
#include "math.h"
#include <stdio.h>

	
/*-----------------------------------------------------------------------------
	COLOR.CPP
-----------------------------------------------------------------------------*/

const mcolor	mcolor::kWhite	=	mcolor(1,1,1,1);
const mcolor	mcolor::kBlack	=	mcolor(0,0,0,1);
const mcolor	mcolor::kRed	=	mcolor(1,0,0,1);
const mcolor	mcolor::kGreen	=	mcolor(0,1,0,1);
const mcolor	mcolor::kBlue	=	mcolor(0,0,1,1);


mcolor mcolor::saturate( unsigned int value ) const
{
	//COLORREF color = RGB( r, g, b );

	// Convert to HLS Color space
	unsigned int hue		= 0;
	unsigned int luminance	= 0;
	unsigned int saturation = 0;
	//ColorRGBToHLS( color, &hue, &luminance, &saturation );

	saturation = clamp<unsigned int>( value, 1, 240 );

	// Now convert back to RGB.
	//color = ColorHLSToRGB( hue, luminance, saturation );

	//return EColor( GetRValue(color), GetGValue(color), GetBValue(color), a );
	return mcolor();
}


float mcolor::luminance( void ) const
{
	//COLORREF color = RGB( r, g, b );

	// Convert to HLS Color space
	unsigned int hue		= 0;
	float luminance	= 0;
	unsigned int saturation = 0;
	//ColorRGBToHLS( color, &hue, &luminance, &saturation );

	return luminance;
}



mcolor mcolor::fromString( const char *text )
{
	mcolor c(0,0,0,0);
	sscanf(text, "%f%f%f%f", &c.x, &c.y, &c.z, &c.w);
	return c;
}