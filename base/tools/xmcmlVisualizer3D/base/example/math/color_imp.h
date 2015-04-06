/*
	The MIT License

	Copyright (c) 2010 IFMO/GameDev Studio

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
*/

#pragma once

/*-----------------------------------------------------------------------------
	COLOR.H
-----------------------------------------------------------------------------*/


inline bool mcolor::isEqual( const mcolor &other, float eps ) const
{
	if ( fabs(r - other.r) > eps ) {
		return false;
	}
	if ( fabs(g - other.g) > eps ) {
		return false;
	}
	if ( fabs(b - other.b) > eps ) {
		return false;
	}
	if ( fabs(a - other.a) > eps ) {
		return false;
	}
	return true;
}


inline mcolor mcolor::lerpTo( const mcolor &target, float factor ) const
{
	return mcolor (	r*(1-factor) +	target.r*factor,
					g*(1-factor) +	target.g*factor,
					b*(1-factor) +	target.b*factor,
					a*(1-factor) +	target.a*factor );
}


inline mcolor mcolor::transform( const mmatrix &xform ) const
{
	mcolor	temp;
	temp.v[0] =  xform(0,0)*v[0]  +  xform(1,0)*v[1]  +  xform(2,0)*v[2]  +  xform(3,0)*v[3];
	temp.v[1] =  xform(0,1)*v[0]  +  xform(1,1)*v[1]  +  xform(2,1)*v[2]  +  xform(3,1)*v[3];
	temp.v[2] =  xform(0,2)*v[0]  +  xform(1,2)*v[1]  +  xform(2,2)*v[2]  +  xform(3,2)*v[3];
	temp.v[3] =  xform(0,3)*v[0]  +  xform(1,3)*v[1]  +  xform(2,3)*v[2]  +  xform(3,3)*v[3];
	return temp;
}

