	
#pragma once

/*-----------------------------------------------------------------------------
	COLOR.H
-----------------------------------------------------------------------------*/

class mcolor {
	public:
					mcolor			( void );
					mcolor			( float rr, float gg, float bb, float aa=1.0f );
					mcolor			( const mcolor &other );
		explicit	mcolor			( const float *floatptr );
		
		float		*Ptr			( void );
		const float	*Ptr			( void ) const;
	
		mcolor&		operator =		( const mcolor &src );

		mcolor		operator -		( void ) const;		
		mcolor		operator +		( const mcolor &other ) const;
		mcolor		operator -		( const mcolor &other ) const;
		mcolor		operator *		( const float	scale ) const;
		mcolor		operator /		( const float	scale ) const;
		bool		operator ==		( const mcolor &other ) const;
		bool		operator !=		( const mcolor &other ) const;

		bool		isEqual			( const mcolor &other, float eps ) const;		
		mcolor		lerpTo			( const mcolor &target, float factor ) const;
		
		mcolor		transform		( const mmatrix &xform ) const;
		mcolor		saturate		( unsigned int value ) const;
		float		luminance		( void ) const;
		
	public:
		static const mcolor	kWhite;
		static const mcolor	kBlack;
		static const mcolor	kRed;
		static const mcolor	kGreen;
		static const mcolor	kBlue;
		
		static mcolor	fromString	( const char *text );
	
	public:
		union {
			float v[4];
			struct { float x, y, z, w; };
			struct { float r, g, b, a; };
		};
	};


/*-----------------------------------------------------------------------------
	COLOR IMPLEMENTATION :
-----------------------------------------------------------------------------*/


inline mcolor::mcolor( void ) : r(0), g(0), b(0), a(1)
{

}

inline mcolor::mcolor( float rr, float gg, float bb, float aa/*=1.0f */ ) : r(rr), g(gg), b(bb), a(aa)
{

}

inline mcolor::mcolor( const mcolor &other ) : r(other.r), g(other.g), b(other.b), a(other.a)
{

}

inline mcolor::mcolor( const float *floatptr ) : r(floatptr[0]), g(floatptr[1]), b(floatptr[2]), a(floatptr[3])
{

}

inline float* mcolor::Ptr( void )
{
	return v;
}

inline const float	* mcolor::Ptr( void ) const
{
	return v;
}

inline mcolor& mcolor::operator=( const mcolor &src )
{
	r = src.r;
	g = src.g;
	b = src.b;
	a = src.a;
	return *this;
}

inline mcolor mcolor::operator-( void ) const
{
	return mcolor( -r, -g, -b, -a );
}

inline mcolor mcolor::operator+( const mcolor &other ) const
{
	return mcolor( r + other.r, g + other.g, b + other.b, a + other.a );
}

inline mcolor mcolor::operator-( const mcolor &other ) const
{
	return mcolor( r - other.r, g - other.g, b - other.b, a - other.a );
}

inline mcolor mcolor::operator*( const float scale ) const
{
	return mcolor( r*scale, g*scale, b*scale, a*scale );
}

inline mcolor mcolor::operator/( const float scale ) const
{
	return mcolor( r/scale, g/scale, b/scale, a/scale );
}

inline bool mcolor::operator==( const mcolor &other ) const
{
	return (r==other.r && g==other.g && b==other.b && a==other.a );
}

inline bool mcolor::operator!=( const mcolor &other ) const
{
	return !(*this==other);
}