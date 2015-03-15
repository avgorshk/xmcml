
#pragma once
	
/*-----------------------------------------------------------------------------
	POINT.H
-----------------------------------------------------------------------------*/

class mpoint {
	public:
					mpoint			( void );
					mpoint			( float xx, float yy, float zz=0, float ww=1.0f );
					mpoint			( const mpoint &other );
		explicit	mpoint			( const float *floatptr );
		explicit	mpoint			( const mvector &vector );
					
		float		*ptr			( void );
		const float	*ptr			( void ) const;
	
		mpoint&		operator =		( const mpoint &src );
		
		mpoint		operator +		( const mvector &other ) const;
		mpoint		operator -		( const mvector &other ) const;
		mpoint&		operator +=		( const mvector &other );
		mpoint&		operator -=		( const mvector &other );
		mpoint		operator *		( const float	scale ) const;
		mpoint		operator /		( const float	scale ) const;
		bool		operator ==		( const mpoint &other ) const;
		bool		operator !=		( const mpoint &other ) const;

		bool		isEqual			( const mpoint &other, float eps ) const;
		float		distanceSqr		( const mpoint &other ) const;
		float		distance		( const mpoint &other ) const;
		mpoint		rotate			( const mquaternion &q ) const;
		
		mpoint		transform		( const mmatrix &xform ) const;
		mpoint		lerpTo			( const mpoint &target, float factor ) const;
		mpoint		lerpAlong		( const mvector &direction, float factor ) const;

	public:
		static const mpoint	kOrigin;
		
		static	mpoint	FromString	( const char *str );
		
	public:
		union {
			float v[4];
			struct { float x, y, z, w; };
		};
	};
