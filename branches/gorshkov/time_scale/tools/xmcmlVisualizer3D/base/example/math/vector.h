
#pragma once
	
/*-----------------------------------------------------------------------------
	VECTOR.H
-----------------------------------------------------------------------------*/

class mvector {
	public:
					mvector			( void );
					mvector			( float xx, float yy, float zz=0 );
					mvector			( const mvector &other );
		explicit	mvector			( const float *floatptr );
		explicit	mvector			( const mpoint &point );
					mvector			( const mpoint &a, const mpoint &b );
					mvector			( const mtex_coord &a, const mtex_coord &b );
					
		float		*Ptr			( void );
		const float	*Ptr			( void ) const;
	
		mvector&	operator =		( const mvector &src );

		mvector		operator -		( void ) const;		
		mvector		operator +		( const mvector &other ) const;
		mvector		operator -		( const mvector &other ) const;
		mvector		operator *		( const float	scale ) const;
		mvector		operator /		( const float	scale ) const;
		bool		operator ==		( const mvector &other ) const;
		bool		operator !=		( const mvector &other ) const;

		bool		isEqual			( const mvector &other, float eps ) const;		
		
		float		length			( void ) const;
		float		lengthSqr		( void ) const;
		mvector		normalize		( void ) const;
		mvector		&normalizeSelf	( void );
		mvector		transform		( const mmatrix &xform ) const;
		mvector		rotate			( const mquaternion &quat ) const;
		mvector		lerp			( const mvector &target, float factor ) const;
		
		float		dot				( const mvector &other ) const;
		mvector		cross			( const mvector &other ) const;

	public:
		static const mvector	kZero;
		static const mvector	kOne;
		static const mvector	kX;
		static const mvector	kY;
		static const mvector	kZ;
		static const mvector	kNegX;
		static const mvector	kNegY;
		static const mvector	kNegZ;

		static	mvector	fromString	( const char *str );
	
	public:
		union {
			float v[4];
			struct { float x, y, z, w; };
		};
	};