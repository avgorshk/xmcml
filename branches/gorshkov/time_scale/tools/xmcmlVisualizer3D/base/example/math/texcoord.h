
#pragma once

/*-----------------------------------------------------------------------------
	TEXCOORD.H :
-----------------------------------------------------------------------------*/

class mtex_coord	{
	public:
					mtex_coord		( void );
					mtex_coord		( float uu, float vv );
					mtex_coord		( const mtex_coord &other );
		explicit	mtex_coord		( const float *floatptr );

		float		*ptr			( void );
		const float	*ptr			( void ) const;
	
		mtex_coord&	operator =		( const mtex_coord &src );

		mtex_coord	operator +		( const mvector &other ) const;
		mtex_coord	operator -		( const mvector &other ) const;
		mtex_coord	operator *		( const float	scale ) const;
		mtex_coord	operator /		( const float	scale ) const;
		bool		operator ==		( const mtex_coord &other ) const;
		bool		operator !=		( const mtex_coord &other ) const;

		bool		isEqual			( const mtex_coord &other, float eps ) const;
		mtex_coord	lerpTo			( const mtex_coord &target, float frac ) const;
		
	public:
		static mtex_coord	fromString		( const char *text );
		static mtex_coord	lerp			( const mtex_coord &a, const mtex_coord &b, float factor );
		
	protected:
		union {
			float data[2];
			struct { float x, y; };
			struct { float u, v; };
		};
	};
	
	
/*-----------------------------------------------------------------------------
	IMPLEMENTATION :
-----------------------------------------------------------------------------*/

	