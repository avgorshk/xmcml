
#pragma once
	
/*-----------------------------------------------------------------------------
	QUATERNION.H
-----------------------------------------------------------------------------*/

class mquaternion {
	public:
						mquaternion		( void );
						mquaternion		( float xx, float yy, float zz, float ww );
		explicit		mquaternion		( const float *floatptr );

		float			*Ptr			( void )		{ return &x; }
		const	float	*Ptr			( void ) const	{ return &x; }

		mquaternion&	operator +=		( const mquaternion &q );
		mquaternion&	operator -=		( const mquaternion &q );
		mquaternion&	operator *=		( const mquaternion &q );
		mquaternion&	operator /=		( const mquaternion &q );
		mquaternion&	operator *=		( float a);
		mquaternion&	operator /=		( float a);

		mquaternion		operator +		( void ) const;
		mquaternion		operator -		( void ) const;

		mquaternion		operator + 		( const mquaternion &q ) const;
		mquaternion		operator - 		( const mquaternion &q ) const;
		mquaternion		operator * 		( const mquaternion &q ) const;
		mquaternion		operator / 		( const mquaternion &q ) const;
		mquaternion		operator * 		( float a ) const;
		mquaternion		operator / 		( float a ) const;

		bool			operator ==		( const mquaternion &q ) const;
		bool			operator !=		( const mquaternion &q ) const;
		
		bool			IsEqual			( const mquaternion &other, float eps ) const;

		mquaternion		&SetIdentity		( void );
		
		mquaternion		Conjugate		( void ) const;
		float			Dot				( const mquaternion &q ) const;
		mquaternion		Inverse			( void ) const;
		float			Length			( void ) const;
		float			LengthSqr		( void ) const;

		mquaternion		Normalize		( void ) const;
		mquaternion		SLerp			( const mquaternion &target, float factor ) const;
		
		mmatrix			ToMatrix		( void ) const;
		void			ToAngles		( float &yaw, float &pitch, float &roll ) const;
		void			ToAnglesRad		( float &yaw, float &pitch, float &roll ) const;

	public:
		static const	mquaternion		kIdentity;
		
	public:
		static mquaternion	fromMatrix			( const mmatrix &xform );
		static mquaternion	fromAngles			( float yaw, float pitch, float roll );
		static mquaternion	fromAnglesRad		( float yaw, float pitch, float roll );
		static mquaternion	fromString			( const char *str );
		
  		static mquaternion	rotateAroundAxis	( float angle, const mvector &axis );

	public:
		union {
			float v[4];
			struct { float x, y, z, w; };
		};
	};

/*-----------------------------------------------------------------------------
	QUATERNION IMPLEMENTATION :
-----------------------------------------------------------------------------*/

inline mquaternion::mquaternion( void )
{
	SetIdentity();
}


inline mquaternion::mquaternion( float xx, float yy, float zz, float ww ) : x(xx), y(yy), z(zz), w(ww)
{

}


inline mquaternion::mquaternion( const float *floatptr ) : x(floatptr[0]), y(floatptr[1]), z(floatptr[2]), w(floatptr[3])
{

}


inline mquaternion& mquaternion::operator+=( const mquaternion &q )
{
	x += q.x;
	y += q.y;
	z += q.z;
	w += q.w;
	return *this;
}


inline mquaternion& mquaternion::operator-=( const mquaternion &q )
{
	x -= q.x;
	y -= q.y;
	z -= q.z;
	w -= q.w;
	return *this;
}


inline mquaternion& mquaternion::operator*=( const mquaternion &q )
{
	*this = *this * q;
	return *this;
}


inline mquaternion& mquaternion::operator*=( float a)
{
	x *= a;
	y *= a;
	z *= a;
	w *= a;
	return *this;
}


inline mquaternion& mquaternion::operator/=( const mquaternion &q )
{
	*this = *this * q.Inverse();
	return *this;
}


inline mquaternion& mquaternion::operator/=( float a)
{
	x /= a;
	y /= a;
	z /= a;
	w /= a;
	return *this;
}


inline mquaternion mquaternion::operator+( void ) const
{
	return mquaternion(x,y,z,w);
}


inline mquaternion mquaternion::operator+( const mquaternion &q ) const
{
	mquaternion q1 = *this;
	return q1 += q;
}


inline mquaternion mquaternion::operator-( void ) const
{
	return mquaternion(-x,-y,-z,-w);
}


inline mquaternion mquaternion::operator-( const mquaternion &q ) const
{
	mquaternion q1 = *this;
	return q1 -= q;
}


inline mquaternion mquaternion::operator*( const mquaternion &q ) const
{
	return mquaternion(	w*q.x + x*q.w + y*q.z - z*q.y,
						w*q.y + y*q.w + z*q.x - x*q.z,
						w*q.z + z*q.w + x*q.y - y*q.x,
						w*q.w - x*q.x - y*q.y - z*q.z );
}


inline mquaternion mquaternion::operator*( float a ) const
{
	mquaternion q1 = *this;
	return q1 *= a;
}


inline mquaternion mquaternion::operator/( const mquaternion &q ) const
{
	return *this * q.Inverse();
}


inline mquaternion mquaternion::operator/( float a ) const
{
	return (*this) * (1.0f/a);
}


inline bool mquaternion::operator==( const mquaternion &q ) const
{
	return (x==q.x && y==q.y && z==q.z && w==q.w ); 
}


inline bool mquaternion::operator!=( const mquaternion &q ) const
{
	return !(*this==q);
}
