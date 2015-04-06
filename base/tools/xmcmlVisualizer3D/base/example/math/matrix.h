
#pragma once
	
/*-----------------------------------------------------------------------------
	MATRIX.H
-----------------------------------------------------------------------------*/

#define MATRIX_INVERSE_EPSILON		1e-14

class mmatrix	{
	public:
						mmatrix			( void );
						mmatrix			( const mmatrix & other );
						mmatrix			(	float a00, float a01, float a02, float a03, 
											float a10, float a11, float a12, float a13,
											float a20, float a21, float a22, float a23,
											float a30, float a31, float a32, float a33 );
		explicit		mmatrix			( const float *floatptr );
						
		float			&operator	()	( unsigned int row, unsigned int col );
		float			operator	()	( unsigned int row, unsigned int col ) const;

		float			*Ptr			( void );
		const	float	*Ptr			( void ) const;
		
		mmatrix			&operator	+=	( const mmatrix &other );
		mmatrix			&operator	-=	( const mmatrix &other );
		mmatrix			&operator	*=	( const mmatrix &other );
		
		mmatrix			operator	+	( const mmatrix &other ) const;
		mmatrix			operator	-	( const mmatrix &other ) const;
		mmatrix			operator	*	( const mmatrix &other ) const;
		mmatrix			operator	*	( float s ) const;
		mmatrix			operator	/	( float s ) const;
					
		mmatrix			&operator	*=	( float s );
		mmatrix			&operator	/=	( float s );
		
		bool			operator	==	( const mmatrix &other ) const;
		bool			operator	!=	( const mmatrix &other ) const;
		
		bool			IsEqual			( const mmatrix &other, float eps ) const;

		mmatrix			&SetIdentity		( void );
		
		mmatrix			Inverse			( void ) const; 
		mmatrix			Transpose		( void ) const;
		mmatrix			AffineInverse	( void ) const; 
		
		void			ToAngles		( float &yaw, float &pitch, float &roll ) const;
		void			ToPose			( mpoint &origin, mquaternion &orient ) const;
		
	public:
		static const	mmatrix			kIdentity;
		
	public:
		static mmatrix	FromAngles			( float yaw, float pitch, float roll );
		static mmatrix	FromPose			( const mpoint &origin, const mquaternion &orient );	
											
		static mmatrix	Translate			( float x, float y, float z );
		static mmatrix	Translate			( const mvector &dir );
		static mmatrix	Scale				( float x, float y, float z );
		static mmatrix	RotateX				( float a );
		static mmatrix	RotateY				( float a );
		static mmatrix	RotateZ				( float a );
		static mmatrix	PerspectiveRH		( float w, float h, float zn, float zf );
		static mmatrix	PerspectiveLH		( float w, float h, float zn, float zf );
		static mmatrix	PerspectiveFovRH	( float fovy, float aspect, float zn, float zf );	
		static mmatrix	PerspectiveFovLH	( float fovy, float aspect, float zn, float zf );	
		static mmatrix	OrthoRH				( float w, float h, float zn, float zf );
		static mmatrix	OrthoLH				( float w, float h, float zn, float zf );
		static mmatrix	OrthoOffCenterRH	( float l, float r, float b, float t, float zn, float zf );	
		static mmatrix	LookAtRH			( const mpoint &eye, const mpoint &at, const mvector &up );	
	
	protected:
		float data[4][4];
	};

/*-----------------------------------------------------------------------------
	MATRIX IMPLEMENTATION :
-----------------------------------------------------------------------------*/

inline mmatrix::mmatrix( void )
{
	SetIdentity();
}


inline mmatrix::mmatrix( const mmatrix & other )
{
	*this = other;
	//memcpy(data, other.Ptr(), sizeof(data));
}


inline mmatrix::mmatrix( float a00, float a01, float a02, float a03, 
				  float a10, float a11, float a12, float a13, 
				  float a20, float a21, float a22, float a23, 
				  float a30, float a31, float a32, float a33 )
{
	data[0][0] = a00;	data[0][1] = a01;	data[0][2] = a02,	data[0][3] = a03;
	data[1][0] = a10;	data[1][1] = a11;	data[1][2] = a12,	data[1][3] = a13;
	data[2][0] = a20;	data[2][1] = a21;	data[2][2] = a22,	data[2][3] = a23;
	data[3][0] = a30;	data[3][1] = a31;	data[3][2] = a32,	data[3][3] = a33;
}


inline mmatrix::mmatrix( const float *floatptr )
{
	memcpy(data, floatptr, sizeof(data));
}


inline float& mmatrix::operator()( unsigned int row, unsigned int col )
{
	return data[row][col];
}


inline float mmatrix::operator()( unsigned int row, unsigned int col ) const
{
	return data[row][col];
}


inline float* mmatrix::Ptr( void )
{
	return &data[0][0];
}


inline const float* mmatrix::Ptr( void ) const
{
	return &data[0][0];
}


inline mmatrix& mmatrix::operator+=( const mmatrix &other )
{
	for (int i=0; i<16; i++)	Ptr()[i] += other.Ptr()[i];
	return *this;
}

inline mmatrix& mmatrix::operator-=( const mmatrix &other )
{
	for (int i=0; i<16; i++)	Ptr()[i] -= other.Ptr()[i];
	return *this;
}

inline mmatrix& mmatrix::operator*=( const mmatrix &other )
{
	mmatrix temp	 = *this;

	for (register int i=0; i<4; i++)
		for (register int j=0; j<4; j++)
		{
			temp(i,j) = 
				data[i][0] * other(0,j) +
				data[i][1] * other(1,j) +
				data[i][2] * other(2,j) +
				data[i][3] * other(3,j);
		}

	*this	=	temp;

	return *this;
}

inline mmatrix& mmatrix::operator*=( float s )
{
	for (int i=0; i<16; i++)	Ptr()[i] *= s;
	return *this;
}

inline mmatrix mmatrix::operator+( const mmatrix &other ) const
{
	mmatrix temp(*this);
	temp += other;
	return temp;
}

inline mmatrix mmatrix::operator-( const mmatrix &other ) const
{
	mmatrix temp(*this);
	temp -= other;
	return temp;
}

inline mmatrix mmatrix::operator*( const mmatrix &other ) const
{
	mmatrix temp(*this);
	temp *= other;
	return temp;
}

inline mmatrix mmatrix::operator*( float s ) const
{
	mmatrix temp(*this);
	temp *= s;
	return temp;
}

inline mmatrix mmatrix::operator/( float s ) const
{
	mmatrix temp(*this);
	temp /= s;
	return temp;
}

inline mmatrix& mmatrix::operator/=( float s )
{
	for (int i=0; i<16; i++)	Ptr()[i] /= s;
	return *this;
}

inline bool mmatrix::operator==( const mmatrix &other ) const
{
	if (memcmp(this, &other, sizeof(mmatrix))==0) {
		return true;
	} else {
		return false;
	}
}

inline bool mmatrix::operator!=( const mmatrix &other ) const
{
	return !(*this==other);
}
