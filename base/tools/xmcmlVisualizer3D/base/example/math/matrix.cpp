
#include "math.h"
	
/*-----------------------------------------------------------------------------
	MATRIX.CPP
-----------------------------------------------------------------------------*/

bool mmatrix::IsEqual( const mmatrix &other, float eps ) const
{
	for (int i=0; i<16; i++) {
		if( fabs( Ptr()[i] - other.Ptr()[i]) > eps ) {
			return false;
		}
	}
	return true;
}


mmatrix& mmatrix::SetIdentity( void )
{
	data[0][0] = 1;	data[0][1] = 0;	data[0][2] = 0;	data[0][3] = 0;
	data[1][0] = 0;	data[1][1] = 1;	data[1][2] = 0;	data[1][3] = 0;
	data[2][0] = 0;	data[2][1] = 0;	data[2][2] = 1;	data[2][3] = 0;
	data[3][0] = 0;	data[3][1] = 0;	data[3][2] = 0;	data[3][3] = 1;
	return *this;
}


mmatrix mmatrix::Inverse( void ) const
{
	// 84+4+16 = 104 multiplications
	//			   1 division
	double det, invDet;

	mmatrix mat = *this;

	// 2x2 sub-determinants required to calculate 4x4 determinant
	float det2_01_01 = mat(0,0) * mat(1,1) - mat(0,1) * mat(1,0);
	float det2_01_02 = mat(0,0) * mat(1,2) - mat(0,2) * mat(1,0);
	float det2_01_03 = mat(0,0) * mat(1,3) - mat(0,3) * mat(1,0);
	float det2_01_12 = mat(0,1) * mat(1,2) - mat(0,2) * mat(1,1);
	float det2_01_13 = mat(0,1) * mat(1,3) - mat(0,3) * mat(1,1);
	float det2_01_23 = mat(0,2) * mat(1,3) - mat(0,3) * mat(1,2);

	// 3x3 sub-determinants required to calculate 4x4 determinant
	float det3_201_012 = mat(2,0) * det2_01_12 - mat(2,1) * det2_01_02 + mat(2,2) * det2_01_01;
	float det3_201_013 = mat(2,0) * det2_01_13 - mat(2,1) * det2_01_03 + mat(2,3) * det2_01_01;
	float det3_201_023 = mat(2,0) * det2_01_23 - mat(2,2) * det2_01_03 + mat(2,3) * det2_01_02;
	float det3_201_123 = mat(2,1) * det2_01_23 - mat(2,2) * det2_01_13 + mat(2,3) * det2_01_12;

	det = ( - det3_201_123 * mat(3,0) + det3_201_023 * mat(3,1) - det3_201_013 * mat(3,2) + det3_201_012 * mat(3,3) );

	if ( abs( det ) < MATRIX_INVERSE_EPSILON ) {
		return mat;
	}

	invDet = 1.0f / det;

	// remaining 2x2 sub-determinants
	float det2_03_01 = mat(0,0) * mat(3,1) - mat(0,1) * mat(3,0);
	float det2_03_02 = mat(0,0) * mat(3,2) - mat(0,2) * mat(3,0);
	float det2_03_03 = mat(0,0) * mat(3,3) - mat(0,3) * mat(3,0);
	float det2_03_12 = mat(0,1) * mat(3,2) - mat(0,2) * mat(3,1);
	float det2_03_13 = mat(0,1) * mat(3,3) - mat(0,3) * mat(3,1);
	float det2_03_23 = mat(0,2) * mat(3,3) - mat(0,3) * mat(3,2);

	float det2_13_01 = mat(1,0) * mat(3,1) - mat(1,1) * mat(3,0);
	float det2_13_02 = mat(1,0) * mat(3,2) - mat(1,2) * mat(3,0);
	float det2_13_03 = mat(1,0) * mat(3,3) - mat(1,3) * mat(3,0);
	float det2_13_12 = mat(1,1) * mat(3,2) - mat(1,2) * mat(3,1);
	float det2_13_13 = mat(1,1) * mat(3,3) - mat(1,3) * mat(3,1);
	float det2_13_23 = mat(1,2) * mat(3,3) - mat(1,3) * mat(3,2);

	// remaining 3x3 sub-determinants
	float det3_203_012 = mat(2,0) * det2_03_12 - mat(2,1) * det2_03_02 + mat(2,2) * det2_03_01;
	float det3_203_013 = mat(2,0) * det2_03_13 - mat(2,1) * det2_03_03 + mat(2,3) * det2_03_01;
	float det3_203_023 = mat(2,0) * det2_03_23 - mat(2,2) * det2_03_03 + mat(2,3) * det2_03_02;
	float det3_203_123 = mat(2,1) * det2_03_23 - mat(2,2) * det2_03_13 + mat(2,3) * det2_03_12;
																		  		
	float det3_213_012 = mat(2,0) * det2_13_12 - mat(2,1) * det2_13_02 + mat(2,2) * det2_13_01;
	float det3_213_013 = mat(2,0) * det2_13_13 - mat(2,1) * det2_13_03 + mat(2,3) * det2_13_01;
	float det3_213_023 = mat(2,0) * det2_13_23 - mat(2,2) * det2_13_03 + mat(2,3) * det2_13_02;
	float det3_213_123 = mat(2,1) * det2_13_23 - mat(2,2) * det2_13_13 + mat(2,3) * det2_13_12;
																		  		
	float det3_301_012 = mat(3,0) * det2_01_12 - mat(3,1) * det2_01_02 + mat(3,2) * det2_01_01;
	float det3_301_013 = mat(3,0) * det2_01_13 - mat(3,1) * det2_01_03 + mat(3,3) * det2_01_01;
	float det3_301_023 = mat(3,0) * det2_01_23 - mat(3,2) * det2_01_03 + mat(3,3) * det2_01_02;
	float det3_301_123 = mat(3,1) * det2_01_23 - mat(3,2) * det2_01_13 + mat(3,3) * det2_01_12;

	mat(0,0) = - (float) ( det3_213_123 * invDet );
	mat(1,0) = + (float) ( det3_213_023 * invDet );
	mat(2,0) = - (float) ( det3_213_013 * invDet );
	mat(3,0) = + (float) ( det3_213_012 * invDet );

	mat(0,1) = + (float) ( det3_203_123 * invDet );
	mat(1,1) = - (float) ( det3_203_023 * invDet );
	mat(2,1) = + (float) ( det3_203_013 * invDet );
	mat(3,1) = - (float) ( det3_203_012 * invDet );

	mat(0,2) = + (float) ( det3_301_123 * invDet );
	mat(1,2) = - (float) ( det3_301_023 * invDet );
	mat(2,2) = + (float) ( det3_301_013 * invDet );
	mat(3,2) = - (float) ( det3_301_012 * invDet );

	mat(0,3) = - (float) ( det3_201_123 * invDet );
	mat(1,3) = + (float) ( det3_201_023 * invDet );
	mat(2,3) = - (float) ( det3_201_013 * invDet );
	mat(3,3) = + (float) ( det3_201_012 * invDet );

	return mat;
}


mmatrix mmatrix::Transpose( void ) const
{
	mmatrix	temp(*this);

	Swap<float>( temp(0,1), temp(1,0) );
	Swap<float>( temp(0,2), temp(2,0) );
	Swap<float>( temp(0,3), temp(3,0) );
	Swap<float>( temp(2,1), temp(1,2) );
	Swap<float>( temp(3,1), temp(1,3) );
	Swap<float>( temp(3,2), temp(2,3) );

	return temp;
}


mmatrix mmatrix::AffineInverse( void ) const
{
	mmatrix	iM (
			data[0][0],		data[1][0],		data[2][0],		0,
			data[0][1],		data[1][1],		data[2][1],		0,
			data[0][2],		data[1][2],		data[2][2],		0,
					 0,				 0,				 0,		1
		);
		
	mpoint	o1( -data[3][0], -data[3][1], -data[3][2] );
	mpoint	o2	=	o1.transform(iM);
	iM(3,0)	= o2.v[0];
	iM(3,1)	= o2.v[1];
	iM(3,2)	= o2.v[2];
	return iM;
}


void mmatrix::ToAngles( float &yaw, float &pitch, float &roll ) const
{
	double		theta;
	double		cp;
	float		sp;

	sp = data[0][2];

	// cap off our sin value so that we don't get any NANs
	if ( sp > 1.0f ) {
		sp = 1.0f;
	} else if ( sp < -1.0f ) {
		sp = -1.0f;
	}

	theta = -asin( sp );
	cp = cos( theta );

	if ( cp > 8192.0f * FLOAT_EPSILON ) {
		pitch	= mmath::deg( (float)theta );
		yaw		= mmath::deg( atan2( data[0][1], data[0][0] ) );
		roll	= mmath::deg( atan2( data[1][2], data[2][2] ) );
	} else {
		pitch	= mmath::deg( (float)theta );
		yaw		= mmath::deg( -atan2( data[1][0], data[1][1] ) );
		roll	= 0;
	}
}


void mmatrix::ToPose( mpoint &origin, mquaternion &orient ) const
{
	origin.x	=	data[3][0];
	origin.y	=	data[3][1];
	origin.z	=	data[3][2];
	origin.w	=	data[3][3];

	mmatrix	mat	=	Transpose();

	float		trace;
	float		s;
	float		t;
	int     	i;
	int			j;
	int			k;

	static int 	next[ 3 ] = { 1, 2, 0 };

	trace = mat(0,0) + mat(1,1) + mat(2,2);

	if ( trace > 0.0f ) {

		t = trace + 1.0f;
		s = 0.5f / sqrt( t );

		orient.v[3] = s * t;
		orient.v[0] = ( mat(2,1) - mat(1,2) ) * s;
		orient.v[1] = ( mat(0,2) - mat(2,0) ) * s;
		orient.v[2] = ( mat(1,0) - mat(0,1) ) * s;

	} else {

		i = 0;
		if ( mat(1,1) > mat(0,0) ) {
			i = 1;
		}
		if ( mat(2,2) > mat(i,i) ) {
			i = 2;
		}
		j = next[i];
		k = next[j];

		t = ( mat(i,i) - ( mat(j,j) + mat(k,k) ) ) + 1.0f;
		s = 0.5f / sqrt( t );

		orient.v[i] = s * t;
		orient.v[3] = ( mat(k,j) - mat(j,k) ) * s;
		orient.v[j] = ( mat(j,i) + mat(i,j) ) * s;
		orient.v[k] = ( mat(k,i) + mat(i,k) ) * s;
	}
}


mmatrix mmatrix::FromAngles( float yaw, float pitch, float roll )
{
	mmatrix M;
	float sr, sp, sy, cr, cp, cy;

	mmath::sincos( mmath::rad( yaw ),	sy, cy );
	mmath::sincos( mmath::rad( pitch ),	sp, cp );
	mmath::sincos( mmath::rad( roll ),	sr, cr );

	M(0,0) = cp * cy;					M(0,1) = cp * sy, 					M(0,2) = - sp		;
	M(1,0) = sr * sp * cy + cr * -sy,	M(1,1) = sr * sp * sy + cr * cy,	M(1,2) =   sr * cp	;
	M(2,0) = cr * sp * cy + -sr * -sy,	M(2,1) = cr * sp * sy + -sr * cy,	M(2,2) =   cr * cp	;

	return M;
}


mmatrix mmatrix::FromPose( const mpoint &origin, const mquaternion &q )
{
	mmatrix	T	=	Translate( origin.x, origin.y, origin.z );

	float	wx, wy, wz;
	float	xx, yy, yz;
	float	xy, xz, zz;
	float	x2, y2, z2;

	x2 = q.x + q.x;
	y2 = q.y + q.y;
	z2 = q.z + q.z;

	xx = q.x * x2;
	xy = q.x * y2;
	xz = q.x * z2;

	yy = q.y * y2;
	yz = q.y * z2;
	zz = q.z * z2;

	wx = q.w * x2;
	wy = q.w * y2;
	wz = q.w * z2;

	mmatrix R;

    R(0,0) = 1.0f - (yy + zz);
    R(1,0) = xy - wz;
    R(2,0) = xz + wy;

    R(0,1) = xy + wz;
    R(1,1) = 1.0f - (xx + zz);
    R(2,1) = yz - wx;

    R(0,2) = xz - wy;
    R(1,2) = yz + wx;
    R(2,2) = 1.0f - (xx + yy);
    
	return R * T;
}


mmatrix mmatrix::Translate( float x, float y, float z )
{
	return	mmatrix(	1,	0,	0,	0,
						0,	1,	0,	0,
						0,	0,	1,	0,
						x,	y,	z,	1	);
}


mmatrix mmatrix::Translate( const mvector &dir )
{
	return Translate( dir.x, dir.y, dir.z );
}


mmatrix mmatrix::Scale( float x, float y, float z )
{
	return	mmatrix(	x,	0,	0,	0,
						0,	y,	0,	0,
						0,	0,	z,	0,
						0,	0,	0,	1	);
}


mmatrix mmatrix::RotateX( float a )
{
	float c	=	cosf(a);
	float s	=	sinf(a);
	return	mmatrix(	1,	0,	0,	0,
						0,	c,	s,	0,
						0,	-s,	c,	0,
						0,	0,	0,	1	);
}


mmatrix mmatrix::RotateY( float a )
{
	float c	=	cosf(a);
	float s	=	sinf(a);
	return	mmatrix(	c,	0,	-s,	0,
						0,	1,	0,	0,
						s,	0,	c,	0,
						0,	0,	0,	1	);
}


mmatrix mmatrix::RotateZ( float a )
{
	float c	=	cosf(a);
	float s	=	sinf(a);
	return	mmatrix(	c,	s,	0,	0,
						-s,	c,	0,	0,
						0,	0,	1,	0,
						0,	0,	0,	1	);
}


mmatrix mmatrix::PerspectiveRH( float w, float h, float zn, float zf )
{
	return	mmatrix(
					2*zn/w,  0,       0,              0,
					0,       2*zn/h,  0,              0,
					0,       0,       zf/(zn-zf),    -1,
					0,       0,       zn*zf/(zn-zf),  0 );
}


mmatrix mmatrix::PerspectiveLH( float w, float h, float zn, float zf )
{
	return mmatrix(
					2*zn/w,  0,       0,             0,
					0,       2*zn/h,  0,             0,
					0,       0,       zf/(zf-zn),    1,
					0,       0,       zn*zf/(zn-zf), 0 );
}


mmatrix mmatrix::PerspectiveFovRH( float fovy, float aspect, float zn, float zf )
{
	float y_scale	=	1.0f/tan(fovy/2.0f);
	float x_scale	=	aspect * y_scale;
	return mmatrix(
					x_scale, 0,       0,             0,
					0,       y_scale, 0,             0,
					0,       0,       zf/(zn-zf),    -1,
					0,       0,       zn*zf/(zn-zf), 0 );
}


mmatrix mmatrix::PerspectiveFovLH( float fovy, float aspect, float zn, float zf )
{
	float y_scale	=	1.0f/tan(fovy/2.0f);
	float x_scale	=	aspect * y_scale;
	return mmatrix(
					x_scale, 0,       0,             0,
					0,       y_scale, 0,             0,
					0,       0,       zf/(zf-zn),    1,
					0,       0,       zn*zf/(zn-zf), 0 );
}


mmatrix mmatrix::OrthoRH( float w, float h, float zn, float zf )
{
	return mmatrix
		(	2/w,	0,		0,			0,
			0,		2/h,	0,			0,
			0,		0,		1/(zn-zf),	0,
			0,		0,		zn/(zn-zf),	1	);
}


mmatrix mmatrix::OrthoLH( float w, float h, float zn, float zf )
{
	return mmatrix
		(	2/w,	0,		0,			0,
			0,		2/h,	0,			0,
			0,		0,		1/(zf-zn),	0,
			0,		0,		zn/(zn-zf),	1	);
}


mmatrix mmatrix::OrthoOffCenterRH( float l, float r, float b, float t, float zn, float zf )
{
	return mmatrix(
					2/(r-l)		,   0            ,  0           ,   0,
					0           ,	2/(t-b)      ,	0           ,	0,
					0           ,	0            ,	1/(zn-zf)   ,	0,
					(l+r)/(l-r) ,	(t+b)/(b-t)  ,	zn/(zn-zf)  ,	1 );
}


mmatrix mmatrix::LookAtRH( const mpoint &eye, const mpoint &at, const mvector &up )
{
	mvector	zaxis = mvector(at, eye); //eye - at;
	zaxis.normalizeSelf();
	mvector	xaxis = up.cross(zaxis);
	xaxis.normalizeSelf();
	mvector	yaxis = zaxis.cross(xaxis);

	return mmatrix (   
				 xaxis.x,		   yaxis.x,		   zaxis.x,    0,
				 xaxis.y,		   yaxis.y,	   	   zaxis.y,    0,
				 xaxis.z,		   yaxis.z,	 	   zaxis.z,	   0,
		 -xaxis.dot( mvector(eye.x, eye.y, eye.z) ),  -yaxis.dot( mvector(eye.x, eye.y, eye.z) ), -zaxis.dot( mvector(eye.x, eye.y, eye.z) ),   1 );
}
