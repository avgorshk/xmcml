
#include "math.h"
#include <stdio.h>
	
/*-----------------------------------------------------------------------------
	QUATERNION.CPP
-----------------------------------------------------------------------------*/

const mquaternion mquaternion::kIdentity = mquaternion(0,0,0,1);


bool mquaternion::IsEqual ( const mquaternion &other, float eps ) const
{
	if ( fabs(x - other.x) > eps ) {
		return false;
	}
	if ( fabs(y - other.y) > eps ) {
		return false;
	}
	if ( fabs(z - other.z) > eps ) {
		return false;
	}
	if ( fabs(w - other.w) > eps ) {
		return false;
	}
	return true;
}

mquaternion& mquaternion::SetIdentity( void )
{
	x = y = z = 0;
	w = 1;
	return *this;
}

mquaternion mquaternion::Conjugate( void ) const
{
	return mquaternion(-x, -y, -z, w);
}

float mquaternion::Dot( const mquaternion &q ) const
{
	return	x * q.x
		+	y * q.y
		+	z * q.z
		+	w * q.w ;
}

mquaternion mquaternion::Inverse( void ) const
{
	return Conjugate() / LengthSqr();
}

float mquaternion::Length( void ) const
{
	return sqrt( Dot(*this) );
}

float mquaternion::LengthSqr( void ) const
{
	return Dot(*this);
}

mquaternion mquaternion::Normalize( void ) const
{
	float len = Length();
	if (len!=0) {
		return mquaternion(	x/len, y/len, z/len, w/len );
	}
	return *this;
}

//
//	QuatSLerp
//	this function works with non-normalized quaternions,
//	and always returns normalized vector.
//
//	iq = (q*sin((1-t)*omega) + q'*sin(t*omega))/sin(omega),
//	where cos(omega) = inner_product(q,q') 
//
mquaternion mquaternion::SLerp( const mquaternion &target, float factor ) const
{
	mquaternion	from = *this;
	mquaternion	to	= target;

	const float DELTA = 0.01f;

	float p1[4];
	double omega, cosom, sinom, scale0, scale1;

	//	cosinus :
	cosom = x*target.x + y*target.y + z*target.z + w*target.w;

	if ( cosom < 0.0 ) { 
		cosom = -cosom;
		p1[0] = - target.x;  p1[1] = - target.y;
		p1[2] = - target.z;  p1[3] = - target.w;
	}
	else {
		p1[0] = target.x;    p1[1] = target.y;
		p1[2] = target.z;    p1[3] = target.w;
	}

	//	angle is bigger than delta :
	if ( (1.0 - cosom) > DELTA ) {
		omega = acos(cosom);
		sinom = sin(omega);
		scale0 = sin((1.0 - factor) * omega) / sinom;
		scale1 = sin(factor * omega) / sinom;
	} else {        
		scale0 = 1.0 - factor;
		scale1 = factor;
	}

	mquaternion	result(
		(float)( scale0 * x + scale1 * p1[0] ),
		(float)( scale0 * y + scale1 * p1[1] ),
		(float)( scale0 * z + scale1 * p1[2] ),
		(float)( scale0 * w + scale1 * p1[3] ) );

	return result;
}

mmatrix mquaternion::ToMatrix( void ) const
{
	float	wx, wy, wz;
	float	xx, yy, yz;
	float	xy, xz, zz;
	float	x2, y2, z2;

	x2 = x + x;
	y2 = y + y;
	z2 = z + z;

	xx = x * x2;
	xy = x * y2;
	xz = x * z2;

	yy = y * y2;
	yz = y * z2;
	zz = z * z2;

	wx = w * x2;
	wy = w * y2;
	wz = w * z2;

	mmatrix m;

	m(0,0) = 1.0f - (yy + zz);
	m(1,0) = xy - wz;
	m(2,0) = xz + wy;

	m(0,1) = xy + wz;
	m(1,1) = 1.0f - (xx + zz);
	m(2,1) = yz - wx;

	m(0,2) = xz - wy;
	m(1,2) = yz + wx;
	m(2,2) = 1.0f - (xx + yy);


	return m;
}


void mquaternion::ToAngles( float &yaw, float &pitch, float &roll ) const
{
	mmatrix	M	=	ToMatrix();
	M.ToAngles( yaw, pitch, roll );
}


void mquaternion::ToAnglesRad( float &yaw, float &pitch, float &roll ) const
{
	mmatrix	M	=	ToMatrix();
	M.ToAngles( yaw, pitch, roll );

	yaw		=	mmath::rad(yaw);
	pitch	=	mmath::rad(pitch);
	roll	=	mmath::rad(roll);
}


mquaternion mquaternion::fromMatrix( const mmatrix &xform )
{
	mmatrix	mat	=	xform.Transpose();

	mquaternion	q;
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
		s = 0.5f / sqrtf( t );

		q.v[3] = s * t;
		q.v[0] = ( mat(2,1) - mat(1,2) ) * s;
		q.v[1] = ( mat(0,2) - mat(2,0) ) * s;
		q.v[2] = ( mat(1,0) - mat(0,1) ) * s;

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

		q.v[i] = s * t;
		q.v[3] = ( mat(k,j) - mat(j,k) ) * s;
		q.v[j] = ( mat(j,i) + mat(i,j) ) * s;
		q.v[k] = ( mat(k,i) + mat(i,k) ) * s;
	}
	return q;
}


mquaternion mquaternion::fromAngles( float yaw, float pitch, float roll )
{
	float	rad_yaw		=	mmath::rad( yaw	  );
	float	rad_pitch	=	mmath::rad( pitch );
	float	rad_roll	=	mmath::rad( roll  );
	return fromAnglesRad( rad_yaw, rad_pitch, rad_roll );
}


mquaternion mquaternion::fromAnglesRad( float yaw, float pitch, float roll )
{
	//	TODO : optimize QuatFromAnglesRad
	mquaternion	qrx		=	rotateAroundAxis( (roll),	mvector(1,0,0) );
	mquaternion	qry		=	rotateAroundAxis( (pitch),	mvector(0,1,0) );
	mquaternion	qrz		=	rotateAroundAxis( (yaw),	mvector(0,0,1) );
	mquaternion	q		=	qrz * qry * qrx;

	return q;
}


mquaternion mquaternion::rotateAroundAxis( float angle, const mvector &axis )
{
	//	to return normalized quaternion - normalize axis vector :
	mvector	naxis = axis.normalize();

	float s = sin(angle/2);
	float c = cos(angle/2);

	return mquaternion( naxis.x*s, naxis.y*s, naxis.z*s, c );
}


mquaternion mquaternion::fromString( const char *str )
{
	mquaternion p(0,0,0,1);
	sscanf(str, "%f%f%f%f", &p.x, &p.y, &p.z, &p.w);
	return p;
}