
#include "math.h"
	
/*-----------------------------------------------------------------------------
	TRANSFORM.CPP
-----------------------------------------------------------------------------*/

//
// ETransform::FromAnglesZB()
//
mtransform mtransform::fromAnglesZB( const mpoint &origin, float yaw, float pitch, float roll )
{
	mquaternion	z_up	=	mquaternion::rotateAroundAxis( -PI/2.0f, mvector(0,0,1)) * mquaternion::rotateAroundAxis(PI/2.0f, mvector(1,0,0));
	mquaternion	qrx		=	mquaternion::rotateAroundAxis( mmath::rad(roll),	mvector(1,0,0) );
	mquaternion	qry		=	mquaternion::rotateAroundAxis( mmath::rad(pitch),	mvector(0,1,0) );
	mquaternion	qrz		=	mquaternion::rotateAroundAxis( mmath::rad(yaw),		mvector(0,0,1) );
	mquaternion	q		=	qrz * qry * qrx * z_up;
	//mpoint		p		=	mpoint(x, y, z);

	return mtransform(origin, q);
}


//
// ETransform::FromAnglesXF()
//
mtransform	mtransform::fromAnglesXF( const mpoint &origin, float yaw, float pitch, float roll )
{
	mquaternion	z_up	=	mquaternion::rotateAroundAxis( -PI/2.0f, mvector(0,0,1)) * mquaternion::rotateAroundAxis(PI/2.0f, mvector(1,0,0));
	mquaternion	qrx		=	mquaternion::rotateAroundAxis( mmath::rad(roll),	mvector(1,0,0) );
	mquaternion	qry		=	mquaternion::rotateAroundAxis( mmath::rad(pitch),	mvector(0,1,0) );
	mquaternion	qrz		=	mquaternion::rotateAroundAxis( mmath::rad(yaw),		mvector(0,0,1) );
	mquaternion	q		=	qrx * qry * qrz * z_up;
	//mpoint		p		=	mpoint(x, y, z);

	return mtransform(origin, q);
}


//
// ETransform::LookAtZB()
//
mtransform	mtransform::lookAtZB( const mpoint &eye, const mpoint &at, const mvector &up )
{
	mvector	zaxis = mvector(at, eye); //eye - at;
	zaxis.normalizeSelf();
	mvector	xaxis = up.cross(zaxis);
	xaxis.normalizeSelf();
	mvector	yaxis = zaxis.cross(xaxis);

	mmatrix mat = mmatrix (   
				 xaxis.x,		   yaxis.x,		   zaxis.x,    0,
				 xaxis.y,		   yaxis.y,	   	   zaxis.y,    0,
				 xaxis.z,		   yaxis.z,	 	   zaxis.z,	   0,
		 -xaxis.dot( mvector(eye.x, eye.y, eye.z) ),  -yaxis.dot( mvector(eye.x, eye.y, eye.z) ), -zaxis.dot( mvector(eye.x, eye.y, eye.z) ),   1 );

	mpoint		pos;
	mquaternion orient;

	mat.ToPose(pos, orient);

	return mtransform(pos, orient);
}


//
// ETransform::LookAtXF()
//
mtransform	mtransform::lookAtXF( const mpoint &origin, const mpoint &target, const mvector &up )
{
	return mtransform();
}


//
// ETransform::AddTransforms()
//
mtransform	mtransform::addTransforms( const mtransform &parent, const mtransform &child )
{
	return mtransform(child.position.rotate(parent.orient) + mvector(parent.position), parent.orient * child.orient);
}
