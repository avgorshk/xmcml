
#pragma once
	
/*-----------------------------------------------------------------------------
	TRANSFORM.H
-----------------------------------------------------------------------------*/

class	mtransform {
 	public:
						mtransform			( void ) : position(mpoint::kOrigin), orient(mquaternion::kIdentity) {}
						mtransform			( const mpoint &pos, const mquaternion &orient=mquaternion::kIdentity ) : position(pos), orient(orient) {}
						~mtransform			( void ) {}	  

		mpoint			getPosition			( void ) const { return position; }
		mquaternion		getOrient			( void ) const { return orient; }
		void			setPosition			( const mpoint &pos	)			{ this->position = pos;		}
		void			setOrient			( const mquaternion	&orient )	{ this->orient   = orient;	}

		void			setPose				( const mpoint &pos, const mquaternion &orient=mquaternion::kIdentity );
		void			getPose				( mpoint &pos, mquaternion &orient ) const;
		
		mmatrix			computMatrix		( void ) const;
										
		mvector			getVector			( const mvector &local_vector ) const;
		mpoint			getPoint			( const mpoint &local_point ) const;
		
		mvector			getLocalVector		( const mvector &global_vector ) const;
		mpoint			getLocalPoint		( const mpoint &global_point ) const;
		
		mtransform		reverseTransform	( void ) const;
		
	public:
	
		static	mtransform	fromAnglesZB	( const mpoint &origin, float yaw, float pitch, float roll );
		static	mtransform	fromAnglesXF	( const mpoint &origin, float yaw, float pitch, float roll );
		static	mtransform	lookAtZB		( const mpoint &origin, const mpoint &target, const mvector &up );
		static	mtransform	lookAtXF		( const mpoint &origin, const mpoint &target, const mvector &up );
		static	mtransform	addTransforms	( const mtransform &parent, const mtransform &child );
		
	protected:
		mpoint		position;
		mquaternion	orient;
	};


//
// ETransform::SetPose()
//
inline void mtransform::setPose( const mpoint &pos, const mquaternion &orient/*=mquaternion::kIdentity*/ )
{
	this->position = pos;		
	this->orient   = orient;	
}


//
// ETransform::GetPose()
//
inline void mtransform::getPose( mpoint &pos, mquaternion &orient ) const
{
	pos		= this->position;		
	orient  = this->orient;
}


//
// ETransform::Computmmatrix()
//
inline mmatrix mtransform::computMatrix( void ) const
{
	mmatrix	T	=	mmatrix::Translate( position.x, position.y, position.z );

	float	wx, wy, wz;
	float	xx, yy, yz;
	float	xy, xz, zz;
	float	x2, y2, z2;

	x2 = orient.x + orient.x;
	y2 = orient.y + orient.y;
	z2 = orient.z + orient.z;

	xx = orient.x * x2;
	xy = orient.x * y2;
	xz = orient.x * z2;

	yy = orient.y * y2;
	yz = orient.y * z2;
	zz = orient.z * z2;

	wx = orient.w * x2;
	wy = orient.w * y2;
	wz = orient.w * z2;

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


//
// ETransform::GetVector()
//
inline mvector mtransform::getVector( const mvector &local_vector ) const
{
	return local_vector.rotate(orient);
}


//
// ETransform::GetPoint()
//
inline mpoint mtransform::getPoint( const mpoint &local_point ) const
{
	return local_point.rotate(orient) + mvector(position);
}

//
// ETransform::GetLocalVector()
//
inline mvector mtransform::getLocalVector( const mvector &global_vector ) const
{
	return global_vector.rotate(orient.Conjugate());
}


//
// ETransform::GetLocalPoint()
//
inline mpoint mtransform::getLocalPoint( const mpoint &global_point ) const
{
	return (global_point - mvector(position)).rotate(orient.Conjugate());
}


//
// ETransform::ReverseTransform()
//
inline mtransform mtransform::reverseTransform( void ) const
{
	mquaternion inv_q = orient.Inverse();
	mpoint		inv_p = mpoint(-position.x, -position.y, -position.z);

	inv_p = inv_p.rotate(inv_q);
	
	return mtransform(inv_p, inv_q);
}
