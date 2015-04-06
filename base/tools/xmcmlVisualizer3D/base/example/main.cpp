
#include <assert.h>
#include <math.h>
#include <sstream>
#include <algorithm>
#include <iterator>

#ifdef WIN32
#include <Windows.h>
#endif

#include "GL/glew.h"
#include "GL/glut.h"

#include "base_demo.h"
#include "math/math.h"

const int MAX_PARTICLES = 600;
const int TRACE_SIZE = 60;

struct vec4 {
	float x,y,z,w;
};


mvector rand_radial_vector(float min_r, float max_r)
{
 	mvector v;
	float l;

	do {
		v.x = mmath::randf(-max_r, max_r);
		v.y = mmath::randf(-max_r, max_r);
		v.z = mmath::randf(-max_r, max_r);
		l = v.length();
	} while (l<min_r || l>max_r);

	return v;
}

/*-----------------------------------------------------------------------------
	class declaration :
-----------------------------------------------------------------------------*/

class test_demo : public base_demo {
	public:
						test_demo( int argc, char **argv, const char *demo_name );
		virtual			~test_demo( void );

		mpoint  traces[MAX_PARTICLES][TRACE_SIZE];
		mpoint  positions[MAX_PARTICLES];
		mvector velocities[MAX_PARTICLES];
		mcolor  colors[MAX_PARTICLES];
		float	temperature[MAX_PARTICLES];
		//mpoint	center = ;

		ui_float_control_exp	g_control;
		float					gravity;

		virtual	void	init();
		virtual void	update( float dt );
		virtual void	draw3d( float dt );
		virtual void	draw2d( float dt );
	};


/*-----------------------------------------------------------------------------
	class implementation :
-----------------------------------------------------------------------------*/

test_demo::test_demo( int argc, char **argv, const char *demo_name )
 : base_demo( argc, argv, demo_name ), 
 g_control("Gravity ", &gravity,    1.1f )
{
	gravity = 0.05;
}


test_demo::~test_demo()
{
}


void test_demo::init()
{
	add_control( &g_control );

	for (int i=0; i<MAX_PARTICLES; i++) {
		positions[i]	= mpoint( rand_radial_vector(4.0, 4.001f) );// + mvector(i>512?8:-8,0,0);
		velocities[i]	= mmath::cross( positions[i] - mpoint(0,0,0), rand_radial_vector(0.5,1) ).normalize() * 10;
		temperature[i]	= 0;
		colors[i]		= mcolor(1,0,0,1);
	}
}


void test_demo::update( float dt )
{
	dt = 0.001f;

//	center = kinect_wristL.lerpTo( kinect_wristR, 0.5f ) * 10;

	#pragma omp parallel num_threads(6)
	{
		#pragma omp for
 		for (int i=0; i<MAX_PARTICLES; i++) {

			mvector total_force(0,0,0);

			//	mutual forces :
			for (int j=0; j<MAX_PARTICLES; j++) {	

				if (i==j) continue;

				mvector  r = positions[j] - positions[i];
				float d = r.length() + 0.0001;
				r = r / d;

				total_force = total_force + ( r / d / d * gravity );// - r/d_sq * temperature[i];
			}

			//	center of the galaxy :
			mvector r = mpoint(0,0,0) - positions[i];
			float d = r.length();
			r = r / d;
			total_force = total_force + ( r / d / d * 1000 );// - r/d_sq * temperature[i];

			if (positions[i].distance( mpoint(0,0,0) ) < 1 ) {
				mvector r = positions[i] - mpoint(0,0,0);
				r.normalizeSelf();
				positions[i] = mpoint(0,0,0) + r * 1;
			}

			//	integrate velocities and positions :
			velocities[i] = velocities[i] + total_force * dt - velocities[i] * 0.01;

			positions[i]  = positions[i] + velocities[i] * dt;

			//	do trace :
			for (int k=TRACE_SIZE-1; k>0; k--) {
				traces[i][k] = traces[i][k-1];
			}
			traces[i][0] = positions[i];
		}
	}
}


void test_demo::draw3d( float dt )
{
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glPointSize(2);

	glEnable(GL_FOG);
	glFogi( GL_FOG_MODE, GL_EXP );
	glFogf( GL_FOG_DENSITY, 0.02f );

	//glEnable(GL_POINT_SMOOTH);

	glBegin( GL_POINTS );


		for (int i=0; i<MAX_PARTICLES; i++) {

			glColor4f( 0.9, 0.9, 1.0, 1.0 );
			glVertex3fv( positions[i].ptr() );
			
			for (int k=0; k<TRACE_SIZE; k++) {
				mpoint pos= traces[i][k];
				float f = (1-k/(float)TRACE_SIZE)*0.1;

				glColor4f( 0.8*f, 0.9*f, 1.0*f, 1.0 );
				glVertex3fv( pos.ptr() );
			}
		}

	glEnd();


	float size = 128;

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	glBegin( GL_LINES );

		for ( float t = -size; t<=size; t+=2 ) {		
			glColor4f( 0.4f, 0.4f, 0.4f, 1.0f );
			glVertex3f( t, -size, -4 );
			glVertex3f( t, +size, -4 );
			glVertex3f( -size, t, -4 );
			glVertex3f( +size, t, -4 );
		}
		for ( float t = -size; t<=size; t+=16 ) {		
			glColor4f( 0.4f, 0.4f, 0.4f, 1.0f );
			glVertex3f( t, -size, -4 );
			glVertex3f( t, +size, -4 );
			glVertex3f( -size, t, -4 );
			glVertex3f( +size, t, -4 );
		}

	glEnd();


	glBegin(GL_LINES);

		glColor4f (1,1,0,1);

		/*glVertex3fv( (center + mvector( 1, 0, 0)).ptr() );
		glVertex3fv( (center + mvector(-1, 0, 0)).ptr() );
		glVertex3fv( (center + mvector( 0, 1, 0)).ptr() );
		glVertex3fv( (center + mvector( 0,-1, 0)).ptr() );
		glVertex3fv( (center + mvector( 0, 0, 1)).ptr() );
		glVertex3fv( (center + mvector( 0, 0,-1)).ptr() ); */

		float xsz = 16;
		float ysz = 16;
		float zsz = 16;

		glColor4f (1,0,0,1);
		glVertex3f(0,0,0);
		glVertex3f(0.1,0,0);

		glColor4f (0,1,0,1);
		glVertex3f(0,0,0);
		glVertex3f(0,0.1,0);

		glColor4f (0,0,1,1);
		glVertex3f(0,0,0);
		glVertex3f(0,0,0.1);

		glColor4f (0.01,0.01,0.01,1);
		
		glVertex3f(  xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2,  ysz/2, -zsz/2 );
		glVertex3f( -xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2, -ysz/2, -zsz/2 );
		glVertex3f( -xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2, -ysz/2, -zsz/2 );
		glVertex3f(  xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2,  ysz/2, -zsz/2 );
												 	
		glVertex3f(  xsz/2,  ysz/2,  zsz/2 );	 	glVertex3f( -xsz/2,  ysz/2,  zsz/2 );
		glVertex3f( -xsz/2,  ysz/2,  zsz/2 );	 	glVertex3f( -xsz/2, -ysz/2,  zsz/2 );
		glVertex3f( -xsz/2, -ysz/2,  zsz/2 );	 	glVertex3f(  xsz/2, -ysz/2,  zsz/2 );
		glVertex3f(  xsz/2, -ysz/2,  zsz/2 );	 	glVertex3f(  xsz/2,  ysz/2,  zsz/2 );
												 	
		glVertex3f(  xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2,  ysz/2,  zsz/2 );
		glVertex3f( -xsz/2,  ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2,  ysz/2,  zsz/2 );
		glVertex3f( -xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f( -xsz/2, -ysz/2,  zsz/2 );
		glVertex3f(  xsz/2, -ysz/2, -zsz/2 );	 	glVertex3f(  xsz/2, -ysz/2,  zsz/2 );
												 	
	glEnd();

	glDisable(GL_BLEND);
}


void test_demo::draw2d( float dt )
{
	draw_string( char_width * 2, char_height * 10, "White",	1,1,1,1 );
	draw_string( char_width * 2, char_height * 11, "Red",	1,0,0,1 );
	draw_string( char_width * 2, char_height * 12, "Green",	0,1,0,1 );
	draw_string( char_width * 2, char_height * 13, "Blue",	0,0,1,1 );
}



/*-----------------------------------------------------------------------------
	class declaration :
-----------------------------------------------------------------------------*/

class test_demo2 : public base_demo {
	public:
						test_demo2( int argc, char **argv, const char *demo_name );
		virtual			~test_demo2( void );

		struct vert_s {
			int		id;
			mpoint	pos;
			mpoint	posA;
			mvector vel;
			mvector force;
			mcolor	color;
			mcolor	colorA;
			int		edge_count;
			float	z, zA;
			
			float		size, sizeA;
			bool		verified;
			int			age;
			char		gender;
			std::string	role;
			std::string	birth_place;
			std::string nationality;
		};

		struct edge_s {
			int		i0, i1;
		};

		std::vector<vert_s>	vertices;
		std::vector<edge_s>	edges;

		int find_vert( int id ) {
			for (int i=0; i<vertices.size(); i++) {
				if (vertices[i].id == id) {
					return i;
				}
			}
			return -1;
		}

		ui_float_control_exp	g_control;
		ui_float_control_exp	size_control;
		ui_int_control			dots_control ;
		ui_int_control			lines_control;
		ui_int_control			eval_control ;
		ui_int_control			grid_control ;
		float					gravity;
		int		dots ;
		int		lines;
		int		eval ;
		int		grid;
		float	size;

		virtual	void	init();
		virtual void	update( float dt );
		virtual void	draw3d( float dt );
		virtual void	draw2d( float dt );


		void paint_friends();
		void paint_gender();
		void paint_age();
		void paint_nationality();	 
		void paint_country();
		virtual void mouse_click ( int x, int y );
		//void paint_nationality();

		bool click;
		int click_x, click_y;
		int show_vert_id;
	};


/*-----------------------------------------------------------------------------
	class implementation :
-----------------------------------------------------------------------------*/

test_demo2::test_demo2( int argc, char **argv, const char *demo_name )
 : base_demo( argc, argv, demo_name ), 
 g_control("Gravity ", &gravity,    1.1f ),
 dots_control	("Draw Vertices : ", &dots  , 1, 0, 1),
 lines_control	("Draw Edges    : ", &lines , 1, 0, 1 ),
 eval_control	("Evaluation    : ", &eval  , 1, 0, 1 ),
 grid_control	("Grid          : ", &grid  , 1, 0, 1 ),
 size_control	("Dot size      : ", &size  , 1.1 )
{
	show_vert_id = -1;
	click = false;

	gravity = 1.5;
	dots = 1;
	lines = 1;
	eval = 1;
	size = 4;
	grid = 0;

	FILE *f = fopen("cann_edges.edge", "r");
	if (f==0) {	
		exit(-1);
	}

	int i0=-1, i1=-1;
	while (!feof(f)) {
		fscanf(f, "%d %d", &i0, &i1 );

		if ( find_vert(i0) == -1 ) {	
			vert_s v;
			v.id  = i0;
			v.pos =	mpoint(0,0,0) + rand_radial_vector(0.9f, 2.0f);
			v.vel = mvector(0,0,0);
			//v.pos.z = 0;
			v.edge_count = 0;
			v.verified = false;
			v.z = v.zA = v.pos.z;
			vertices.push_back( v );
		}

		if ( find_vert(i1) == -1 ) {	
			vert_s v;
			v.id  = i1;
			v.pos =	mpoint(0,0,0) + rand_radial_vector(0.9f, 2.0f);
			//v.pos.z = 0;
			v.vel = mvector(0,0,0);
			v.edge_count = 0;
			v.verified = false;
			v.z = v.zA = v.pos.z;
			vertices.push_back( v );
		}

		edge_s e;
		e.i0 = find_vert(i0);
		e.i1 = find_vert(i1);
		vertices[ find_vert(i0) ].edge_count++;
		vertices[ find_vert(i1) ].edge_count++;

		edges.push_back(e);
	}

	fclose(f);


	f = fopen("cann_vertex.edge", "r");
	if (f==0) {	
		exit(-1);
	}

	while (!feof(f)) {
		char buf[1024];
		fgets( buf, 1023, f );
		std::string line = buf;
		std::string s;
		std::istringstream is(line);
		std::vector<std::string> list;

		while (std::getline(is, s, '\t')) {
			int smc = s.find(':');
			list.push_back(s.substr(smc+1, string::npos));
		}

		int id	=	atoi(list[0].c_str());
		int i	=	find_vert(id);

		if (i==-1) {	
			continue;
		}

		vertices[i].verified	=	true;
		vertices[i].age			=	atoi( list[1].c_str() );
		vertices[i].gender		=	list[2].c_str()[0];
		vertices[i].role		=	list[3];
		vertices[i].birth_place	=	list[4];
		vertices[i].nationality	=	list[5].substr(0, list[5].length()-1);
		
	}

	paint_friends();
	paint_gender();
	paint_age();
}


test_demo2::~test_demo2()
{
}


void test_demo2::init()
{
	add_control( &g_control );
	add_control( &dots_control  );
	add_control( &lines_control );
	add_control( &eval_control  );
	add_control( &size_control );
	add_control( &grid_control );
}


void test_demo2::mouse_click ( int x, int y ) 
{
	click_x = x;
	click_y = y;
	click = true;
}


float arrange_weight = 0;

float push( float x, float power ) {
	float s = x > 0 ? 1 : -1;
	return (float)pow( abs(x), power ) * s;
}

#define SPATIAL		0
#define BY_AGE		1
#define BY_FRIENDS	2
int arrange_mode = SPATIAL;


void test_demo2::update( float dt )
{
	dt = 0.001f;

	if (GetKeyState(0x31)&0x80) {	paint_friends();	}
	if (GetKeyState(0x32)&0x80) {	paint_gender();	}
	if (GetKeyState(0x33)&0x80) {	paint_age();	}
	if (GetKeyState(0x34)&0x80) {	paint_country();	}

	if (GetKeyState(0x35)&0x80) {	arrange_mode = 	SPATIAL;	}
	if (GetKeyState(0x36)&0x80) {	arrange_mode = 	BY_AGE;		}
	if (GetKeyState(0x37)&0x80) {	arrange_mode = 	BY_FRIENDS; }
	//this->g_view_phi += 100*dt;

	#pragma omp parallel num_threads(6)
	{
		#pragma omp for
 		for (int i=0; i<vertices.size(); i++) {

			vertices[i].force = mvector(0,0,0);

			vertices[i].colorA = vertices[i].colorA.lerpTo( vertices[i].color, 0.1f );
			vertices[i].sizeA  = mmath::lerp( vertices[i].sizeA, vertices[i].size, 0.1f );
				

			//	mutual forces :
			for (int j=0; j<vertices.size(); j++) {	

				if (i==j) continue;

				mvector  r = vertices[j].pos - vertices[i].pos;
				float d = r.length() + 0.0001;
				r = r / d;
				float m = 1;//ertices[i].edge_count;

				vertices[i].force = vertices[i].force - ( r / d * gravity * m );
			}	//*/
		}
	

		for (int i=0; i<edges.size(); i++ ) {
			
			edge_s e = edges[i];
			int i0 = e.i0;
			int i1 = e.i1;

			mvector r = vertices[i0].pos - vertices[i1].pos;

			vertices[i0].force = vertices[i0].force - r * 3.1f * r.length() * pow(vertices[i1].edge_count, 0.5);
			vertices[i1].force = vertices[i1].force + r * 3.1f * r.length() * pow(vertices[i0].edge_count, 0.5);

		}


 		for (int i=0; i<vertices.size(); i++) {
			float m = 1;//vertices[i].edge_count + 0.1;
			if (eval) {
				vertices[i].vel		= vertices[i].vel + vertices[i].force * dt / m - vertices[i].vel * 0.01;
				vertices[i].pos		= vertices[i].pos + vertices[i].vel * dt;
			}
			vertices[i].posA.x	= vertices[i].pos.x;
			vertices[i].posA.y	= vertices[i].pos.y;

			if (arrange_mode==SPATIAL)		vertices[i].z = vertices[i].pos.z;
			if (arrange_mode==BY_AGE)		vertices[i].z = (((vertices[i].age-10.0f)/60.0f) * 2 - 1) * 5;
			if (arrange_mode==BY_FRIENDS)	vertices[i].z = vertices[i].edge_count/10.0f;

			vertices[i].zA = mmath::lerp( vertices[i].zA, vertices[i].z, 0.1f );

			vertices[i].posA.z	= vertices[i].zA;

			//vertices[i].posA.x = mmath::lerp( vertices[i].pos.x, push(vertices[i].pos.x,0.5f), arrange_weight );
			//vertices[i].posA.y = mmath::lerp( vertices[i].pos.y, push(vertices[i].pos.y,0.5f), arrange_weight );
		}
	}

}


void test_demo2::draw3d( float dt )
{
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glPointSize(2);

	glEnable(GL_FOG);
	glFogi( GL_FOG_MODE, GL_EXP );
	glFogf( GL_FOG_DENSITY, 0.002f );

	float pp[3] = {0,0,1};

	glLineWidth(1);

	//glEnable(GL_POINT_SMOOTH);
	glPointParameterf(GL_POINT_SIZE_MIN, 2.0f );
	glPointParameterf(GL_POINT_SIZE_MAX, 50.0f );
	glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 4.2f );
	glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, pp );



	double model_matrix[16], proj_matrix[16];
	GLint  viewport[4];

	glGetIntegerv ( GL_VIEWPORT,			viewport		);
	glGetDoublev  ( GL_MODELVIEW_MATRIX,	model_matrix	);
	glGetDoublev  ( GL_PROJECTION_MATRIX,	proj_matrix		);

	if (click) {
		int i;
		for (i=0; i<vertices.size(); i++) {
			double x, y, z;
			gluProject( vertices[i].posA.x, vertices[i].posA.y, vertices[i].posA.z, 
						model_matrix, proj_matrix, viewport, &x, &y, &z );

			y = viewport[3] - y;

			float dx = click_x - x;
			float dy = click_y - y;
			float d  = sqrt( dx*dx + dy*dy );

			if (d<5) {
				show_vert_id = i;
				vertices[i].sizeA = 150;
				break;
			}
		}
		if (i==vertices.size()) {
			show_vert_id = -1;
		}
		click = false;
	}



	if (dots) {
		for (int i=0; i<vertices.size(); i++) {
			glPointSize( vertices[i].sizeA * size );
			glBegin( GL_POINTS );
				glColor4fv( &vertices[i].colorA.r );
				glVertex3fv( &vertices[i].posA.x );
			glEnd();
		}
	}

	if (lines) {
		glBegin( GL_LINES );
			for (int i=0; i<edges.size(); i++) {
				edge_s e = edges[i];
				glColor4f(0.2, 0.2, 0.2, 1);
				float s = 0.2f;
				glColor4f  ( vertices[e.i0].colorA.r*s, vertices[e.i0].colorA.g*s, vertices[e.i0].colorA.b*s, 1 );
				glVertex3fv( &vertices[e.i0].posA.x );
				glColor4f  ( vertices[e.i1].colorA.r*s, vertices[e.i1].colorA.g*s, vertices[e.i1].colorA.b*s, 1 );
				glVertex3fv( &vertices[e.i1].posA.x );
			}
		glEnd();
	}

	float size = 128;

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	glFogf( GL_FOG_DENSITY, 0.01f );

	if (grid) {
		glBegin( GL_LINES );

			for ( float t = -size; t<=size; t+=2 ) {		
				glColor4f( 0.2f, 0.2f, 0.2f, 1.0f );
				glVertex3f( t, -size, -4 );
				glVertex3f( t, +size, -4 );
				glVertex3f( -size, t, -4 );
				glVertex3f( +size, t, -4 );
			}
			for ( float t = -size; t<=size; t+=32 ) {		
				glColor4f( 0.2f, 0.2f, 0.2f, 1.0f );
				glVertex3f( t, -size, -4 );
				glVertex3f( t, +size, -4 );
				glVertex3f( -size, t, -4 );
				glVertex3f( +size, t, -4 );
			}

		glEnd();
	}

	glBegin(GL_LINES);
		glColor4f (1,1,0,1);
		float xsz = 16;
		float ysz = 16;
		float zsz = 16;
		glColor4f (1,0,0,1);		glVertex3f(0,0,0);		glVertex3f(0.1,0,0);
		glColor4f (0,1,0,1);		glVertex3f(0,0,0);		glVertex3f(0,0.1,0);
		glColor4f (0,0,1,1);		glVertex3f(0,0,0);		glVertex3f(0,0,0.1);
	glEnd();

	glDisable(GL_BLEND);
}


void test_demo2::draw2d( float dt )
{
	char line[256];

	draw_string( char_width * 2, char_height * 15, "[1] - paint by friends",	1,1,1,1 );
	draw_string( char_width * 2, char_height * 16, "[2] - paint by gender",		1,1,1,1 );
	draw_string( char_width * 2, char_height * 17, "[3] - paint by age",		1,1,1,1 );
	draw_string( char_width * 2, char_height * 18, "[4] - paint by birth place",1,1,1,1 );
	draw_string( char_width * 2, char_height * 19, "[5] - 3D layout",			1,1,1,1 );
	draw_string( char_width * 2, char_height * 20, "[6] - arrange by age",		1,1,1,1 );
	draw_string( char_width * 2, char_height * 21, "[7] - arrange by friends",	1,1,1,1 );

	    sprintf( line, "Selected ID   : %d", show_vert_id );
	draw_string( char_width * 2, char_height * 25, line,	1,1,1,1 );

	int h = this->display_height - char_height * 30;
	//int h = 10;

	if (show_vert_id!=-1) {
		sprintf( line, "Age           : %d", vertices[show_vert_id].age );
		draw_string( char_width * 2, h + char_height * 21, line,	1,1,1,1 );

		sprintf( line, "Gender        : %s", vertices[show_vert_id].gender=='m' ? "Male" : "Female" );
		draw_string( char_width * 2, h + char_height * 22, line,	1,1,1,1 );

		sprintf( line, "Birth country : %s", vertices[show_vert_id].birth_place.c_str() );
		draw_string( char_width * 2, h + char_height * 23, line,	1,1,1,1 );

		sprintf( line, "Nationality   : %s", vertices[show_vert_id].nationality.c_str() );
		draw_string( char_width * 2, h + char_height * 24, line,	1,1,1,1 );

		sprintf( line, "Role          : %s", vertices[show_vert_id].role.c_str() );
		draw_string( char_width * 2, h + char_height * 25, line,	1,1,1,1 );
	}

	//sprintf( line, "%d %d", click_x, click_y );
	//draw_string( char_width * 2, char_height * 11, line,	1,1,1,1 );

	/*draw_string( char_width * 2, char_height * 11, "Red",	1,0,0,1 );
	draw_string( char_width * 2, char_height * 12, "Green",	0,1,0,1 );
	draw_string( char_width * 2, char_height * 13, "Blue",	0,0,1,1 );*/ 
}



void test_demo2::paint_friends()
{
	FIBITMAP *pal = FreeImage_Load(FIF_PNG, "pal01.png", 0);

	unsigned int	bpp   = FreeImage_GetBPP(pal);
	unsigned int	w	  = FreeImage_GetWidth(pal);
	byte			*bits = FreeImage_GetBits(pal);
		

	for (int i=0; i<vertices.size(); i++) {
		float v = vertices[i].edge_count / 30.0f;
			  v = mmath::clamp( v, 0, 1 );
		int   b = (int)(255 * v);
		vertices[i].size  = vertices[i].edge_count;
		vertices[i].color = mcolor( bits[bpp/8*b+2]/255.0f, bits[bpp/8*b+1]/255.0f, bits[bpp/8*b+0]/255.0f );
	}
}



void test_demo2::paint_gender()
{
	for (int i=0; i<vertices.size(); i++) {
		vertices[i].color = mcolor( 0.0f, 1.0f, 0.0f, 1.0f );
		if (vertices[i].gender=='m') { vertices[i].color = mcolor( 0.0f, 0.0f, 1.0f, 1.0f ); }
		if (vertices[i].gender=='w') { vertices[i].color = mcolor( 1.0f, 1.0f, 0.0f, 1.0f ); }
	}
}



void test_demo2::paint_age()
{
	FIBITMAP *pal = FreeImage_Load(FIF_PNG, "pal02.png", 0);

	unsigned int	bpp   = FreeImage_GetBPP(pal);
	unsigned int	w	  = FreeImage_GetWidth(pal);
	byte			*bits = FreeImage_GetBits(pal);
		

	for (int i=0; i<vertices.size(); i++) {
		float v = (vertices[i].age - 10) / 60.0f;
			  v = mmath::clamp( v, 0, 1 );
		int   b = (int)(255 * v);
		vertices[i].size  = vertices[i].age/2;
		vertices[i].color = mcolor( bits[bpp/8*b+2]/255.0f, bits[bpp/8*b+1]/255.0f, bits[bpp/8*b+0]/255.0f );
	}
}



void test_demo2::paint_country()
{
	for (int i=0; i<vertices.size(); i++) {

		bool nl = (vertices[i].birth_place=="NEDERLAND");

		vertices[i].size  = vertices[i].edge_count;
		vertices[i].color = nl ? mcolor(0,0,1,1) : mcolor(1,1,0,1);
	}
}



/*-----------------------------------------------------------------------------
	entry point :
-----------------------------------------------------------------------------*/
			   
int main( int argc, char **argv )
{
//	test_demo	demo( argc, argv, "Test Demo" );
	test_demo2	demo( argc, argv, "Test Demo 2" );


	demo.run();
}
