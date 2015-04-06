#include <stdexcept>
#ifdef WIN32
#include <Windows.h>
#endif

#include "../include/GL/glew.h"
#include "../include/GL/glut.h"

#include "base_demo.h"



base_demo* base_demo::instance			= 0;
bool	base_demo::use_stereo			= false;
bool	base_demo::batch_mode			= false;
int		base_demo::display_width		= 0;
int		base_demo::display_height		= 0;
int		base_demo::g_buttonState		= 0;
int		base_demo::g_mouse_x			= 0;
int		base_demo::g_mouse_y			= 0;
float	base_demo::stereo_separation	= 0.125f;
float	base_demo::g_view_phi			= 45;
float	base_demo::g_view_theta			= 20;
float	base_demo::g_view_dist			= 20.0f;
clock_t	base_demo::prev_clock			= clock();
int		base_demo::argc					= 0;
char**	base_demo::argv					= 0;
bool	base_demo::menu_visible			= false;
int		base_demo::refresh_counter		= 0;
mpoint	base_demo::g_view_pos			= mpoint(0,0,0);




class exit_control : public ui_control {
	public:
		virtual void	next() {}
		virtual void	prev() {}
		virtual void	enter() { exit(0); }
		virtual void	draw( int x, int y, bool active, float dt ) { base_demo::draw_string(x,y, "Exit", 1,0,0,1); }
	};

class fullscr_control : public ui_control {
	public:
		virtual void	next() {}
		virtual void	prev() {}
		virtual void	enter() { glutFullScreen(); }
		virtual void	draw( int x, int y, bool active, float dt ) { base_demo::draw_string(x,y, "Fullscreen mode", 1,1,1,1); }
	};

class windowed_control : public ui_control {
	public:
		virtual void	next() {}
		virtual void	prev() {}
		virtual void	enter() { glutReshapeWindow(800,600); glutPositionWindow(200,100); }
		virtual void	draw( int x, int y, bool active, float dt ) { base_demo::draw_string(x,y, "Windowed mode", 1,1,1,1); }
	};

exit_control		_exit_control;
fullscr_control		_fullscr_control;
windowed_control	_windowed_control;
float stereo_separation;

ui_float_control_lin stereo_sep_control("Stereo factor", &base_demo::stereo_separation, 0.125/4, -10, 10 );

/*-----------------------------------------------------------------------------
	class implementation :
-----------------------------------------------------------------------------*/

base_demo::base_demo( int _argc, char **_argv, const char *demo_name  )
{
	argc	=	_argc;
	argv	=	_argv;

	assert( instance==0 );
	instance = this;

	use_stereo	=	arg_check_key("-stereo");
	batch_mode	=	arg_check_key("-batch");
	int w		=	atoi( arg_get_value( "-width",	"800" ) );
	int h		=	atoi( arg_get_value( "-height",	"600" ) );
	int x		=	atoi( arg_get_value( "-xpos",	"200" ) );
	int y		=	atoi( arg_get_value( "-ypos",	"100" ) );
	bool msaa	=	arg_check_key("-msaa");
	bool fs		=	arg_check_key("-fullscr");

	if (batch_mode) {
		fs = false;
		use_stereo = false;
	}
	
	unsigned int init_flag = 
		GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE
		| (use_stereo ? GLUT_STEREO : 0)
		| (msaa       ? GLUT_MULTISAMPLE : 0)
		;
		
	glutInit( &argc, argv );
	glutInitDisplayMode( init_flag );
    glutInitWindowSize( w, h );
	glutInitWindowPosition( x, y );

    glutCreateWindow(demo_name);

	if (fs) {
		glutFullScreen();
	}

    GLenum err = glewInit();
    assert(err==GLEW_OK);

	#ifdef USE_KINECT
		HRESULT hr = init_kinect();
		if (FAILED(hr)) {
			printf("Failed to init Kinect.");
			exit(-1);
		}
	#endif

    glutDisplayFunc	( display	);
    glutReshapeFunc	( reshape	);
    glutMotionFunc	( motion	);
    glutMouseFunc	( mouse		);
	glutKeyboardFunc( keyboard	);
    glutIdleFunc	( idle		);
	glutSpecialFunc	( special	);
}


base_demo::~base_demo(void)
{
}



bool base_demo::arg_check_key( const char *key )
{
	assert( key[0]=='-' );

	for (int i=0; i<argc; i++) {
		if ( strcmp(argv[i], key)==0 ) {
			return true;
		}
	}
	return false;
}



const char* base_demo::arg_get_value( const char *key, const char *def_value )
{
	assert( key[0]=='-' );

	for (int i=0; i<argc-1; i++) {
		if ( strcmp(argv[i], key)==0 ) {
			if (argv[i+1][0]=='-') {
				return def_value;
			} else {
				return argv[i+1];
			}
		}
	}
	return def_value;
}



void base_demo::run()
{
	init();

	active_control = 0;

	add_command( "quit",		quit_f );
	add_command( "exit",		quit_f );
	add_command( "refresh",		refresh_f );
	add_command( "screenshot",	screenshot_f );

	add_control( &stereo_sep_control );
	add_control( &_fullscr_control );
	add_control( &_windowed_control );
	add_control( &_exit_control );

	if (batch_mode) {
		printf("Visualizer batch mode\r\n");
		glutMainLoop();
	} else {
		glutMainLoop();
	}
}



bool quit = false;

void base_demo::batch()
{

}


/*-----------------------------------------------------------------------------
	GLUT handlers :
-----------------------------------------------------------------------------*/

bool is_space( char ch ) { return ch==' ' || ch=='\t'; }
bool is_cmdchar( char ch ) { return isalnum(ch) || ch=='_' || ch=='.'; }

std::vector<std::string> parse_command ( const char *cmd )
{
	const char *scan = cmd;

	std::vector<std::string> parsed;

	try {

		while (true) {

			if ( *scan=='\0' ) {
				break;
			}

			//	skip spaces :
			while ( is_space( *scan ) || *scan=='\r' ) {
				scan++;
				continue;
			}
		
			if ( *scan=='"' ) {

				scan++;

				std::string token;

				while (1) {
					if (*scan=='"') { scan++; break; }
					if (*scan=='\0' || *scan=='\n' || *scan=='\r') { throw std::logic_error("unterminated quoted string"); }
					token.push_back(*scan);
					scan++;
				}

				parsed.push_back( token );

				continue;
			}

			if ( is_cmdchar(*scan) ) {

				std::string token;

				while ( is_cmdchar(*scan) ) {
					token.push_back( *scan );
					scan++;
				}

				parsed.push_back( token );

				continue;
			}


			char err[128];
			sprintf(err, "unexpected character '%c'", *scan );
			throw std::runtime_error(err);
		}
	} catch (const std::exception &ex) {
		printf("Error : %s\r\n", ex.what());
	}

	return parsed;
}





void base_demo::display()
{
	//	update timing :
	clock_t curr_time	= clock();
	clock_t	dtime		= curr_time - prev_clock;
	float	ftime		= ((float)dtime) / CLOCKS_PER_SEC;
	prev_clock			= curr_time;



	if (batch_mode) {

		char buffer[512];

		while ( refresh_counter <= 0) {
		
			printf(">");
			gets( buffer );
			printf("%s\r\n", buffer);

			std::vector<std::string> parsed = parse_command( buffer );

			int i;
			for (i=0; i<parsed.size(); i++) {
				printf("[%s]", parsed[i].c_str());
			}
			printf("\r\n");
	
			if ( parsed.size()==0 ) {
				continue;
			}

			for ( i=0; i<instance->cmds.size(); i++) {
				if ( strcmp( instance->cmds[i].name, parsed[0].c_str() )==0 ) {
					instance->cmds[i].cmd( parsed );
					break;
				}
			}
			if (i==parsed.size()) {
				printf("unknown command : %s\r\n", parsed[0].c_str());				
			}

			/*if ( strcmp( parsed[0].c_str(), "quit"		) == 0 ) { exit(0); }
			if ( strcmp( parsed[0].c_str(), "exit"		) == 0 ) { exit(0); }

			if ( strcmp( parsed[0].c_str(), "refresh"	) == 0 ) { break; }*/

		};
	}

	refresh_counter--;


	instance->update( ftime );

	if (use_stereo) {

		glDrawBuffer( GL_BACK_LEFT );

		draw3d_internal( ftime,  stereo_separation/2 );
		draw2d_internal( ftime );

		glDrawBuffer( GL_BACK_RIGHT );

		draw3d_internal( ftime, -stereo_separation/2 );
		draw2d_internal( ftime );

	} else {
	
		glDrawBuffer( GL_BACK );	

		draw3d_internal( ftime, 0 );
		draw2d_internal( ftime );
	}

	//	swap buffers :
	glutSwapBuffers();
}


void base_demo::draw3d_internal(float dt, float eye_offset)
{
	glPointSize(1);
	glLineWidth(1);

	//	clear backbuffer :
	glClearColor( 0,0,0,0 );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );
	
	//	setup prjection matrix for 3d view :	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float aspect = (float) display_height / (float) display_width;
	glFrustum(-0.1f, 0.1f, -0.1f*aspect, 0.1f*aspect, 0.1f, 10000.0f );
	
	//	setup view matrix :	
	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();

	float tx = - sin( g_view_phi/57.29f );
	float ty = + cos( g_view_phi/57.29f );
	float x = g_view_dist * cos( g_view_phi/57.29f ) * cos( g_view_theta / 57.29 ) + tx * eye_offset;
	float y = g_view_dist * sin( g_view_phi/57.29f ) * cos( g_view_theta / 57.29 ) + ty * eye_offset;
	float z = g_view_dist * sin( g_view_theta/57.29 );
	gluLookAt( x,y,z, 0,0,0, 0,0,1 );

	g_view_pos = mpoint(x,y,z);

	/*glTranslatef(0, 0, -g_view_dist);
	glRotatef(g_view_theta, 1, 0, 0);
	glRotatef(g_view_phi, 0, 1, 0);*/

	#ifdef USE_KINECT
	instance->ProcessSkeleton();
	#endif

	glPointSize(1);
	glLineWidth(1);

	//	ready to draw 3d stuff :
	instance->draw3d( dt );
}


void base_demo::draw2d_internal(float dt)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho( 0, display_width, display_height, 0, -9999, 9999 );
    
	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();


	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);


	instance->draw2d( dt );

	if (!batch_mode) {
		draw_menu();

		//	show fps :
		char s[25];
		sprintf(s, "FPS %3.1f", 1 / dt );
		draw_string( display_width - char_width * strlen(s), char_height, s, 1,1,1,1);
	}
}




void base_demo::reshape(int w, int h)
{
	display_width = w;
	display_height = h;
    glViewport(0, 0, w, h);
}



void base_demo::motion(int x, int y)
{
    int dx = x - g_mouse_x;
    int dy = y - g_mouse_y;
    
    if (g_buttonState & 1)
    {
        g_view_phi -= 0.5f * dx;
        g_view_theta += 0.5f * dy;
		if (g_view_theta < -89) g_view_theta = -89;
		if (g_view_theta > +89) g_view_theta = +89;
    }
    if (g_buttonState & 4)
    {
		g_view_dist *= pow(1.007f, -dy);
    }

    g_mouse_x = x;
    g_mouse_y = y;
    glutPostRedisplay();
}



void base_demo::mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
        g_buttonState |= 1<<button;
	} else if (state == GLUT_UP) {
        g_buttonState = 0;
	}
    
	//printf("%d", button);
	if (button==1 && state == GLUT_UP) {
		instance->mouse_click( x, y );
	}

    g_mouse_x = x;
    g_mouse_y = y;
    glutPostRedisplay();
}



void base_demo::keyboard(unsigned char key, int x, int y)
{
	if (key==KEY_ESCAPE) {
		exit(0);
	}
	if (instance->menu_visible) {
		if (key==KEY_ENTER) {
			instance->controls[ instance->active_control ]->enter();
		}
	}
}

void base_demo::special(int key, int x, int y)
{
	if (key==GLUT_KEY_F1) {
		menu_visible = !menu_visible;
	}
	if (key==GLUT_KEY_F12) {
		make_screenshot();
	}
	if (instance->menu_visible) {
		if (key==GLUT_KEY_DOWN) {
			instance->active_control ++;
			if (instance->active_control >= (int)instance->controls.size()) instance->active_control = (int)instance->controls.size()-1;
		} 
		if (key==GLUT_KEY_UP) {
			instance->active_control --;
			if (instance->active_control < 0) instance->active_control = 0;
		} 
		if (key==GLUT_KEY_RIGHT) {
			instance->controls[ instance->active_control ]->next();
		} 
		if (key==GLUT_KEY_LEFT) {
			instance->controls[ instance->active_control ]->prev();
		} 
	}
}



void base_demo::idle()
{
    glutPostRedisplay();
}



void base_demo::draw_string( int x, int y, const char *string, float r, float g, float b, float a )
{
	glColor4f( r, g, b, a );

	glRasterPos2i( x, y );

	const char *s = string;
	while (*s) {
		glutBitmapCharacter( GLUT_BITMAP_8_BY_13, *s );
		s++;
	}
}



void base_demo::draw_menu()
{
	int x = char_width * 3;
	int y = char_height;
	int dh = char_height;

	draw_string( 0, y, "[ F1 ]", 1,1,1,1 );

	if ( !menu_visible ) {
		return;
	}

	//y += dh;
	//draw_string( 0, y, "--------------------", 1,1,1,1 );

	for (int i=0; i<instance->controls.size(); i++) {
		y += dh;
		if (instance->active_control==i) {
			draw_string( 0, y, ">>", 1,1,1,1);
		}
		instance->controls[i]->draw( x, y, instance->active_control==i, 0.1f );
	}
}


void base_demo::add_control( ui_control* control )
{
	controls.push_back( control );
}




void base_demo::make_screenshot( const char *path )
{
	char sbuffer[256];

	if (!path) {
		struct tm * timeinfo;
		time_t		rawtime;

		time ( &rawtime );
		timeinfo = localtime ( &rawtime );
		sprintf ( sbuffer, "shot_%08X.png", (unsigned int)rawtime );
		path = sbuffer;
	}

	printf("writing screenshot : %s\r\n", path);

	GLint	viewport_size[4];
	GLint				last_buffer;
	GLenum				gl_format	= GL_BGRA;
	unsigned int		bpp			= 32;

	glGetIntegerv(GL_VIEWPORT, viewport_size);

	int w = viewport_size[2];
	int h = viewport_size[3];
	
	// Read bits from color buffer
	glPixelStorei(GL_PACK_ALIGNMENT,   1);
	glPixelStorei(GL_PACK_ROW_LENGTH,  0);
	glPixelStorei(GL_PACK_SKIP_ROWS,   0);
	glPixelStorei(GL_PACK_SKIP_PIXELS, 0);

	std::vector<char> buffer;
	buffer.resize( viewport_size[2]*viewport_size[3]*(bpp/8) );

	glGetIntegerv(GL_READ_BUFFER, &last_buffer);
	glReadBuffer(GL_FRONT);

	glReadPixels(0, 0, viewport_size[2], viewport_size[3], gl_format, GL_UNSIGNED_BYTE, &buffer[0]);

	// Set last used read buffer
	glReadBuffer(last_buffer);



	FREE_IMAGE_FORMAT	fif_format;
	int	flag	= 0;

	fif_format	= FIF_PNG;
	flag		= PNG_DEFAULT; // PNG_Z_NO_COMPRESSION;

	FIBITMAP *bitmap = FreeImage_Allocate(w, h, bpp);

	if (!bitmap) {
		throw std::runtime_error("can't allocate image");
	}

	byte *bits = FreeImage_GetBits(bitmap);

	memcpy(bits, &buffer[0], w*h*(bpp/8));

	// Save image to file
	FreeImage_Save(fif_format, bitmap, path, flag); 

	// Free allocated memory
	FreeImage_Unload(bitmap);
}


/*-------------------------------------------------------------------------------------------------
	Kinect stuff :
-------------------------------------------------------------------------------------------------*/
#ifdef USE_KINECT

HRESULT base_demo::init_kinect()
{

    INuiSensor * pNuiSensor;
    HRESULT hr;

    int iSensorCount = 0;
    hr = NuiGetSensorCount(&iSensorCount);
    if (FAILED(hr))
    {
        return hr;
    }

    // Look at each Kinect sensor
    for (int i = 0; i < iSensorCount; ++i) {
        // Create the sensor so we can check status, if we can't create it, move on to the next
        hr = NuiCreateSensorByIndex(i, &pNuiSensor);
        if (FAILED(hr)) {
            continue;
        }

        // Get the status of the sensor, and if connected, then we can initialize it
        hr = pNuiSensor->NuiStatus();
        if (S_OK == hr)	{
            m_pNuiSensor = pNuiSensor;
            break;
        }

        // This sensor wasn't OK, so release it since we're not using it
        pNuiSensor->Release();
    }

    if (NULL != m_pNuiSensor) {
        // Initialize the Kinect and specify that we'll be using color
        hr = m_pNuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_SKELETON); 
        if (SUCCEEDED(hr)) {
			/* COLOR */
            // Create an event that will be signaled when color data is available
            m_hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

            // Open a color image stream to receive color frames
            hr = m_pNuiSensor->NuiImageStreamOpen(
                NUI_IMAGE_TYPE_COLOR,
                NUI_IMAGE_RESOLUTION_640x480,
                0,
                2,
                m_hNextColorFrameEvent,
                &m_pColorStreamHandle);

			/* SKElETON */
            // Create an event that will be signaled when skeleton data is available
            m_hNextSkeletonEvent = CreateEventW(NULL, TRUE, FALSE, NULL);

            // Open a skeleton stream to receive skeleton data
            hr = m_pNuiSensor->NuiSkeletonTrackingEnable(m_hNextSkeletonEvent, 0); 
        }
    }

    if (NULL == m_pNuiSensor || FAILED(hr)) {
        printf("No ready Kinect found!");
        return E_FAIL;
    }

	m_pNuiSensor->NuiSkeletonTrackingEnable(m_hNextSkeletonEvent, NUI_SKELETON_TRACKING_FLAG_ENABLE_SEATED_SUPPORT);


    return hr;
}



void lerp_to( mpoint &p, mpoint &old, mvector &v, NUI_SKELETON_DATA &skeletonData, int jointId )
{
    NUI_SKELETON_POSITION_TRACKING_STATE jointState = skeletonData.eSkeletonPositionTrackingState[jointId];

	Vector4 pp = skeletonData.SkeletonPositions[jointId];

	old = p;

	if (jointState==NUI_SKELETON_POSITION_TRACKED) {
		p.x	=	mmath::lerp( p.x, pp.x, 0.8 );
		p.y	=	mmath::lerp( p.y, pp.y, 0.8 );
		p.z	=	mmath::lerp( p.z, pp.z, 0.8 );
	}
	if (jointState==NUI_SKELETON_POSITION_INFERRED) {
		p.x	=	mmath::lerp( p.x, pp.x, 0.2 );
		p.y	=	mmath::lerp( p.y, pp.y, 0.2 );
		p.z	=	mmath::lerp( p.z, pp.z, 0.2 );
	}

	v = v.lerp( p - old, 0.5 );
}



void base_demo::update_joints( NUI_SKELETON_DATA &skeletonData )
{
	lerp_to( kinect_wristL		,	kinect_old_wristL		, kinect_vel_wristL		, skeletonData, NUI_SKELETON_POSITION_WRIST_LEFT		);
	lerp_to( kinect_wristR		,	kinect_old_wristR		, kinect_vel_wristR		, skeletonData, NUI_SKELETON_POSITION_WRIST_RIGHT		);
	lerp_to( kinect_head		,	kinect_old_head			, kinect_vel_head		, skeletonData, NUI_SKELETON_POSITION_HEAD				);
	lerp_to( kinect_shoulderL	,	kinect_old_shoulderL	, kinect_vel_shoulderL	, skeletonData, NUI_SKELETON_POSITION_SHOULDER_LEFT		);
	lerp_to( kinect_shoulderR	,	kinect_old_shoulderR	, kinect_vel_shoulderR	, skeletonData, NUI_SKELETON_POSITION_SHOULDER_RIGHT	);
	lerp_to( kinect_shoulderC	,	kinect_old_shoulderC	, kinect_vel_shoulderC	, skeletonData, NUI_SKELETON_POSITION_SHOULDER_CENTER	);

	if ( kinect_wristL.y > kinect_shoulderC.y && kinect_wristR.y > kinect_shoulderC.y ) {
		g_view_dist -= 10 * kinect_vel_wristL.z;
		g_view_dist -= 10 * kinect_vel_wristR.z;
		//g_view_dist += 10 * (kinect_vel_wristR.x - kinect_vel_wristL.x);

	}

	if ( kinect_wristL.y > kinect_shoulderC.y ) {
		g_view_phi	-= 100 * kinect_vel_wristL.x;
	}

	if ( kinect_wristR.y > kinect_shoulderC.y ) {
		g_view_phi	-= 100 * kinect_vel_wristR.x;
	}
}



void base_demo::ProcessSkeleton()
{
    NUI_SKELETON_FRAME skeletonFrame = {0};

    HRESULT hr = m_pNuiSensor->NuiSkeletonGetNextFrame(0, &skeletonFrame);
    if ( FAILED(hr) ) {
        return;
    }


    // smooth out the skeleton data
    m_pNuiSensor->NuiTransformSmooth(&skeletonFrame, NULL);

    int width  = display_width;
    int height = display_height;

    for (int i = 0 ; i < NUI_SKELETON_COUNT; ++i) {
        NUI_SKELETON_TRACKING_STATE trackingState = skeletonFrame.SkeletonData[i].eTrackingState;

        if (trackingState == NUI_SKELETON_TRACKED) {
			update_joints( skeletonFrame.SkeletonData[i] );
            //DrawSkeleton(skeletonFrame.SkeletonData[i], width, height);
        }
        else if (trackingState == NUI_SKELETON_POSITION_ONLY) {

			glPointSize(3);

			glBegin( GL_POINTS );

				Vector4 p = skeletonFrame.SkeletonData[i].Position;

				glColor4f(1,1,0,1);
				glVertex3f( p.x, p.y, p.z );

			glEnd();
        }
    }
}


void base_demo::DrawBone( const NUI_SKELETON_DATA & skel, NUI_SKELETON_POSITION_INDEX joint0, NUI_SKELETON_POSITION_INDEX joint1 )
{
    NUI_SKELETON_POSITION_TRACKING_STATE joint0State = skel.eSkeletonPositionTrackingState[joint0];
    NUI_SKELETON_POSITION_TRACKING_STATE joint1State = skel.eSkeletonPositionTrackingState[joint1];

    // If we can't find either of these joints, exit
    if (joint0State == NUI_SKELETON_POSITION_NOT_TRACKED || joint1State == NUI_SKELETON_POSITION_NOT_TRACKED) {
        return;
    }
    
    // Don't draw if both points are inferred
    if (joint0State == NUI_SKELETON_POSITION_INFERRED && joint1State == NUI_SKELETON_POSITION_INFERRED) {
        return;
    }

	Vector4 p0 = skel.SkeletonPositions[joint0];
	Vector4 p1 = skel.SkeletonPositions[joint1];

    // We assume all drawn bones are inferred unless BOTH joints are tracked
    if (joint0State == NUI_SKELETON_POSITION_TRACKED && joint1State == NUI_SKELETON_POSITION_TRACKED) {
		glColor4f(0,1,0,1);
        glVertex3f( p0.x, p0.y, p0.z );
		glColor4f(0,1,0,1);
        glVertex3f( p1.x, p1.y, p1.z );
    }
    else {
		glColor4f(1,0,0,1);
        glVertex3f( p0.x, p0.y, p0.z );
		glColor4f(1,0,0,1);
        glVertex3f( p1.x, p1.y, p1.z );
    }
}


void base_demo::DrawSkeleton( const NUI_SKELETON_DATA & skel, int windowWidth, int windowHeight )
{
    int i;

	glPointSize(7);
	glLineWidth(2);

    /*for (i = 0; i < NUI_SKELETON_POSITION_COUNT; ++i)
    {
        m_Points[i] = SkeletonToScreen(skel.SkeletonPositions[i], windowWidth, windowHeight);
    } */

	glBegin( GL_LINES );

		// Render Torso
		DrawBone(skel, NUI_SKELETON_POSITION_HEAD, NUI_SKELETON_POSITION_SHOULDER_CENTER);
		DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_LEFT);
		DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_RIGHT);
		DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SPINE);
		DrawBone(skel, NUI_SKELETON_POSITION_SPINE, NUI_SKELETON_POSITION_HIP_CENTER);
		DrawBone(skel, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_LEFT);
		DrawBone(skel, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_RIGHT);

		// Left Arm
		DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_LEFT, NUI_SKELETON_POSITION_ELBOW_LEFT);
		DrawBone(skel, NUI_SKELETON_POSITION_ELBOW_LEFT, NUI_SKELETON_POSITION_WRIST_LEFT);
		DrawBone(skel, NUI_SKELETON_POSITION_WRIST_LEFT, NUI_SKELETON_POSITION_HAND_LEFT);

		// Right Arm
		DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_RIGHT, NUI_SKELETON_POSITION_ELBOW_RIGHT);
		DrawBone(skel, NUI_SKELETON_POSITION_ELBOW_RIGHT, NUI_SKELETON_POSITION_WRIST_RIGHT);
		DrawBone(skel, NUI_SKELETON_POSITION_WRIST_RIGHT, NUI_SKELETON_POSITION_HAND_RIGHT);

		// Left Leg
		DrawBone(skel, NUI_SKELETON_POSITION_HIP_LEFT, NUI_SKELETON_POSITION_KNEE_LEFT);
		DrawBone(skel, NUI_SKELETON_POSITION_KNEE_LEFT, NUI_SKELETON_POSITION_ANKLE_LEFT);
		DrawBone(skel, NUI_SKELETON_POSITION_ANKLE_LEFT, NUI_SKELETON_POSITION_FOOT_LEFT);

		// Right Leg
		DrawBone(skel, NUI_SKELETON_POSITION_HIP_RIGHT, NUI_SKELETON_POSITION_KNEE_RIGHT);
		DrawBone(skel, NUI_SKELETON_POSITION_KNEE_RIGHT, NUI_SKELETON_POSITION_ANKLE_RIGHT);
		DrawBone(skel, NUI_SKELETON_POSITION_ANKLE_RIGHT, NUI_SKELETON_POSITION_FOOT_RIGHT);

	glEnd();
    
    // Draw the joints in a different color
	glBegin( GL_POINTS );

    for (i = 0; i < NUI_SKELETON_POSITION_COUNT; ++i)
    {
		Vector4 p = skel.SkeletonPositions[i];

        if ( skel.eSkeletonPositionTrackingState[i] == NUI_SKELETON_POSITION_INFERRED ) {
			glColor4f(1,0,0,1);
            glVertex3f( p.x, p.y, p.z );
        }
        else if ( skel.eSkeletonPositionTrackingState[i] == NUI_SKELETON_POSITION_TRACKED ) {
			glColor4f(0,1,0,1);
            glVertex3f( p.x, p.y, p.z );
        }
    }

	glEnd();
}

#endif



