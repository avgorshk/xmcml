#pragma once

#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <stdexcept>
#include "keys.h"
#include "math/math.h"
#include "../include/FreeImage.h"

using std::runtime_error;
using std::string;

//#define USE_KINECT
#ifdef WIN32
	#pragma comment(lib, "glew32.lib")
	#pragma comment(lib, "FreeImage.lib")
#endif

#if defined WIN32 && defined USE_KINECT
	#include "NuiApi.h"
	#pragma comment(lib, "Kinect10.lib")
#endif

#ifndef WIN32
#undef USE_KINECT
#endif

#ifndef byte
typedef unsigned char byte;
#endif

/*-------------------------------------------------------------------------------------------------
	FEATURES :
		- Main menu on F1
			Float
			Int
			Buttons
		- ~~FPS~~
		- Stereo
		- ~~W/FS + resolution~~
		- ~~timing~~

-------------------------------------------------------------------------------------------------*/

class base_demo;

typedef void (*command_t)( std::vector<std::string> &cmdline );

inline float clamp( float v, float min, float max ) {
	if (v<min) return min;
	if (v>max) return max;
	return v;
}


class ui_control {
	public:
		virtual			~ui_control() {};
		virtual void	next() = 0;
		virtual void	prev() = 0;
		virtual void	enter() = 0;
		virtual void	draw( int x, int y, bool active, float dt ) = 0;	
	protected:
	};



class base_demo {
	public:
						base_demo( int argc, char **argv, const char *demo_name );
		virtual			~base_demo( void );

				void	run();

		virtual	void	init() = 0;
		virtual void	update( float dt ) = 0;
		virtual void	draw3d( float dt ) = 0;
		virtual void	draw2d( float dt ) = 0;

		virtual void	mouse_click ( int x, int y ) {}

		enum {
			char_width	=	8,
			char_height	=	13
		};

	public:

		static	float		g_view_phi;
		static	float		g_view_theta;
		static	float		g_view_dist;
		static	float		stereo_separation;
		static	mpoint		g_view_pos;
		
		static	void		draw_string( int x, int y, const char *string, float r=1, float g=1, float b=1, float a=1 );
		static	bool		arg_check_key( const char *key );
		static	const char*	arg_get_value( const char *key, const char *def_value );

		void	add_control		( ui_control* control );
		void	remove_control	( ui_control* control );
		void	add_command		( const char *name, command_t cmd ) { cmds.push_back( cmd_t(name, cmd) ); }


		#ifdef USE_KINECT
		mpoint	kinect_wristL		, kinect_old_wristL		;	mvector kinect_vel_wristL		;
		mpoint	kinect_wristR		, kinect_old_wristR		;	mvector kinect_vel_wristR		;
		mpoint	kinect_head			, kinect_old_head		;	mvector kinect_vel_head			;
		mpoint	kinect_shoulderL	, kinect_old_shoulderL	;	mvector kinect_vel_shoulderL	;
		mpoint	kinect_shoulderR	, kinect_old_shoulderR	;	mvector kinect_vel_shoulderR	;
		mpoint	kinect_shoulderC	, kinect_old_shoulderC	;	mvector kinect_vel_shoulderC	;
		#endif

	private:

		struct cmd_t {
			cmd_t ( const char *n, command_t c ) { name=n; cmd=c; }
			const char	*name;
			command_t	cmd;
		};

		std::vector<cmd_t> cmds;

		#ifdef USE_KINECT
		HRESULT		init_kinect ();
	    INuiSensor* m_pNuiSensor;
		HANDLE		m_pColorStreamHandle;
		HANDLE		m_hNextColorFrameEvent;
		HANDLE		m_pSkeletonStreamHandle;
		HANDLE		m_hNextSkeletonEvent;
		void		ProcessSkeleton();
		void		DrawBone(const NUI_SKELETON_DATA & skel, NUI_SKELETON_POSITION_INDEX bone0, NUI_SKELETON_POSITION_INDEX bone1);
		void		DrawSkeleton(const NUI_SKELETON_DATA & skel, int windowWidth, int windowHeight);
		void		update_joints(NUI_SKELETON_DATA &skeletonData);
		#endif


		static	bool	menu_visible;
		int							active_control;
		std::vector<ui_control*>	controls;

		public:
		static	bool	use_stereo;
		static	bool	batch_mode;
		static	int		display_width;
		static	int		display_height;
		static	int		g_buttonState;
		static	int		g_mouse_x;	
		static	int		g_mouse_y;
		static	clock_t	prev_clock;

		static	int		argc;
		static	char**	argv;


		static base_demo *instance;

		static	void	batch();

		static	void	display();
		static	void	reshape(int w, int h);
		static	void	motion(int x, int y);
		static	void	mouse(int button, int state, int x, int y);
		static	void	keyboard(unsigned char key, int x, int y);
		static	void	special(int key, int x, int y);
		static	void	idle();

		static	void	make_screenshot( const char *path = 0 );

		static	void	draw3d_internal(float dt, float eye_offset);
		static	void	draw2d_internal(float dt);

		static	void	draw_menu();

		static	int		refresh_counter;

		static	void	exit_f	( std::vector<std::string> &parsed ) { exit(0); }
		static	void	quit_f	( std::vector<std::string> &parsed ) { exit(0); }
		static	void	refresh_f( std::vector<std::string> &parsed ) { 
			if (parsed.size()==1) refresh_counter=1; 
			else refresh_counter=atoi(parsed[1].c_str()); 
			printf("refreshing...\r\n");
		}
		static	void	screenshot_f( std::vector<std::string> &parsed ) { 
			if (parsed.size()==1) make_screenshot(0); 
			else make_screenshot(parsed[1].c_str()); 

		}
	};



class ui_float_control_lin : public ui_control {
	public:
		ui_float_control_lin ( const char *name, float *valuePtr, float step, float low, float high ) {
			this->low		=	low;
			this->high		=	high;
			this->step		=	step;
			this->valuePtr	=	valuePtr;
			this->name		=	name;
		}

		virtual void	next() { *valuePtr = clamp(*valuePtr + step, low, high); };
		virtual void	prev() { *valuePtr = clamp(*valuePtr - step, low, high); };
		virtual void	enter() {};
		virtual void	draw( int x, int y, bool active, float dt )	{
			char buffer[256];
			sprintf( buffer, "%s : %g", name.c_str(), *valuePtr );
			base_demo::draw_string( x, y, buffer, 1,1,1,1 );
		}	

	protected:
		std::string name;
		float  step;
		float  low;
		float  high;
		float* valuePtr;
	};


class ui_float_control_exp : public ui_control {
	public:
		ui_float_control_exp ( const char *name, float *valuePtr, float step ) {
			this->step		=	step;
			this->valuePtr	=	valuePtr;
			this->name		=	name;
		}

		virtual void	next() { *valuePtr = *valuePtr * step; };
		virtual void	prev() { *valuePtr = *valuePtr / step; };
		virtual void	enter() {};
		virtual void	draw( int x, int y, bool active, float dt )	{
			char buffer[256];
			sprintf( buffer, "%s : %g", name.c_str(), *valuePtr );
			base_demo::draw_string( x, y, buffer, 1,1,1,1 );
		}	

	protected:
		std::string name;
		float  step;
		float* valuePtr;
	};


class ui_int_control: public ui_control {
	public:
		ui_int_control ( const char *name, int *valuePtr, int step, int low, int high ) {
			this->low		=	low;
			this->high		=	high;
			this->step		=	step;
			this->valuePtr	=	valuePtr;
			this->name		=	name;
		}

		virtual void	next() { *valuePtr = clamp(*valuePtr + step, low, high); };
		virtual void	prev() { *valuePtr = clamp(*valuePtr - step, low, high); };
		virtual void	enter() {};
		virtual void	draw( int x, int y, bool active, float dt )	{
			char buffer[256];
			sprintf( buffer, "%s : %d", name.c_str(), *valuePtr );
			base_demo::draw_string( x, y, buffer, 1,1,1,1 );
		}	

	protected:
		std::string name;
		int  step;
		int  low;
		int  high;
		int* valuePtr;
	};

