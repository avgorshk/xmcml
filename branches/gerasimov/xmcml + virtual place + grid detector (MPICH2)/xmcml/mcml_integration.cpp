#include "mcml_integration.h"

#include <math.h>

#define PI 3.14159265358979323846

double ComputeWeightIntegral(double dot, double anisotropy, double eps, uint n)
{
    const double coeff = 2.0 * sqrt(3.0);

	double a, b, c, d;
	double xp, xm, yp, ym;
	double sin_theta_xp, sin_theta_xm, sin_phi_yp, sin_phi_ym, 
		cos_phi_yp, cos_phi_ym;
	double func_xp_yp, func_xp_ym, func_xm_yp, func_xm_ym;
	double anisotropy2 = anisotropy * anisotropy;
	double sqrt_dot = sqrt(1 - dot * dot);
	double fx, px, tmp;
	double f, p;

	double xh = PI / n;
	double yh = 2 * PI / n;
	double xyh = xh * yh;

	double sum = 0.0;
	for (uint i = 0; i < n; ++i)
	{
		a = xh * i;
		b = a + xh;
		xp = 0.5 * (a + b) + (b - a) / coeff;
		xm = 0.5 * (a + b) - (b - a) / coeff;
		for (uint j = 0; j < n; ++j)
		{			
			c = yh * j;
			d = c + yh;
			yp = 0.5 * (c + d) + (d - c) / coeff;
			ym = 0.5 * (c + d) - (d - c) / coeff;
			
			sin_theta_xp = sin(xp);
			sin_theta_xm = sin(xm);
			sin_phi_yp = sin(yp);
			sin_phi_ym = sin(ym);
			cos_phi_yp = cos(yp);
			cos_phi_ym = cos(ym);

			//func(xp, yp)
			px = sin_theta_xp * cos_phi_yp;
			fx = dot * px + sqrt_dot * sin_theta_xp * sin_phi_yp;
			tmp = 1 + anisotropy2 - 2 * anisotropy * fx;
			f = (1 - anisotropy2) / (2 * tmp * sqrt(tmp));
            p = pow(0.1 + 0.9 * 0.5 * (1 + px), eps);
			func_xp_yp = f * p * sin_theta_xp;
			
			//func(xm, yp)
			px = sin_theta_xm * cos_phi_yp;
			fx = dot * px + sqrt_dot * sin_theta_xm * sin_phi_yp;
			tmp = 1 + anisotropy2 - 2 * anisotropy * fx;
			f = (1 - anisotropy2) / (2 * tmp * sqrt(tmp));
            p = pow(0.1 + 0.9 * 0.5 * (1 + px), eps);
			func_xm_yp = f * p * sin_theta_xm;

			//func(xp, ym)
			px = sin_theta_xp * cos_phi_ym;
			fx = dot * px + sqrt_dot * sin_theta_xp * sin_phi_ym;
			tmp = 1 + anisotropy2 - 2 * anisotropy * fx;
			f = (1 - anisotropy2) / (2 * tmp * sqrt(tmp));
            p = pow(0.1 + 0.9 * 0.5 * (1 + px), eps);
			func_xp_ym = f * p * sin_theta_xp;

			//func(xm, ym)
			px = sin_theta_xm * cos_phi_ym;
			fx = dot * px + sqrt_dot * sin_theta_xm * sin_phi_ym;
			tmp = 1 + anisotropy2 - 2 * anisotropy * fx;
			f = (1 - anisotropy2) / (2 * tmp * sqrt(tmp));
			p = pow(0.1 + 0.9 * 0.5 * (1 + px), eps);
			func_xm_ym = f * p * sin_theta_xm;

			sum += 0.25 * (b - a) * (d - c) * (func_xp_yp + func_xp_ym + func_xm_yp + func_xm_ym);
		}
	}

	return sum / (2.0 * PI);
}