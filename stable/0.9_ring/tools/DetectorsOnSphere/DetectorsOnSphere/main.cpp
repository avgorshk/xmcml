#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

typedef double Func(double x);

double TrapeziumFormula(double a, double b, int n, Func func)
{
    double h = (b - a) / n;
    double result = 0.0;
    for (int i = 1; i < n; ++i)
    {
        result += func(a + i * h);
    }
    result += (func(a) + func(b)) / 2.0;
    result *= h;
    return result;
}

double func1(double x)
{
    double r = 80.0;
    return asin(0.5 / sqrt(r * r - x * x)) - asin(-0.5 / sqrt(r * r - x * x));
}

double GetX(double r, double l)
{
    double theta = l / r;
    double alpha = (PI - theta) / 2.0;
    double z = r * (1 - sin(theta) * sin(theta) / (2.0 * sin(alpha) * sin(alpha)));
    double x = sqrt(r * r - z * z);
    return x;
}

double GetLenght(double r, double x)
{
    double z = sqrt(r * r - x * x);
    double d = sqrt(x * x + (r - z) * (r - z));
    double cosTheta = 1 - d * d / (2.0 * r * r);
    return r * acos(cosTheta);
}

void main()
{
    int n = 40;
    double r = 80.0;
    double prev_x = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double x = GetX(r, 1.0 + i);
        double c = prev_x + (x - prev_x) / 2.0;
        double l = x - prev_x;
        /*printf("detector %d: c = %.12lf l = %.12lf | %.2lf || %.12lf\n", i, c, l, GetLenght(r, x),
            r * TrapeziumFormula(c - l / 2.0, c + l / 2.0, 10000, func1));*/
        printf("%.12lf\n", 
            r * TrapeziumFormula(c - l / 2.0, c + l / 2.0, 10000, func1));
        prev_x = x;
    }
}