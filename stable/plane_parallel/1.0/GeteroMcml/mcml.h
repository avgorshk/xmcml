#ifndef __MCML_H
#define __MCML_H

/***********************************************************
 * Copyright Univ. of Texas M.D. Anderson Cancer Center
 * 1992.
 *
 * Monte Carlo simulation of photon distribution in
 * multi-layered turbid media in ANSI Standard C.
 ****
 * Starting Date: 10/1991.
 * Current Date: 6/1992.
 *
 * Lihong Wang, Ph. D.
 * Steven L. Jacques, Ph. D.
 * Laser Biology Research Laboratory - 17
 * M.D. Anderson Cancer Center
 * University of Texas
 * 1515 Holcombe Blvd.
 * Houston, TX 77030
 * USA
 *
 * This program was based on:
 * (1) The Pascal code written by Marleen Keijzer and
 * Steven L. Jacques in this laboratory in 1989, which
 * deals with multi-layered turbid media.
 *
 * (2) Algorithm for semi-infinite turbid medium by
 * S.A. Prahl, M. Keijzer, S.L. Jacques, A.J. Welch,
 * SPIE Institute Series Vol. IS 5 (1989), and by
 * A.N. Witt, The Astrophysical journal Supplement
 * Series 35, 1-6 (1977).
 *
 * Major modifications include:
 * . Conform to ANSI Standard C.
 * . Removal of limit on number of array elements,
 * because arrays in this program are dynamically
 * allocated. This means that the program can accept
 * any number of layers or gridlines as long as the
 * memory permits.
 * . Avoiding global variables whenever possible. This
 * program has not used global variables so far.
 * . Grouping variables logically using structures.
 * . Top-down design, keep each subroutine clear &
 * short.
 * . Reflectance and transmittance are angularly
 * resolved.
 ****
 * General Naming Conventions:
 * Preprocessor names: all capital letters,
 * e.g. #define PREPROCESSORS
 * Globals: first letter of each word is capital, no
 * underscores,
 * e.g. short GlobalVar;
 * Dummy variables: first letter of each word is capital,
 * and words are connected by underscores,
 * e.g. void NiceFunction(char Dummy_Var);
 * Local variables: all lower cases, words are connected
 * by underscores,
 * e.g. short local_var;
 * Function names or data types: same as Globals.
 *
 ****
 * Dimension of length: cm.
 *
 ****/

 #include <math.h>
 #include <stdlib.h>
 #include <stdio.h>
 #include <stddef.h>
 #include <time.h>
 #include <string.h>
 #include <ctype.h>

#include "mcml_getero.h"

 #define PI 3.1415926
 #define WEIGHT 1E-4 /* Critical weight for roulette. */
 #define CHANCE 0.1 /* Chance of roulette survival. */

 #define Boolean char

 #define SIGN(x) ((x)>=0 ? 1:-1)

 double *AllocVector(short, short);
 double **AllocMatrix(short, short,short, short);
 void FreeVector(double *, short, short);
 void FreeMatrix(double **, short, short, short, short);
 void nrerror(char *);

#endif //__MCML_H