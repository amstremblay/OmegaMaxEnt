
/* 
 file includeDef.h
 list of headers and definitions used by the program OmegaMaxEnt (main source files: OmegaMaxEnt_main.cpp, OmegaMaxEnt_data.h, OmegaMaxEnt_data.cpp, graph_2D.h, graph_2D.cpp, generique.h, generique.cpp)
 
 Copyright (C) 2015 Dominic Bergeron (dominic.bergeron@usherbrooke.ca)
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef INCLUDEDEF_H
#define INCLUDEDEF_H

#include "fftw3.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <complex>
#include <string>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdio.h>
#include <sys/stat.h>
#include <random>
#include <gsl/gsl_linalg.h>
#include "armadillo"

#ifndef PI
#define PI acos((double)-1.0)
#endif

#ifndef EPSILON
#define EPSILON numeric_limits<double>::epsilon()
#endif

#ifndef DBL_MIN
#define DBL_MIN numeric_limits<double>::min()
#endif

#ifndef DBL_MAX
#define DBL_MAX numeric_limits<double>::max()
#endif

#ifndef INF
#define INF numeric_limits<double>::infinity()
#endif

using namespace std;

typedef complex<double> dcomplex;
typedef unsigned int  uint;

#endif
