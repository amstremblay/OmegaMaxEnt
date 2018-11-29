/*
 file generique.h
 deinition of class "generique" containing multipurpose numerical routines. The best way to use it is to derive a class from it.
 
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

#ifndef GENERIQUE_H
#define GENERIQUE_H

#include "includeDef.h"
#include "armadillo"

using namespace arma;

extern "C++" {
	
//! Note: to use the routines that take a pointer to a function as a parameter, use the static_cast() function to cast the pointer to a function of your class as a function of this class. Ex: fctPtr1 Ptr=static_cast<fctPtr1> (&myclass::myfunc);
class generique
{
 public:
	typedef double (generique::*fctPtr1) (double, void*[]);
	typedef dcomplex (generique::*cx_fctPtr1) (double, void*[]);
	
	typedef double (generique::*fctPtr) (double, double[]);
	typedef double (generique::*IntPtr)(double, double, int[], double[], fctPtr, double[2]);
	
	typedef dcomplex (generique::*complexFctPtr) (double, double[]);
	typedef dcomplex (generique::*complexIntPtr)(double, double, int[], double[], complexFctPtr, double[2]);
	
						
//@{
//! Constructor
	generique(){};
//! Destructor
	~generique(){};
//@}

//@{ root finding routines
//! root finding with the bisection method
	bool find_zero_bisect(fctPtr func, double params[], double root[], double lims[], double tol2);
	
//! root finding routine	
	bool find_zero(fctPtr func, double init[], double params[], double root[]);
	
//! root finding routine	
	bool find_zero(fctPtr func, double init[], double params[], double root[], double lims[]);
	
//! root finding routine with provided tolerance
	bool find_zero(fctPtr func, double init[], double params[], double root[], double lims[], double tol2);
//@}
	
//@{ Adaptive quadratic 1D and 2D integration routines
//! 1D integation routine (also used by 2D routines). In direct use for 1D functions, last two arguments should be NULL
	double quadInteg(fctPtr, double lims[2], double tol, int nbEval[], double params[], IntPtr, double limx[2]);
//! 1D subinterval integration step
	double quadStep(fctPtr, double lims[2], double vals[3], double tol, int nbEval[], double hmin, double params[], 
					IntPtr, double limx[2]);
//! 2D integration
	double quadInteg2D(fctPtr, double limx[2], double limy[2], double tol, int nbEval[], double params[]);
//! Inner (x directions) integral in 2D routine
	double innerInteg(double y, double tol, int nbEval[], double params[], fctPtr, double limx[2]);
	
	//! complex 1D integration routine whith a pointer array as parameter
	dcomplex cx_quadInteg1D(cx_fctPtr1, double lims[2], double tol, int nbEval[], void *params[]);
	//! 1D subinterval complex integration step
	dcomplex cx_quadStep1D(cx_fctPtr1, double lims[2], dcomplex vals[3], double tol, int nbEval[], double hmin, void *params[]);
	
//! another 1D integration routine whith a pointer array as parameter
	double quadInteg1D(fctPtr1, double lims[2], double tol, int nbEval[], void *params[]);
//! 1D subinterval integration step
	double quadStep1D(fctPtr1, double lims[2], double vals[3], double tol, int nbEval[], double hmin, void *params[]);
//@}
	
	//! compute the solution to the linear system of equations Ax=B using LU decomposition (uses gsl functions). vec is the vector type of the armadillo library
	int solve_LU(vec &X, mat &A, vec &B);
	
	//! compute the coefficients of a cubic spline using linear combinations of the first and second derivatives as the two mandatory additionnal constraints to the spline. x0 is the position vector, F the function vector, LC contains the two right-hand side values of the constraints, coeffs_LC(0) and coeffs_LC(1) are the coefficients of the first derivatives at the left and right boundary and coeffs_LC(2) and coeffs_LC(3) are the coefficients of the second derivatives at the same boundaries.
	void spline_coeffs_LC(vec x0, vec F, vec LC, vec coeffs_LC, vec &coeffs);
	
	//! compute the coefficients of a clamped spline.
	void spline_coefficients(double *vec_coeff,const double *xx,const double *yy,const double *fp,const int size);
	
	//! evaluate the spline at x	
	double Interp_1d_spline(const double x,const double *coeff,const double *xx,const int size);
	
	//! locate the index of the interval where x is located
	int locate(const double x,const double *xx,const int size);
	
	//! Integrate the spline
	double oned_spline_integral(const double *coeff,const double *xx,const double *yy,const int size);

	//! compute the coefficients of a cubic spline for V(x) of the form S_j(x)=a_j*(x-x_j)^3+b_j*(x-x_j)^2+c_j*(x-x_j)+d_j, where j is the interval's index
	//coeffs[0] and coeffs[1] contains the derivatives of V(x) at x_1 et x_N0, the size of x0 and V is N0 and the size of coeffs is 4*(N0-1)
	void spline_coeffs_rel(double *x0, double *V, int N0, double *coeffs);
	
	//! compute the coefficients of a cubic spline for V(x) of the form S_j(x)=a_j*x^3+b_j*x^2+c_j*x+d_j, where j is the interval's index
	//coeffs[0] and coeffs[1] contains the derivatives of V(x) at x_1 et x_N0, the size of x0 and V is N0 and the size of coeffs is 4*(N0-1)
	void spline_coeffs(double *x0, double *V, int N0, double *coeffs);
	
	//! return the value at x of the cubic spline computed with spline_coeffs()
	double spline(double x, double *x0, double *V, int N0, double *coeffs);

	//! pade approximant for an input value x after coef is computed with pade_cont_frac_coef (or pade_cont_frac_coef_rec),
	//! n: number of input points in the Pade approximant, x0: vector of the input points
	dcomplex pade(dcomplex x, int n, dcomplex *x0, dcomplex *coef);
	
	//! compute recursively the coefficients in the continued fraction representation of the Pade aproximant,
	//! func: vector of values of the function, x: vector of the input points, N: size of 'func' and 'x',
	//! coefficients are returned in 'coef'
	void pade_cont_frac_coef_rec(dcomplex *func, dcomplex *x, int N, dcomplex *coef) { for (int j=0; j<N; j++) coef[j]=pade_recursion(j, j, func, x, coef); }
	
	//! recursive formula in the Pade coefficients calculation
	dcomplex pade_recursion(int indFunc, int indx, dcomplex *func, dcomplex *x, dcomplex *coef);
	
	//! compute the coefficients in the continued fraction representation of the Pade aproximant,
	// func: vector of values of the function, x: vector of the input points, N: size of 'func' and 'x',
	// coefficients are returned in 'coef'
	inline void pade_cont_frac_coef(dcomplex *func, dcomplex *x, int N, dcomplex *coef);
	
	//! find the minimum of a function with the golden section method
	bool find_min_golden(fctPtr func, double lims[], double params[], double min[], double tol);
	
	//! find the minimum of a function
	bool find_min(fctPtr func, double init[], double params[], double min[]);
	
	//! absolute value for complex number (modulus)
	double fabs(dcomplex a){ return sqrt( real(a)*real(a) + imag(a)*imag(a) ); }
	
	//! find the roots of the polynomial Ax^2+Bx+C with high precision
	void find_roots_2pol(dcomplex coef[], dcomplex roots[]);
	
	//! find the roots of the polynomial Ax^3+Bx^2+Cx+D
	void find_roots_3pol(dcomplex coef[], dcomplex roots[]);
	
	//! find the roots of the polynomial Ax^4+Bx^3+Cx^2+Dx+F
	void find_roots_4pol(dcomplex coef[], dcomplex roots[]);
	
	//! simpson integration
	double simpson_integ(vec f, double dx);

	//! test if a vector contains NaN
	bool contains_NaN(double *v, long int size)
	{
		bool NaN_present=false;
		for (long int j=0; j<size; j++)
			if (v[j]!=v[j]) 
			{
				NaN_present=true;
				break;
			}
		return NaN_present;
	}
	
	bool copy_file(const string fname, const string in_dir, const string out_dir, const string out_name="");
		
};
	
} /* extern "C++" */

#endif /* !defined GENERIQUE_H */			

			
