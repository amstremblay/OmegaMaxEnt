/*
 file generique.cpp
 functions definitions for class "generique" defined in file generique.h
 
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

#include "includeDef.h"
#include "generique.h"

extern "C"
{
	// routines LAPACK  (descriptions sur http://www.netlib.org/lapack )
	// resout le systeme d'equations AX=B ou A est tridiagonal, reels double precision 	
	void dgtsv_(int *N, int *NRHS, double *DL, double *D, double *DU, double *B, int *LDB, int *INFO );
	// resout le systeme d'equations AX=B ou A est tridiagonal, complex double precision 	
	void zgtsv_(int *N, int *NRHS, dcomplex *DL, dcomplex *D, dcomplex *DU, dcomplex *B, int *LDB, int *INFO );
	// resout le systeme d'equations AX=B ou A a plusieurs diagonales, reals double precision
	void dgbsv_(int *N, int *KL, int *KU, int *NRHS, double *AB, int *LDAB, int *IPIV, double *B, int *LDB, int *INFO );
	// routine avancee pour resoudre le systeme d'equations AX=B ou A a plusieurs diagonales, reals double precision
	void dgbsvx_( char *FACT, char *TRANS, int *N, int *KL, int *KU, int *NRHS, double *AB, int *LDAB, double *AFB,
				 int *LDAFB, int *IPIV, char *EQUED, double *R, double *C, double *B, int *LDB, double *X, int *LDX,
				 double *RCOND, double *FERR, double *BERR, double *WORK, int *IWORK, int *INFO );
	// resout le systeme d'equations AX=B ou A a plusieurs diagonales, complex double precision
	void zgbsv_(int *N, int *KL, int *KU, int *NRHS, dcomplex *AB, int *LDAB, int *IPIV, dcomplex *B, int *LDB, int *INFO );
}

double generique::simpson_integ(vec f, double dx)
{
	int N=f.n_rows;
	
	if (!N%2)
	{
		cout<<"generique::simpson_integ(): number of points must be odd\n";
		return 0;
	}
	
	rowvec wgt(N);
	
	wgt(0)=1;
	wgt(1)=4;
	int j=1;
	while (j<(N-1)/2)
	{
		wgt(2*j)=2;
		wgt(2*j+1)=4;
		j++;
	}
	wgt(N-1)=1;
	
	mat sum=wgt*f;
	
	return sum(0)*dx/3;
}

vec generique::solve_LU(mat &A, vec &B)
{
	int N=B.n_rows;
	int j, sgn;
	
	gsl_matrix_view Ag= gsl_matrix_view_array (A.memptr(), N, N);
	gsl_vector_view Bg= gsl_vector_view_array (B.memptr(), N);
	gsl_vector *xg = gsl_vector_alloc (N);
	gsl_permutation * pg = gsl_permutation_alloc (N);
	gsl_linalg_LU_decomp (&Ag.matrix, pg, &sgn);
	gsl_linalg_LU_solve (&Ag.matrix, pg, &Bg.vector, xg);
	vec X=zeros<vec>(N);
	for (j=0; j<N; j++) X(j)=xg->data[j];
	gsl_permutation_free (pg);
	gsl_vector_free (xg);
	
	return X;
}


bool generique::copy_file(const string fname, const string in_dir, const string out_dir, const string out_name)
{
	string fname_in, fname_out;
	
	fname_in=in_dir;
	fname_out=out_dir;
	
	if (fname_in.back()!='/') fname_in+='/';
	if (fname_out.back()!='/') fname_out+='/';
	
	fname_in+=fname;
	if (out_name.size())
		fname_out+=out_name;
	else
		fname_out+=fname;
	
//	cout<<"fname_in: "<<fname_in<<endl;
//	cout<<"fname_out: "<<fname_out<<endl;
	
	ifstream file_in(fname_in);
	ofstream file_out(fname_out);
	
	if (file_in && file_out)
	{
		string str;
		getline(file_in,str);
		while (!file_in.eof())
		{
			file_out<<str<<'\n';
			getline(file_in,str);
		}
		file_in.close();
		file_out.close();
	}
	else
	{
		cout<<"generique::copy_file(): file "<<fname_in<<" or file "<<fname_out<<" does not exist.\n";
		return false;
	}
	
	return true;
}


dcomplex generique::cx_quadInteg1D(cx_fctPtr1 func, double lims[2], double tol, int nbEval[], void *params[])
{
	double a=lims[0], b=lims[1];
	
	//	double h = 0.13579*(b-a);
	double hmin=1e-19*fabs(b-a);
	
	const unsigned int nbPoints0=3;
	
	double x[] = {a, (a+b)/2.0, b};
	
	dcomplex y[nbPoints0];
	
	
	for (int l=0;l<nbPoints0; l++) y[l]=(this->*func)(x[l],params);
	nbEval[0] += nbPoints0;
	
	
	dcomplex sum=0.0;
	
	double stepLims[2];
	dcomplex stepVals[3];
	
	stepLims[0]=x[0];
	stepLims[1]=x[2];
	
	stepVals[0]=y[0];
	stepVals[1]=y[1];
	stepVals[2]=y[2];
	
	sum = cx_quadStep1D(func, stepLims, stepVals, tol, nbEval, hmin, params);
	
	return sum;
}

dcomplex generique::cx_quadStep1D(cx_fctPtr1 func,  double lims[2], dcomplex vals[3], double tol, int nbEval[], double hmin, void *params[])
{
	int nbEvalmax=100000000;
	double a=lims[0], b=lims[1];
	
	dcomplex sum1=0.0, sum2=0.0, sum=0.0;
	dcomplex fa=vals[0], fc=vals[1], fb=vals[2], fd, fe;
	
	double h=b-a, c=(a+b)/2.0;
	
	if (fabs(h) < hmin || c == a || c == b)
	{
		sum = h*fc;
		cout<<"quadStep1D(): smallest interval reached!\n";
		//		cout<<"a: "<<a<<"   c: "<<c<<"   b: "<<b<<"  ky: "<<params[0]<<'\n';
		return sum;
	}
	
	double d=(a + c)/2.0, e=(c + b)/2.0;
	
	
	fd=(this->*func)(d,params);
	fe=(this->*func)(e,params);
	nbEval[0] += 2;
	
	
	if (nbEval[0] > nbEvalmax)
	{
		cout<<"quadStep1D : maximum number of function calls exceeded!\n";
		sum = h*fc;
		//cout<<"a: "<<a<<"   c: "<<c<<"   b: "<<b<<"  ky: "<<params[0]<<'\n';
		return sum;
	}
	
	
	//Three point Simpson's rule.
	sum1 = (h/6.0)*(fa + 4.0*fc + fb);
	
	//Five point double Simpson's rule.
	sum2 = (h/12.0)*(fa + 4.0*fd + 2.0*fc + 4.0*fe + fb);
	
	// One step of Romberg extrapolation (or Richardson extrapolation).
	sum = sum2 + (sum2 - sum1)/15.0;
	
	// Check accuracy of integral over this subinterval.
	if (fabs(sum2 - sum)  <= tol)
		return sum;
	else // Subdivide into two subintervals.
	{
		double stepLims[2];
		dcomplex stepVals[3];
		
		stepLims[0]=a;
		stepLims[1]=c;
		
		stepVals[0]=fa;
		stepVals[1]=fd;
		stepVals[2]=fc;
		
		sum = cx_quadStep1D(func, stepLims, stepVals, tol, nbEval, hmin, params);
		
		stepLims[0]=c;
		stepLims[1]=b;
		
		stepVals[0]=fc;
		stepVals[1]=fe;
		stepVals[2]=fb;
		
		sum= sum + cx_quadStep1D(func, stepLims, stepVals, tol, nbEval, hmin, params);
	}
	
	return sum;
}


//! compute the coefficients of a cubic spline using linear combinations of the first and second derivatives as the two mandatory additionnal constraints to the spline. x is the position vector, F the function vector, LC contains the two right-hand side values of the constraints, coeffs_LC(0) and coeffs_LC(1) are the coefficients of the first derivatives at the left and right boundary and coeffs_LC(2) and coeffs_LC(3) are the coefficients of the second derivatives at the same boundaries. coeffs has the form [a_1 b_1 c_1 d_1 ... a_N b_N c_N d_N] and the spline values are given by a_i*(x-x0_i)^3+b_i*(x-x0_i)^2+c_i*(x-x0_i)+d_i in the ith interval.
void generique::spline_coeffs_LC(vec x0, vec F, vec LC, vec coeffs_LC, vec &coeffs)
{
	int N0=x0.n_rows;
	 
	int Ns=N0-1;
	
	mat A=zeros<mat>(3*Ns,3*Ns);
	vec b=zeros<vec>(3*Ns);
	 
	double x=x0(1)-x0(0);
	A(0,0)=pow(x,3);
	A(0,1)=pow(x,2);
	A(0,2)=x;
	 
	double xN=x0(N0-1)-x0(N0-2);
	A(1,2)=coeffs_LC(0);
	A(1,3*Ns-3)=3*coeffs_LC(1)*pow(xN,2);
	A(1,3*Ns-2)=2*coeffs_LC(1)*xN;
	A(1,3*Ns-1)=coeffs_LC(1);
	 
	A(2,1)=2*coeffs_LC(2);
	A(2,3*Ns-3)=6*coeffs_LC(3)*xN;
	A(2,3*Ns-2)=2*coeffs_LC(3);
	
	b(0)=F(1)-F(0);
	b(1)=LC(0);
	b(2)=LC(1);
	
 	int j;
	for (j=2; j<=Ns; j++)
	{
		x=x0(j)-x0(j-1);
	 
		A(3*j-3,3*j-3)=pow(x,3);
		A(3*j-3,3*j-2)=pow(x,2);
		A(3*j-3,3*j-1)=x;
	 
		A(3*j-2,3*j-6)=-6*x;
		A(3*j-2,3*j-5)=-2;
		A(3*j-2,3*j-2)=2;
	 
		A(3*j-1,3*j-6)=-3*pow(x,2);
		A(3*j-1,3*j-5)=-2*x;
		A(3*j-1,3*j-4)=-1;
		A(3*j-1,3*j-1)=1;
	 
	 	b(3*j-3)=F(j)-F(j-1);
	}
	
	vec coeffs_tmp=solve(A,b);
//	vec coeffs_tmp=solve_LU(A,b);
	coeffs.zeros(4*Ns);
	
	uvec indv=linspace<uvec>(0,Ns-1,Ns);
	coeffs.rows(4*indv)=coeffs_tmp.rows(3*indv);
	coeffs.rows(4*indv+1)=coeffs_tmp.rows(3*indv+1);
	coeffs.rows(4*indv+2)=coeffs_tmp.rows(3*indv+2);
	coeffs.rows(4*indv+3)=F.rows(0,Ns-1);
}

//! calcule les coefficients du spline cubique pour V(x), la taille de coeffs doit etre 4*(N0-1), en entree,
//coeffs[0] et coeffs[1] contiennent les derivees de V(x) a x_1 et x_N0, la taille de x0 et V est N0 et celle de coeffs est 4*(N0-1) 
void generique::spline_coeffs_rel(double *x0, double *V, int N0, double *coeffs)
{
	int j;
	
	int NS=N0-1;
	int N=3*NS-1;
	int KL=3;
	int KU=2;
	int NA=2*KL+KU+1;
	int SA=NA*N;
	
	double *A=new double[SA];
	for (j=0; j<SA; j++) A[j]=0;
	int *P=new int[N];
	double *coeffs_tmp=new double[N];
	for (j=0; j<N; j++) coeffs_tmp[j]=0;
	
	double dV1=coeffs[0], dVN0=coeffs[1];
	double x;
	
	x=x0[1]-x0[0];
	
	A[KL+KU]=(double) (x*x*x);
	A[KL+KU+1]=(double) (-6.0*x);
	A[KL+KU+2]=(double) (-3.0*x*x);
	
	A[NA+KL+KU-1]=(double) (x*x);
	A[NA+KL+KU]=(double) (-2.0);
	A[NA+KL+KU+1]=(double) (-2.0*x);
	
	coeffs_tmp[0]=(double) (V[1]-V[0]-dV1*x);
	coeffs_tmp[2]=(double)dV1;
/*	
	x=x0[2]-x0[1];
	
	A[2*NA+KL+KU+1]=(double) (x*x*x);
	A[2*NA+KL+KU+2]=(double) (-6.0*x);
	A[2*NA+KL+KU+3]=(double) (-3.0*x*x);
	
	A[3*NA+KL]=2.0;
	A[3*NA+KL+KU]=(double) (x*x);
	A[3*NA+KL+KU+1]=(double) (-2.0);
	A[3*NA+KL+KU+2]=(double) (-2.0*x);
	
	A[4*NA+KL]=1.0;
	A[4*NA+KL+1]=(double) x;
	A[4*NA+KL+KU+1]=-1.0;
	
	coeffs_tmp[3]=(double) (V[2]-V[1]);
*/	
	for (j=1; j<NS-1; j++)
	{
		x=x0[j+1]-x0[j];
		
		A[(3*j-1)*NA+KL+KU+1]=(double) (x*x*x);
		A[(3*j-1)*NA+KL+KU+2]=(double) (-6.0*x);
		A[(3*j-1)*NA+KL+KU+3]=(double) (-3.0*x*x);
		
		A[3*j*NA+KL]=2.0;
		A[3*j*NA+KL+KU]=(double) (x*x);
		A[3*j*NA+KL+KU+1]=(double) (-2.0);
		A[3*j*NA+KL+KU+2]=(double) (-2.0*x);
		
		A[(3*j+1)*NA+KL]=1.0;
		A[(3*j+1)*NA+KL+1]=(double) x;
		A[(3*j+1)*NA+KL+KU+1]=-1.0;
		
		coeffs_tmp[3*j]=(double) (V[j+1]-V[j]);
	}
	
	j=NS-1;
	x=x0[j+1]-x0[j];
	
	A[(3*j-1)*NA+KL+KU+1]=(double) (x*x*x);
	A[(3*j-1)*NA+KL+KU+2]=(double) (3.0*x*x);
	
	A[3*j*NA+KL]=2.0;
	A[3*j*NA+KL+KU]=(double) (x*x);
	A[3*j*NA+KL+KU+1]=(double) (2.0*x);
	
	A[(3*j+1)*NA+KL]=1.0;
	A[(3*j+1)*NA+KL+1]=(double) x;
	A[(3*j+1)*NA+KL+KU]=1.0;
	
	coeffs_tmp[3*j]=(double)(V[j+1]-V[j]);
	coeffs_tmp[3*j+1]=(double)dVN0;
	
	int NRHS=1;
	int INFO=0;
	dgbsv_(&N, &KL, &KU, &NRHS, A, &NA, P, coeffs_tmp, &N, &INFO );
	
	coeffs[0]=coeffs_tmp[0];
	coeffs[1]=coeffs_tmp[1];
	coeffs[2]=dV1;
	coeffs[3]=V[0];
	for (j=1; j<NS; j++)
	{
		coeffs[4*j]=coeffs_tmp[3*j-1];
		coeffs[4*j+1]=coeffs_tmp[3*j];
		coeffs[4*j+2]=coeffs_tmp[3*j+1];
		coeffs[4*j+3]=V[j];
	}
	
	if (INFO)	cout<<"spline_coeffs(): INFO:  "<<INFO<<'\n';
	
	delete [] A;
	delete [] P;
	delete [] coeffs_tmp;
}

//! calcule les coefficients du spline cubique pour V(x), la taille de coeffs doit etre 4*(N0-1), en entree,
//coeffs[0] et coeffs[1] contiennent les derivees de V(x) a x_1 et x_N0, la taille de x0 et V est N0 et celle de coeffs est 4*(N0-1) 
void generique::spline_coeffs(double *x0, double *V, int N0, double *coeffs)
{
	int j;
	
	int NS=N0-1;
	int N=4*NS;
	int NL=4;
	int NA=3*NL+1;
	
	double *A=new double[NA*N];
	int *P=new int[N];
	
	double dV1=coeffs[0], dVN0=coeffs[1];
	double x;
	
	x=x0[0];
	A[2*NL]=x*x*x;
	A[NL+3+NA]=x*x;
	A[NL+2+2*NA]=x;
	A[NL+1+3*NA]=1;
	A[NL+4*NA]=0;
	
	A[2*NL+1]=3*x*x;
	A[2*NL+NA]=2*x;
	A[NL+3+2*NA]=1;
	A[NL+2+3*NA]=0;
	A[NL+1+4*NA]=0;
	A[NL+5*NA]=0;
	
	x=x0[1];
	A[2*NL+2]=x*x*x;
	A[2*NL+1+NA]=x*x;
	A[2*NL+2*NA]=x;
	A[NL+3+3*NA]=1;
	A[NL+2+4*NA]=0;
	A[NL+1+5*NA]=0;
	A[NL+6*NA]=0;
	
	A[2*NL+3]=-3*x*x;
	A[2*NL+2+NA]=-2*x;
	A[2*NL+1+2*NA]=-1;
	A[2*NL+3*NA]=0;
	A[NL+3+4*NA]=3*x*x;
	A[NL+2+5*NA]=2*x;
	A[NL+1+6*NA]=1;
	A[NL+7*NA]=0;
	
	coeffs[0]=V[0];
	coeffs[1]=dV1;
	coeffs[2]=V[1];
	coeffs[3]=0;
	
	for (j=1; j<NS-1; j++)
	{
		A[2*NL+4+(4*j-4)*NA]=-6*x;
		A[2*NL+3+(4*j-3)*NA]=-2;
		A[2*NL+2+(4*j-2)*NA]=0;
		A[2*NL+1+(4*j-1)*NA]=0;
		A[2*NL+4*j*NA]=6*x;
		A[NL+3+(4*j+1)*NA]=2;
		A[NL+2+(4*j+2)*NA]=0;
		A[NL+1+(4*j+3)*NA]=0;
		A[NL+(4*j+4)*NA]=0;
		
		A[2*NL+4+(4*j-3)*NA]=0;
		A[2*NL+3+(4*j-2)*NA]=0;
		A[2*NL+2+(4*j-1)*NA]=0;
		A[2*NL+1+4*j*NA]=x*x*x;
		A[2*NL+(4*j+1)*NA]=x*x;
		A[NL+3+(4*j+2)*NA]=x;
		A[NL+2+(4*j+3)*NA]=1;
		A[NL+1+(4*j+4)*NA]=0;
		A[NL+(4*j+5)*NA]=0;
		
		x=x0[j+1];
		A[2*NL+4+(4*j-2)*NA]=0;
		A[2*NL+3+(4*j-1)*NA]=0;
		A[2*NL+2+4*j*NA]=x*x*x;
		A[2*NL+1+(4*j+1)*NA]=x*x;
		A[2*NL+(4*j+2)*NA]=x;
		A[NL+3+(4*j+3)*NA]=1;
		A[NL+2+(4*j+4)*NA]=0;
		A[NL+1+(4*j+5)*NA]=0;
		A[NL+(4*j+6)*NA]=0;
		
		A[2*NL+4+(4*j-1)*NA]=0;
		A[2*NL+3+4*j*NA]=-3*x*x;
		A[2*NL+2+(4*j+1)*NA]=-2*x;
		A[2*NL+1+(4*j+2)*NA]=-1;
		A[2*NL+(4*j+3)*NA]=0;
		A[NL+3+(4*j+4)*NA]=3*x*x;
		A[NL+2+(4*j+5)*NA]=2*x;
		A[NL+1+(4*j+6)*NA]=1;
		A[NL+(4*j+7)*NA]=0;		
		
		coeffs[4*j]=0;
		coeffs[4*j+1]=V[j];
		coeffs[4*j+2]=V[j+1];
		coeffs[4*j+3]=0;
	}
	j=NS-1;
	A[2*NL+4+(4*j-4)*NA]=-6*x;
	A[2*NL+3+(4*j-3)*NA]=-2;
	A[2*NL+2+(4*j-2)*NA]=0;
	A[2*NL+1+(4*j-1)*NA]=0;
	A[2*NL+4*j*NA]=6*x;
	A[NL+3+(4*j+1)*NA]=2;
	A[NL+2+(4*j+2)*NA]=0;
	A[NL+1+(4*j+3)*NA]=0;
	
	A[2*NL+4+(4*j-3)*NA]=0;
	A[2*NL+3+(4*j-2)*NA]=0;
	A[2*NL+2+(4*j-1)*NA]=0;
	A[2*NL+1+4*j*NA]=x*x*x;
	A[2*NL+(4*j+1)*NA]=x*x;
	A[NL+3+(4*j+2)*NA]=x;
	A[NL+2+(4*j+3)*NA]=1;
	
	x=x0[j+1];
	A[2*NL+4+(4*j-2)*NA]=0;
	A[2*NL+3+(4*j-1)*NA]=0;
	A[2*NL+2+4*j*NA]=x*x*x;
	A[2*NL+1+(4*j+1)*NA]=x*x;
	A[2*NL+(4*j+2)*NA]=x;
	A[NL+3+(4*j+3)*NA]=1;
	
	A[2*NL+4+(4*j-1)*NA]=0;
	A[2*NL+3+4*j*NA]=3*x*x;
	A[2*NL+2+(4*j+1)*NA]=2*x;
	A[2*NL+1+(4*j+2)*NA]=1;
	A[2*NL+(4*j+3)*NA]=0;
	
	coeffs[4*j]=0;
	coeffs[4*j+1]=V[j];
	coeffs[4*j+2]=V[j+1];
	coeffs[4*j+3]=dVN0;
	
	int NRHS=1;
	int INFO=0;
	dgbsv_(&N, &NL, &NL, &NRHS, A, &NA, P, coeffs, &N, &INFO );
	
	if (INFO)	cout<<"spline_coeffs(): INFO:  "<<INFO<<'\n';
	
	delete [] A;
	delete [] P;
}

//! retourne la valeur du spline cubique dont les coefficients on ete calcules avec spline_coeffs()
double generique::spline(double x, double *x0, double *V, int N0, double *coeffs)
{
	int ind;
	double a, b, c, d;
	double dx=(x0[N0-1]-x0[0])/(N0-1);
	
	ind=(int)floor(x/dx);
	if (ind==N0-1) return V[N0-1];
	if (x==x0[ind]) return V[ind];
	a=coeffs[4*ind];
	b=coeffs[4*ind+1];
    c=coeffs[4*ind+2];
    d=coeffs[4*ind+3];
	
    return a*x*x*x+b*x*x+c*x+d;
}

void generique::spline_coefficients(double *vec_coeff,const double *xx,const double *yy,const double *fp,const int size)
{
	//Spline with given first derivatives
	double FP0 = fp[0];
	double FPN = fp[1];
	
	int n = size-1;
	
	double *h,*alpha,*mu,*z,*l;
	h = new double[n];
	alpha = new double[n+1];
	mu = new double[n];
	z = new double[n+1];
	l = new double[n+1];
	
	for (int r = 0; r < n; r++ )
	{
		h[r] = xx[r+1]-xx[r];
		if (r == 0)
			alpha[0] = 3.0*(yy[1]-yy[0])/h[0] - 3.0*FP0;
		else
			alpha[r] = 3.0/h[r]*(yy[r+1]-yy[r]) - 3.0/h[r-1]*(yy[r]-yy[r-1] );
	}
	alpha[n] = 3.0*FPN - 3.0*( yy[n] - yy[n-1] )/h[n-1];
	
	l[0] = 2*h[0];
	mu[0] = 0.5;
	z[0] = alpha[0]/l[0];
	
	for (int r = 1;r<n;r++)
	{
		l[r] = 2.0*(xx[r+1]-xx[r-1])-h[r-1]*mu[r-1];
		mu[r] = h[r]/l[r];
		z[r]  = (alpha[r]-h[r-1]*z[r-1]  )/l[r];
	}
	l[n] = h[n-1]*(2.0-mu[n-1]);
	z[n] = ( alpha[n] - h[n-1]*z[n-1]  )/l[n];
	
	double *c,*b,*d;
	c = new double[n+1];
	b = new double[n];
	d = new double[n];
	
	c[n] = z[n];
	for (int r = 0;r<n;r++)
	{
		c[(n-1)-r] = z[(n-1)-r] - mu[(n-1)-r]*c[n-r];
		b[(n-1)-r] = (yy[n-r]-yy[(n-1)-r])/h[(n-1)-r] - h[(n-1)-r]*( c[n-r]+2.0*c[(n-1)-r] )/3.0;
		d[(n-1)-r] = ( c[n-r] - c[(n-1)-r] )/(3.0*h[(n-1)-r]);
	}
	
//	cout<<setiosflags(ios::left)<<setprecision(6);
	for (int r = 0;r<n;r++)
	{
		vec_coeff[4*r] = yy[r];
		vec_coeff[4*r+1] = b[r];
		vec_coeff[4*r+2] = c[r];
		vec_coeff[4*r+3] = d[r];
//		cout<<setw(20)<<vec_coeff[4*r]<<setw(20)<<vec_coeff[4*r+1]<<setw(20)<<vec_coeff[4*r+2]<<setw(20)<<vec_coeff[4*r+3]<<endl;
	}
	
	delete[] h;
	delete[] alpha;
	delete[] mu;
	delete[] z;
	delete[] l;
	delete[] b;
	delete[] c;
	delete[] d;
	
}

double generique::Interp_1d_spline(const double x,const double *coeff,const double *xx,const int size)
{
	int j = locate(x,xx,size);
	double X = x-xx[j];
	return coeff[4*j] + coeff[4*j+1]*X + coeff[4*j+2]*X*X + coeff[4*j+3]*X*X*X;
}


int generique::locate(const double x,const double *xx,const int size)
{
	//Find for which j x_j <= x < x_j+1
    int ju,jm,jl;
    int arrSize = size;
    bool ascnd=(xx[arrSize-1] >= xx[0]); //True if ascending order of table, false otherwise.
	
    jl=0; //Initialize lower
    ju=arrSize-1; //and upper limits.
	
    while (ju-jl > 1)
    { //If we are not yet done,
		jm = (ju+jl) >> 1; //compute a midpoint,
		if (x >= xx[jm] == ascnd)
			jl=jm; //and replace either the lower limit
		else
			ju=jm; //or the upper limit, as appropriate.
    } //Repeat until the test condition is satisfied.
	
    return jl;
}


double generique::oned_spline_integral(const double *coeff,const double *xx,const double *yy,const int size)
{
	double sum = 0.0;
	int n = size-1;
	for (int r = 0;r<n;r++)
		sum += 0.5*( yy[r+1]+yy[r] )*(xx[r+1]-xx[r]) - 1.0/12.0*( coeff[4*(r+1)+2]+coeff[4*r+2]   )*(xx[r+1]-xx[r])*(xx[r+1]-xx[r])*(xx[r+1]-xx[r]);
	return sum;
}


//version de chi.cpp
//zero finding routine
bool generique::find_zero(fctPtr func, double init[], double params[], double root[])
{
	double tol=1.0e-16, tol2=1.0e-10, eps=10.0*EPSILON;
	int nbItermax=40;
	int nbIter=0;
	int nbEvaln=0;
	int ind1, ind2, ind3, indm, indp;
	int indfar=0, indmid;
	double denom=0.0, a=0.0, b=0.0, c=0.0;
	double xtmpp, xtmpm;
	double dy, dyprec, dytol=0.01;
	
	double x0=init[0], xmin=init[1], xmax=init[2];
	
	
	double ymin=(this->*func)(xmin,params);
	double ymax=(this->*func)(xmax,params);
	nbEvaln+=2;
	
	
	double x[]={x0, xmin, xmax}, y[]={0.0, ymin, ymax};
	
	if ( fabs(ymin) < tol)
	{
		root[0]=xmin;
		return true;
	}
	
	if ( fabs(ymax) < tol)
	{
		root[0]=xmax;
		return true;
	}
	
	if (ymin*ymax>0)
	{
		cout<<"find_zero(): pas de changement de signe dans l'intervalle, essaie par Newton-Raphson\n";
		
		y[0]=(this->*func)(x[0],params);
		nbEvaln++;
		
		dy=y[0];
		
		if ( fabs(y[0]) < tol)
		{
			root[0]=x[0];
			return true;
		}
		
		double xtmp;
		if ( y[0]*y[2]<0 )
		{
			cout<<"find_zero(): changement de signe trouve\n";
			//			cout<<"y:  "<<y[0]<<'\n';
			xtmp=x[0];
			x[0]=(x[0]+x[2])/2;
			x[1]=xtmp;
			y[1]=y[0];
		}
		else
		{
			double ytmp, xtmp1, ytmp1;
			
			ytmp=y[0];
			if (fabs(y[1])<fabs(ytmp))	ytmp=y[1], xtmp=x[1];
			if (fabs(y[2])<fabs(ytmp))	ytmp=y[2], xtmp=x[2];
			
			xtmp1=xtmp+10*eps;
			ytmp1=(this->*func)(xtmp1,params);
			nbEvaln++;
			
			xtmp1=xtmp - ytmp*((xtmp1-xtmp)/(ytmp1-ytmp));
			ytmp1=(this->*func)(xtmp1,params);
			nbEvaln++;
			
			//			cout<<setiosflags(ios::left);
			//			cout<<setw(30)<<xtmp1<<ytmp1<<'\n';
			
			nbIter++;
			
			while (fabs(ytmp1)>tol && nbIter<nbItermax && ytmp1*ytmp>0 && isfinite(xtmp1))
			{
				xtmp=xtmp1;
				ytmp=ytmp1;
				
				xtmp1=xtmp+10*eps;
				ytmp1=(this->*func)(xtmp1,params);
				nbEvaln++;
				
				xtmp1=xtmp - ytmp*((xtmp1-xtmp)/(ytmp1-ytmp));
				ytmp1=(this->*func)(xtmp1,params);
				nbEvaln++;
				
				//				cout<<setw(30)<<xtmp1<<ytmp1<<'\n';
				
				nbIter++;
			}
			if ( fabs(ytmp1) < tol)
			{
				cout<<"racine trouvee apres "<<nbIter<<" iterations de Newton-Raphson\n";
				root[0]=xtmp1;
				return true;
			}
			if (ytmp1*ytmp<0)
			{
				cout<<"changement de signe trouve apres "<<nbIter<<" iterations de Newton-Raphson\n";
				x[0]=(xtmp+xtmp1)/2;
				x[1]=xtmp;
				y[1]=ytmp;
				x[2]=xtmp1;
				y[2]=ytmp1;
			}
			if (nbIter>=nbItermax)
			{
				cout<<"nombre d'iterations maximal atteint, derniere valeur obtenue:  "<<fabs(ytmp1)<<'\n';
				return false;
			}
			if (!isfinite(xtmp1))
			{
				cout<<"valeur infinie de x\n";
				return false;
			}
		}
	}
	
	
	if (fabs(xmax-xmin)<eps)
	{
		root[0]=xmin;
		return true;
	}
	
	y[0]=(this->*func)(x[0],params);
	nbEvaln++;
	
	dy=y[0];
	
	if ( fabs(y[0]) < tol)
	{
		root[0]=x[0];
		return true;
	}
	
	//find which points have the same sign
	if ( y[0]*y[1]>0 )
		ind1=0, ind2=1;
	else if (y[0]*y[2]>0)
		ind1=0, ind2=2;
	else
		ind1=1, ind2=2;
	
	ind3=3-ind1-ind2;
	
	
	if (x[ind3]>x[ind1])
	{
		if (x[ind1]>x[ind2])
			indfar=ind2, indmid=ind1;
		else
			indfar=ind1, indmid=ind2;
	}
	else
	{
		if (x[ind1]<x[ind2])
			indfar=ind2, indmid=ind1;
		else
			indfar=ind1, indmid=ind2;
	}
	
	//linear approximation to the root
	x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
	
	
	y[indfar]=(this->*func)(x[indfar],params);
	nbEvaln++;
	
	dy=y[indfar];
	
	
	while (fabs(dy)>tol && nbIter<nbItermax)
	{
		//find which points have the same sign
		if ( y[0]*y[1]>0 )
			ind1=0, ind2=1;
		else if (y[0]*y[2]>0)
			ind1=0, ind2=2;
		else
			ind1=1, ind2=2;
		
		ind3=3-ind1-ind2;
		
		if (x[ind3]>x[ind1])
		{
			if (x[ind1]>x[ind2])
				indfar=ind2, indmid=ind1;
			else
				indfar=ind1, indmid=ind2;
			
			indm=indmid, indp=ind3;
		}
		else
		{
			if (x[ind1]<x[ind2])
				indfar=ind2, indmid=ind1;
			else
				indfar=ind1, indmid=ind2;
			
			indm=ind3, indp=indmid;
		}
		
//		 cout<<"iteration:  "<<nbIter<<'\n';
//		 cout<<x[ind3]<<"   "<<y[ind3]<<'\n';
//		 cout<<x[indmid]<<"   "<<y[indmid]<<'\n';
//		 cout<<x[indfar]<<"   "<<y[indfar]<<'\n';

		
		denom=-x[0]*x[0]*x[2]+x[0]*x[0]*x[1]-x[0]*x[1]*x[1]+x[0]*x[2]*x[2]-x[2]*x[2]*x[1]+x[2]*x[1]*x[1];
		
		if (fabs(denom)>0.0)
		{
			a=(x[0]*y[2]-x[0]*y[1]+x[2]*y[1]-y[2]*x[1]-y[0]*x[2]+y[0]*x[1])/denom;
			b=-(-y[2]*x[1]*x[1]+x[1]*x[1]*y[0]-x[2]*x[2]*y[0]-y[1]*x[0]*x[0]+y[2]*x[0]*x[0]+y[1]*x[2]*x[2])/denom;
			c=(y[2]*x[0]*x[0]*x[1]-y[1]*x[0]*x[0]*x[2]-x[1]*x[1]*x[0]*y[2]+x[2]*x[2]*x[0]*y[1]+x[1]*x[1]*y[0]*x[2]-x[2]*x[2]*y[0]*x[1])/denom;
		}
		
		// if the root is real
		if ( (b*b-4.0*a*c)>0.0 && fabs(denom)>0.0)
		{
			xtmpp=(-b+sqrt(b*b-4.0*a*c))/(2*a);
			xtmpm=(-b-sqrt(b*b-4.0*a*c))/(2*a);
			
			if (xtmpp>x[indm] && xtmpp<x[indp])
			{
				x[indfar]=xtmpp;
			}
			else if (xtmpm>x[indm] && xtmpm<x[indp])
			{
				x[indfar]=xtmpm;
			}
			else if (fabs(y[ind3]-y[indmid])>tol)
			{
				x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
			}
			else
			{
				root[0]=x[indmid];
				return true;
			}
		}
		else  //linearly interpolate
		{
			if (fabs(y[ind3]-y[indmid])>tol)
			{
				x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
			}
			else
			{
				root[0]=x[indmid];
				return true;
			}
			
		}
		
		y[indfar]=(this->*func)(x[indfar],params);
		nbEvaln++;
		
		dyprec=fabs(dy);
		dy=y[indfar];
		
		
		if (fabs(dyprec-fabs(dy))<dytol*dyprec || dyprec<fabs(dy))
		{
			x[indfar]=(x[ind3]+x[indmid])/2.0;
			y[indfar]=(this->*func)(x[indfar],params);
			nbEvaln++;
			dyprec=fabs(dy);
			dy=y[indfar];
		}
		
		//		if (mpi_rank==0) cout<<"iteration: "<<nbIter<<"  x: "<<x[indfar]<<"  y: "<<y[indfar]<<'\n';
		
		nbIter++;
		
	}
	
	cout<<"iteration "<<nbIter-1<<" de l'interpolation quadratique:  x: "<<x[indfar]<<"  y: "<<y[indfar]<<'\n';
	
	if (nbIter>=nbItermax)
	{
		cout<<"find_zero: nombre d'iterations maximal atteint"<<'\n';
		if (fabs(y[0])>tol2 && fabs(y[1])>tol2 && fabs(y[2])>tol2)
			return false;
	}
	
//	root[0]=x[indfar];
	
	ind1=0;
	root[0]=x[ind1];
	if (fabs(y[1])<fabs(y[ind1])) 
	{
		root[0]=x[1];
		ind1=1;
	}
	if (fabs(y[2])<fabs(y[ind1])) root[0]=x[2];
	
	return true;
}

bool generique::find_zero_bisect(fctPtr func, double params[], double root[], double lims[], double tol)
{
	double eps=2*EPSILON;
	int nbItermax=1000;
	int nbIter=0;
	
	double x1=lims[0], x2=lims[1];
	double x=(x1+x2)/2;
	double y=(this->*func)(x,params);
	double y1=(this->*func)(x1,params);
	double y2=(this->*func)(x2,params);
	
//	cout<<setw(20)<<x1<<setw(20)<<y1<<setw(20)<<x<<setw(20)<<y<<setw(20)<<x2<<setw(20)<<y2<<endl;
	
	if (lims[1]<lims[0])
	{
		cout<<"find_zero_bisect(): lims[1]<lims[0]\n";
		return false;
	}
	
	if (y1*y2>0)
	{
		cout<<"find_zero_bisect(): no sign change in the provided interval\n";
		return false;
	}
	
	while ( fabs(y)>tol && nbIter<nbItermax && x2-x1>eps)
	{
		if (y*y1<0)
		{
			x2=x;
			y2=(this->*func)(x2,params);
			x=(x1+x2)/2;
			y=(this->*func)(x,params);
		}
		else
		{
			x1=x;
			y1=(this->*func)(x1,params);
			x=(x1+x2)/2;
			y=(this->*func)(x,params);
		}
//		cout<<setw(20)<<x1<<setw(20)<<y1<<setw(20)<<x<<setw(20)<<y<<setw(20)<<x2<<setw(20)<<y2<<endl;
		
		nbIter++;
	}
	
	if (nbIter>nbItermax && abs(y)>tol)
	{
		cout<<"find_zero_bisect(): maximum number of iterations reached\n";
		cout<<"smallest value found: "<<fabs(y)<<endl;
		return false;
	}
	
	root[0]=x;
	
	return true;
}

//zero finding routine
bool generique::find_zero(fctPtr func, double init[], double params[], double root[], double lims[])
{
	double tol=1.0e-14, tol2=1.0e-6, eps=2.0*EPSILON;
	int nbItermax=30;
	int nbIter=0;
	int nbEvaln=0;
	int ind1, ind2, ind3, indm, indp;
	int indfar=0, indmid;
	double denom=0.0, a=0.0, b=0.0, c=0.0;
	double xtmpp, xtmpm;
	double dy, dyprec, dytol=0.01;
	
	double x0=init[0], xmin=init[1], xmax=init[2];
	
	
	double ymin=(this->*func)(xmin,params);
	double ymax=(this->*func)(xmax,params);
	nbEvaln+=2;
	
	
	double x[]={x0, xmin, xmax}, y[]={0.0, ymin, ymax};
	
	if ( fabs(ymin) < tol)
	{
		root[0]=xmin;
		return true;
	}
	
	if ( fabs(ymax) < tol)
	{
		root[0]=xmax;
		return true;
	}
	
	
	if (ymin*ymax>0)
	{
		cout<<"find_zero(): pas de changement de signe dans l'intervalle, essaie par Newton-Raphson\n";
		
		y[0]=(this->*func)(x[0],params);
		nbEvaln++;
		
		dy=y[0];
		
		if ( fabs(y[0]) < tol)
		{
			root[0]=x[0];
			return true;
		}
		
		double xtmp;
		if ( y[0]*y[2]<0 )
		{
			cout<<"find_zero(): changement de signe trouve\n";
			//			cout<<"y:  "<<y[0]<<'\n';
			xtmp=x[0];
			x[0]=(x[0]+x[2])/2;
			x[1]=xtmp;
			y[1]=y[0];
		}
		else
		{
			double ytmp, xtmp1, ytmp1;
			
			ytmp=y[0];
			if (fabs(y[1])<fabs(ytmp))	ytmp=y[1], xtmp=x[1];
			if (fabs(y[2])<fabs(ytmp))	ytmp=y[2], xtmp=x[2];
			
			xtmp1=xtmp+10*eps;
			ytmp1=(this->*func)(xtmp1,params);
			nbEvaln++;
			
			xtmp1=xtmp - ytmp*((xtmp1-xtmp)/(ytmp1-ytmp));
			ytmp1=(this->*func)(xtmp1,params);
			nbEvaln++;
			
			//			cout<<setiosflags(ios::left);
			//			cout<<setw(30)<<xtmp1<<ytmp1<<'\n';
			
			nbIter++;
			
			while (fabs(ytmp1)>tol && nbIter<nbItermax && ytmp1*ytmp>0 && isfinite(xtmp1))
			{
				xtmp=xtmp1;
				ytmp=ytmp1;
				
				xtmp1=xtmp+10*eps;
				ytmp1=(this->*func)(xtmp1,params);
				nbEvaln++;
				
				xtmp1=xtmp - ytmp*((xtmp1-xtmp)/(ytmp1-ytmp));
				ytmp1=(this->*func)(xtmp1,params);
				nbEvaln++;
				
				//				cout<<setw(30)<<xtmp1<<ytmp1<<'\n';
				
				nbIter++;
			}
			if ( fabs(ytmp1) < tol)
			{
				cout<<"racine trouvee apres "<<nbIter<<" iterations de Newton-Raphson\n";
				root[0]=xtmp1;
				return true;
			}
			if (ytmp1*ytmp<0)
			{
				cout<<"changement de signe trouve apres "<<nbIter<<" iterations de Newton-Raphson\n";
				x[0]=(xtmp+xtmp1)/2;
				x[1]=xtmp;
				y[1]=ytmp;
				x[2]=xtmp1;
				y[2]=ytmp1;
			}
			if (nbIter>=nbItermax)
			{
				cout<<"nombre d'iterations maximal atteint"<<fabs(ytmp1)<<'\n';
				return false;
			}
			if (!isfinite(xtmp1))
			{
				cout<<"valeur infinie de x\n";
				return false;
			}
		}
	}
	
	if (x[1]<lims[0] || x[1]>lims[1] || x[2]<lims[0] || x[2]>lims[1]) 
	{
		cout<<"pas de racine dans l'intervalle specifie\n";
		return false; 
	}
	
	
	if (fabs(xmax-xmin)<eps)
	{
		root[0]=xmin;
		return true;
	}
	
	y[0]=(this->*func)(x[0],params);
	nbEvaln++;
	
	dy=y[0];
	
	if ( fabs(y[0]) < tol)
	{
		root[0]=x[0];
		return true;
	}
	
	//find which points have the same sign
	if ( y[0]*y[1]>0 )
		ind1=0, ind2=1;
	else if (y[0]*y[2]>0)
		ind1=0, ind2=2;
	else
		ind1=1, ind2=2;
	
	ind3=3-ind1-ind2;
	
	
	if (x[ind3]>x[ind1])
	{
		if (x[ind1]>x[ind2])
			indfar=ind2, indmid=ind1;
		else
			indfar=ind1, indmid=ind2;
	}
	else
	{
		if (x[ind1]<x[ind2])
			indfar=ind2, indmid=ind1;
		else
			indfar=ind1, indmid=ind2;
	}
	
	//linear approximation to the root
	x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
	
	
	y[indfar]=(this->*func)(x[indfar],params);
	nbEvaln++;
	
	dy=y[indfar];
	
	while (fabs(dy)>tol && nbIter<nbItermax)
	{
		//find which points have the same sign
		if ( y[0]*y[1]>0 )
			ind1=0, ind2=1;
		else if (y[0]*y[2]>0)
			ind1=0, ind2=2;
		else
			ind1=1, ind2=2;
		
		ind3=3-ind1-ind2;
		
		if (x[ind3]>x[ind1])
		{
			if (x[ind1]>x[ind2])
				indfar=ind2, indmid=ind1;
			else
				indfar=ind1, indmid=ind2;
			
			indm=indmid, indp=ind3;
		}
		else
		{
			if (x[ind1]<x[ind2])
				indfar=ind2, indmid=ind1;
			else
				indfar=ind1, indmid=ind2;
			
			indm=ind3, indp=indmid;
		}
		
		/*
		 cout<<"iteration:  "<<nbIter<<'\n';
		 cout<<x[ind3]<<"   "<<y[ind3]<<'\n';
		 cout<<x[indmid]<<"   "<<y[indmid]<<'\n';
		 cout<<x[indfar]<<"   "<<y[indfar]<<'\n';
		 */
		
		denom=-x[0]*x[0]*x[2]+x[0]*x[0]*x[1]-x[0]*x[1]*x[1]+x[0]*x[2]*x[2]-x[2]*x[2]*x[1]+x[2]*x[1]*x[1];
		
		if (fabs(denom)>0.0)
		{
			a=(x[0]*y[2]-x[0]*y[1]+x[2]*y[1]-y[2]*x[1]-y[0]*x[2]+y[0]*x[1])/denom;
			b=-(-y[2]*x[1]*x[1]+x[1]*x[1]*y[0]-x[2]*x[2]*y[0]-y[1]*x[0]*x[0]+y[2]*x[0]*x[0]+y[1]*x[2]*x[2])/denom;
			c=(y[2]*x[0]*x[0]*x[1]-y[1]*x[0]*x[0]*x[2]-x[1]*x[1]*x[0]*y[2]+x[2]*x[2]*x[0]*y[1]+x[1]*x[1]*y[0]*x[2]-x[2]*x[2]*y[0]*x[1])/denom;
		}
		
		// if the root is real
		if ( (b*b-4.0*a*c)>0.0 && fabs(denom)>0.0)
		{
			xtmpp=(-b+sqrt(b*b-4.0*a*c))/(2*a);
			xtmpm=(-b-sqrt(b*b-4.0*a*c))/(2*a);
			
			if (xtmpp>x[indm] && xtmpp<x[indp])
			{
				x[indfar]=xtmpp;
			}
			else if (xtmpm>x[indm] && xtmpm<x[indp])
			{
				x[indfar]=xtmpm;
			}
			else if (fabs(y[ind3]-y[indmid])>tol)
			{
				x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
			}
			else
			{
				root[0]=x[indmid];
				return true;
			}
		}
		else  //linearly interpolate
		{
			if (fabs(y[ind3]-y[indmid])>tol)
			{
				x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
			}
			else
			{
				root[0]=x[indmid];
				return true;
			}
			
		}
		
		y[indfar]=(this->*func)(x[indfar],params);
		nbEvaln++;
		
		dyprec=fabs(dy);
		dy=y[indfar];
		
		
		if (fabs(dyprec-fabs(dy))<dytol*dyprec || dyprec<fabs(dy))
		{
			x[indfar]=(x[ind3]+x[indmid])/2.0;
			y[indfar]=(this->*func)(x[indfar],params);
			nbEvaln++;
			dyprec=fabs(dy);
			dy=y[indfar];
		}
		
		//cout<<"iteration: "<<nbIter<<"  x: "<<x[indfar]<<"  y: "<<y[indfar]<<'\n';
		
		nbIter++;
		
	}
	
	//	cout<<"iteration "<<nbIter-1<<" de l'interpolation quadratique:  x: "<<x[indfar]<<"  y: "<<y[indfar]<<'\n';
	
	if (nbIter>=nbItermax)
	{
//		cout<<"find_zero: nombre d'iterations maximal atteint"<<'\n';
		if (fabs(y[0])>tol2 && fabs(y[1])>tol2 && fabs(y[2])>tol2)
			return false;
	}
	
//	root[0]=x[indfar];
	
	ind1=0;
	root[0]=x[ind1];
	if (fabs(y[1])<fabs(y[ind1])) 
	{
		root[0]=x[1];
		ind1=1;
	}
	if (fabs(y[2])<fabs(y[ind1])) root[0]=x[2];
	
	
	return true;
}

//zero finding routine
bool generique::find_zero(fctPtr func, double init[], double params[], double root[], double lims[], double tol2)
{
	double tol=1.0e-14, eps=2.0*EPSILON;
	int nbItermax=30;
	int nbIter=0;
	int nbEvaln=0;
	int ind1, ind2, ind3, indm, indp;
	int indfar=0, indmid;
	double denom=0.0, a=0.0, b=0.0, c=0.0;
	double xtmpp, xtmpm;
	double dy, dyprec, dytol=0.01;
	
	double x0=init[0], xmin=init[1], xmax=init[2];
	
	
	double ymin=(this->*func)(xmin,params);
	double ymax=(this->*func)(xmax,params);
	nbEvaln+=2;
	
	
	double x[]={x0, xmin, xmax}, y[]={0.0, ymin, ymax};
	
	if ( fabs(ymin) < tol)
	{
		root[0]=xmin;
		return true;
	}
	
	if ( fabs(ymax) < tol)
	{
		root[0]=xmax;
		return true;
	}
	
	
	if (ymin*ymax>0)
	{
		cout<<"find_zero(): pas de changement de signe dans l'intervalle, essaie par Newton-Raphson\n";
		
		y[0]=(this->*func)(x[0],params);
		nbEvaln++;
		
		dy=y[0];
		
		if ( fabs(y[0]) < tol)
		{
			root[0]=x[0];
			return true;
		}
		
		double xtmp;
		if ( y[0]*y[2]<0 )
		{
			cout<<"find_zero(): changement de signe trouve\n";
			//			cout<<"y:  "<<y[0]<<'\n';
			xtmp=x[0];
			x[0]=(x[0]+x[2])/2;
			x[1]=xtmp;
			y[1]=y[0];
		}
		else
		{
			double ytmp, xtmp1, ytmp1;
			
			ytmp=y[0];
			if (fabs(y[1])<fabs(ytmp))	ytmp=y[1], xtmp=x[1];
			if (fabs(y[2])<fabs(ytmp))	ytmp=y[2], xtmp=x[2];
			
			xtmp1=xtmp+10*eps;
			ytmp1=(this->*func)(xtmp1,params);
			nbEvaln++;
			
			xtmp1=xtmp - ytmp*((xtmp1-xtmp)/(ytmp1-ytmp));
			ytmp1=(this->*func)(xtmp1,params);
			nbEvaln++;
			
			//			cout<<setiosflags(ios::left);
			//			cout<<setw(30)<<xtmp1<<ytmp1<<'\n';
			
			nbIter++;
			
			while (fabs(ytmp1)>tol && nbIter<nbItermax && ytmp1*ytmp>0 && isfinite(xtmp1))
			{
				xtmp=xtmp1;
				ytmp=ytmp1;
				
				xtmp1=xtmp+10*eps;
				ytmp1=(this->*func)(xtmp1,params);
				nbEvaln++;
				
				xtmp1=xtmp - ytmp*((xtmp1-xtmp)/(ytmp1-ytmp));
				ytmp1=(this->*func)(xtmp1,params);
				nbEvaln++;
				
				//				cout<<setw(30)<<xtmp1<<ytmp1<<'\n';
				
				nbIter++;
			}
			if ( fabs(ytmp1) < tol)
			{
				cout<<"racine trouvee apres "<<nbIter<<" iterations de Newton-Raphson\n";
				root[0]=xtmp1;
				return true;
			}
			if (ytmp1*ytmp<0)
			{
				cout<<"changement de signe trouve apres "<<nbIter<<" iterations de Newton-Raphson\n";
				x[0]=(xtmp+xtmp1)/2;
				x[1]=xtmp;
				y[1]=ytmp;
				x[2]=xtmp1;
				y[2]=ytmp1;
			}
			if (nbIter>=nbItermax)
			{
				cout<<"nombre d'iterations maximal atteint"<<fabs(ytmp1)<<'\n';
				return false;
			}
			if (!isfinite(xtmp1))
			{
				cout<<"valeur infinie de x\n";
				return false;
			}
		}
	}
	
	if (x[1]<lims[0] || x[1]>lims[1] || x[2]<lims[0] || x[2]>lims[1])
	{
		cout<<"pas de racine dans l'intervalle specifie\n";
		return false;
	}
	
	
	if (fabs(xmax-xmin)<eps)
	{
		root[0]=xmin;
		return true;
	}
	
	y[0]=(this->*func)(x[0],params);
	nbEvaln++;
	
	dy=y[0];
	
	if ( fabs(y[0]) < tol)
	{
		root[0]=x[0];
		return true;
	}
	
	//find which points have the same sign
	if ( y[0]*y[1]>0 )
		ind1=0, ind2=1;
	else if (y[0]*y[2]>0)
		ind1=0, ind2=2;
	else
		ind1=1, ind2=2;
	
	ind3=3-ind1-ind2;
	
	
	if (x[ind3]>x[ind1])
	{
		if (x[ind1]>x[ind2])
			indfar=ind2, indmid=ind1;
		else
			indfar=ind1, indmid=ind2;
	}
	else
	{
		if (x[ind1]<x[ind2])
			indfar=ind2, indmid=ind1;
		else
			indfar=ind1, indmid=ind2;
	}
	
	//linear approximation to the root
	x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
	
	
	y[indfar]=(this->*func)(x[indfar],params);
	nbEvaln++;
	
	dy=y[indfar];
	
	while (fabs(dy)>tol && nbIter<nbItermax)
	{
		//find which points have the same sign
		if ( y[0]*y[1]>0 )
			ind1=0, ind2=1;
		else if (y[0]*y[2]>0)
			ind1=0, ind2=2;
		else
			ind1=1, ind2=2;
		
		ind3=3-ind1-ind2;
		
		if (x[ind3]>x[ind1])
		{
			if (x[ind1]>x[ind2])
				indfar=ind2, indmid=ind1;
			else
				indfar=ind1, indmid=ind2;
			
			indm=indmid, indp=ind3;
		}
		else
		{
			if (x[ind1]<x[ind2])
				indfar=ind2, indmid=ind1;
			else
				indfar=ind1, indmid=ind2;
			
			indm=ind3, indp=indmid;
		}
		
		/*
		 cout<<"iteration:  "<<nbIter<<'\n';
		 cout<<x[ind3]<<"   "<<y[ind3]<<'\n';
		 cout<<x[indmid]<<"   "<<y[indmid]<<'\n';
		 cout<<x[indfar]<<"   "<<y[indfar]<<'\n';
		 */
		
		denom=-x[0]*x[0]*x[2]+x[0]*x[0]*x[1]-x[0]*x[1]*x[1]+x[0]*x[2]*x[2]-x[2]*x[2]*x[1]+x[2]*x[1]*x[1];
		
		if (fabs(denom)>0.0)
		{
			a=(x[0]*y[2]-x[0]*y[1]+x[2]*y[1]-y[2]*x[1]-y[0]*x[2]+y[0]*x[1])/denom;
			b=-(-y[2]*x[1]*x[1]+x[1]*x[1]*y[0]-x[2]*x[2]*y[0]-y[1]*x[0]*x[0]+y[2]*x[0]*x[0]+y[1]*x[2]*x[2])/denom;
			c=(y[2]*x[0]*x[0]*x[1]-y[1]*x[0]*x[0]*x[2]-x[1]*x[1]*x[0]*y[2]+x[2]*x[2]*x[0]*y[1]+x[1]*x[1]*y[0]*x[2]-x[2]*x[2]*y[0]*x[1])/denom;
		}
		
		// if the root is real
		if ( (b*b-4.0*a*c)>0.0 && fabs(denom)>0.0)
		{
			xtmpp=(-b+sqrt(b*b-4.0*a*c))/(2*a);
			xtmpm=(-b-sqrt(b*b-4.0*a*c))/(2*a);
			
			if (xtmpp>x[indm] && xtmpp<x[indp])
			{
				x[indfar]=xtmpp;
			}
			else if (xtmpm>x[indm] && xtmpm<x[indp])
			{
				x[indfar]=xtmpm;
			}
			else if (fabs(y[ind3]-y[indmid])>tol)
			{
				x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
			}
			else
			{
				root[0]=x[indmid];
				return true;
			}
		}
		else  //linearly interpolate
		{
			if (fabs(y[ind3]-y[indmid])>tol)
			{
				x[indfar]=-y[indmid]*(x[ind3]-x[indmid])/(y[ind3]-y[indmid])+x[indmid];
			}
			else
			{
				root[0]=x[indmid];
				return true;
			}
			
		}
		
		y[indfar]=(this->*func)(x[indfar],params);
		nbEvaln++;
		
		dyprec=fabs(dy);
		dy=y[indfar];
		
		
		if (fabs(dyprec-fabs(dy))<dytol*dyprec || dyprec<fabs(dy))
		{
			x[indfar]=(x[ind3]+x[indmid])/2.0;
			y[indfar]=(this->*func)(x[indfar],params);
			nbEvaln++;
			dyprec=fabs(dy);
			dy=y[indfar];
		}
		
		//cout<<"iteration: "<<nbIter<<"  x: "<<x[indfar]<<"  y: "<<y[indfar]<<'\n';
		
		nbIter++;
		
	}
	
	//	cout<<"iteration "<<nbIter-1<<" de l'interpolation quadratique:  x: "<<x[indfar]<<"  y: "<<y[indfar]<<'\n';
	
	if (nbIter>=nbItermax)
	{
		//		cout<<"find_zero: nombre d'iterations maximal atteint"<<'\n';
		if (fabs(y[0])>tol2 && fabs(y[1])>tol2 && fabs(y[2])>tol2)
			return false;
	}
	
	//	root[0]=x[indfar];
	
	ind1=0;
	root[0]=x[ind1];
	if (fabs(y[1])<fabs(y[ind1]))
	{
		root[0]=x[1];
		ind1=1;
	}
	if (fabs(y[2])<fabs(y[ind1])) root[0]=x[2];
	
	
	return true;
}

bool generique::find_min_golden(fctPtr func, double lims[], double params[], double min[], double tol)
{
	double eps=2*EPSILON;
	int nbItermax=1000;
	int nbIter=0;
	
	double phi=(1.0+sqrt(5))/2;
	
	double x1=lims[0], x2=lims[1];
	double x3=x2-(x2-x1)/phi;
	double x4=x1+(x2-x1)/phi;
	double y1=(this->*func)(x1,params);
	double y2=(this->*func)(x2,params);
	double y3=(this->*func)(x3,params);
	double y4=(this->*func)(x4,params);
	
	//	cout<<setw(20)<<x1<<setw(20)<<y1<<setw(20)<<x<<setw(20)<<y<<setw(20)<<x2<<setw(20)<<y2<<endl;
	
	if (lims[1]<lims[0])
	{
		cout<<"find_min_golden(): lims[1]<lims[0]\n";
		return false;
	}
/*
	if (y3>y1 || y3>y2)
	{
		cout<<"find_min_golden(): the minimum is not in the provided interval\n";
		return false;
	}
*/	
	while ( fabs(y3-y4)>tol && nbIter<nbItermax && fabs(x3-x4)>eps)
	{
		if (y3<y4)
		{
			x2=x4;
			y2=y4;
			x4=x3;
			y4=y3;
			x3=x2-(x2-x1)/phi;
			y3=(this->*func)(x3,params);
		}
		else
		{
			x1=x3;
			y1=y3;
			x3=x4;
			y3=y4;
			x4=x1+(x2-x1)/phi;
			y4=(this->*func)(x4,params);
		}
		
		nbIter++;
	}
	
	if (nbIter>nbItermax && fabs(y3-y4)>tol)
	{
		cout<<"find_min_golden(): maximum number of iterations reached\n";
		cout<<"smallest difference found: "<<fabs(y3-y4)<<endl;
		return false;
	}
	
	if (y3<y4)
		min[0]=x3;
	else
		min[0]=x4;
	
	return true;
}

// find the minimum of a function
bool generique::find_min(fctPtr func, double init[], double params[], double xmin[])
{
	double tol=1.0e-12;
	int nbItermax=50;
	int nbIter=0;
	
	double x[3], y[3];
	double ymin, ymax,  xmax, xtmp, xtmp2, ytmp, denom, a, b, dxmax;
	int j, indmin, indmax;
	
	x[0]=init[0];
	x[1]=init[1];
	x[2]=0.5*(x[0]+x[1]);
	
	
	cout<<setiosflags(ios::left);
	y[0]=(this->*func)(x[0],params);
	y[0]=y[0]*y[0];
	//	cout<<setw(30)<<x[0]<<y[0]<<'\n';
	y[1]=(this->*func)(x[1],params);
	y[1]=y[1]*y[1];
	//	cout<<setw(30)<<x[1]<<y[1]<<'\n';
	y[2]=(this->*func)(x[2],params);
	y[2]=y[2]*y[2];
	//	cout<<setw(30)<<x[2]<<y[2]<<'\n';
	nbIter++;
	
	
	ymin=y[2];
	if (y[1]<ymin || y[0]<ymin)
	{
		cout<<"le minimum n'est pas dans l'intervalle donne\n";
		return false;
	}
	
	xtmp=x[2];
	xtmp2=xtmp+2*tol;
	
	while ( fabs(xtmp2-xtmp)>tol && nbIter<nbItermax )
	{
		xtmp2=xtmp;
		
		denom=-x[0]*x[0]*x[2]+x[0]*x[0]*x[1]-x[0]*x[1]*x[1]+x[0]*x[2]*x[2]-x[2]*x[2]*x[1]+x[2]*x[1]*x[1];
		
		if (fabs(denom)>0.0)
		{
			a=(x[0]*y[2]-x[0]*y[1]+x[2]*y[1]-y[2]*x[1]-y[0]*x[2]+y[0]*x[1])/denom;
			b=-(-y[2]*x[1]*x[1]+x[1]*x[1]*y[0]-x[2]*x[2]*y[0]-y[1]*x[0]*x[0]+y[2]*x[0]*x[0]+y[1]*x[2]*x[2])/denom;
			xtmp=-b/(2.0*a);
			ymin=(this->*func)(xtmp,params);
			ymin=ymin*ymin;
			ymax=ymin;
			indmax=-1;
			for (j=0; j<3; j++)
				if (y[j]>ymax)
				{
					indmax=j;
					ymax=y[j];
				}
			if (indmax>=0)
			{
				x[indmax]=xtmp;
				y[indmax]=ymin;
			}
			else
			{
				//				cout<<"find_min(): nouvelle valeur plus elevee\n";
				ytmp=y[0];
				indmin=0;
				if (y[1]<ytmp) indmin=1, ytmp=y[1];
				if (y[2]<ytmp) indmin=2;
				xmax=x[0];
				indmax=0;
				dxmax=fabs(xmax-x[indmin]);
				if (fabs(x[1]-x[indmin])>dxmax) dxmax=fabs(x[1]-x[indmin]), indmax=1;
				if (fabs(x[2]-x[indmin])>dxmax) indmax=2;
				x[indmax]=xtmp;
				y[indmax]=ymin;
			}
		}
		else
		{
			//			cout<<"find_min(): denominateur nul\n";
			//			cout<<setw(20)<<x[0]<<setw(20)<<x[1]<<x[2]<<'\n';
			xmin[0]=xtmp;
			ytmp=ymin;
			indmin=-1;
			for (j=0; j<3; j++)
				if (y[j]<ytmp)
				{
					indmin=j;
					ytmp=y[j];
				}
			if (indmin>=0) xmin[0]=x[indmin];
			return true;
		}
		nbIter++;
	}
	
	xmin[0]=xtmp;
	if (nbIter==nbItermax)
	{
		cout<<"find_min(): nombre d'iterations maximal atteint\n";
		return false;
	}
	
	return true;
}

//definitions of integration routines for real functions
double generique::quadInteg2D(fctPtr func,	double limx[2], double limy[2], double tol, int nbEval[], double params[])
{
	nbEval[0]=0;
	
	IntPtr inInteg=&generique::innerInteg;
	
	return quadInteg(func, limy, tol, nbEval, params, inInteg, limx);
}


double generique::innerInteg(double y, double tol, int nbEval[], double params[], fctPtr func, double limx[2])
{
	params[0]=y;
	
	return quadInteg(func, limx, tol, nbEval, params, NULL, NULL);
}


double generique::quadInteg(fctPtr func, double lims[2], double tol, int nbEval[], double params[], IntPtr inInteg, double limx[2])
{
	double a=lims[0], b=lims[1];
	
//	double h = 0.13579*(b-a);
	double hmin=1e-19*fabs(b-a);
	
	const unsigned int nbPoints0=3;
	
	double x[] = {a, (a+b)/2.0, b};
	
	double y[nbPoints0];
	
	if (inInteg)
		for (int l=0;l<nbPoints0; l++) y[l]=(this->*inInteg)(x[l], tol, nbEval, params, func, limx);
	else
	{
		for (int l=0;l<nbPoints0; l++) y[l]=(this->*func)(x[l],params);
		nbEval[0] += nbPoints0;
	}
	
	double sum=0.0;
	
	double stepLims[2];
	double stepVals[3];
	
	stepLims[0]=x[0];
	stepLims[1]=x[2];
	
	stepVals[0]=y[0];
	stepVals[1]=y[1];
	stepVals[2]=y[2];
	
	if (inInteg)  sum = quadStep(func, stepLims, stepVals, tol, nbEval, hmin, params, inInteg, limx);
	else sum = quadStep(func, stepLims, stepVals, tol, nbEval, hmin, params, NULL, NULL);
	
	return sum;
}

double generique::quadInteg1D(fctPtr1 func, double lims[2], double tol, int nbEval[], void *params[])
{
	double a=lims[0], b=lims[1];
	
//	double h = 0.13579*(b-a);
	double hmin=1e-19*fabs(b-a);
	
	const unsigned int nbPoints0=3;
	
	double x[] = {a, (a+b)/2.0, b};
	
	double y[nbPoints0];
	
	
	for (int l=0;l<nbPoints0; l++) y[l]=(this->*func)(x[l],params);
	nbEval[0] += nbPoints0;
	
	
	double sum=0.0;
	
	double stepLims[2];
	double stepVals[3];
	
	stepLims[0]=x[0];
	stepLims[1]=x[2];
	
	stepVals[0]=y[0];
	stepVals[1]=y[1];
	stepVals[2]=y[2];
	
	sum = quadStep1D(func, stepLims, stepVals, tol, nbEval, hmin, params);
	
	return sum;
}

double generique::quadStep1D(fctPtr1 func,  double lims[2], double vals[3], double tol, int nbEval[], double hmin, void *params[])
{
	int nbEvalmax=100000000;
	double a=lims[0], b=lims[1];
	
	double sum1=0.0, sum2=0.0, sum=0.0;
	double fa=vals[0], fc=vals[1], fb=vals[2], fd, fe;
	
	double h=b-a, c=(a+b)/2.0;
	
	if (fabs(h) < hmin || c == a || c == b)
	{
		sum = h*fc;
//		cout<<"quadStep1D(): smallest interval reached!\n";
//		cout<<"a: "<<a<<"   c: "<<c<<"   b: "<<b<<'\n';
//		cout<<"fa: "<<fa<<"   fc: "<<fc<<"   fb: "<<fb<<'\n';
		return sum;
	}
	
	double d=(a + c)/2.0, e=(c + b)/2.0;
	
	
	fd=(this->*func)(d,params);
	fe=(this->*func)(e,params);
	nbEval[0] += 2;
	
	
	if (nbEval[0] > nbEvalmax)
	{
		cout<<"quadStep1D : maximum number of function calls exceeded!\n";
		sum = h*fc;
		//cout<<"a: "<<a<<"   c: "<<c<<"   b: "<<b<<"  ky: "<<params[0]<<'\n';
		return sum;
	}
	
	
	//Three point Simpson's rule.
	sum1 = (h/6.0)*(fa + 4.0*fc + fb);
	
	//Five point double Simpson's rule.
	sum2 = (h/12.0)*(fa + 4.0*fd + 2.0*fc + 4.0*fe + fb);
	
	// One step of Romberg extrapolation (or Richardson extrapolation).
	sum = sum2 + (sum2 - sum1)/15.0;
	
	// Check accuracy of integral over this subinterval.
	if (fabs(sum2 - sum)  <= tol)
		return sum;
	else // Subdivide into two subintervals.
	{
		double stepLims[2];
		double stepVals[3];
		
		stepLims[0]=a;
		stepLims[1]=c;
		
		stepVals[0]=fa;
		stepVals[1]=fd;
		stepVals[2]=fc;
		
		sum = quadStep1D(func, stepLims, stepVals, tol, nbEval, hmin, params);
		
		stepLims[0]=c;
		stepLims[1]=b;
		
		stepVals[0]=fc;
		stepVals[1]=fe;
		stepVals[2]=fb;
		
		sum= sum + quadStep1D(func, stepLims, stepVals, tol, nbEval, hmin, params);
	}
	
	return sum;
}

double generique::quadStep(fctPtr func,  double lims[2], double vals[3], double tol, int nbEval[], double hmin, double params[],
						IntPtr inInteg,  double limx[2])
{
	int nbEvalmax=100000000;
	double a=lims[0], b=lims[1];
	
	double sum1=0.0, sum2=0.0, sum=0.0;
	double fa=vals[0], fc=vals[1], fb=vals[2], fd, fe;
	
	double h=b-a, c=(a+b)/2.0;
	
	if (fabs(h) < hmin || c == a || c == b)
	{
		sum = h*fc;
		cout<<"quadStep(): smallest interval reached!\n";
//		cout<<"a: "<<a<<"   c: "<<c<<"   b: "<<b<<"  ky: "<<params[0]<<'\n';
		return sum;
	}
	
	double d=(a + c)/2.0, e=(c + b)/2.0;
	
	if (inInteg)
	{
		fd=(this->*inInteg)(d, tol, nbEval, params, func, limx);
		fe=(this->*inInteg)(e, tol, nbEval, params, func, limx);
	}
	else
	{
		fd=(this->*func)(d,params);
		fe=(this->*func)(e,params);
		nbEval[0] += 2;
	}
	
	if (nbEval[0] > nbEvalmax)
	{
		cout<<"quadstep : maximum number of function calls exceeded!\n";
		sum = h*fc;
		//cout<<"a: "<<a<<"   c: "<<c<<"   b: "<<b<<"  ky: "<<params[0]<<'\n';
		return sum;
	}
	
	
	//Three point Simpson's rule.
	sum1 = (h/6.0)*(fa + 4.0*fc + fb);
	
	//Five point double Simpson's rule.
	sum2 = (h/12.0)*(fa + 4.0*fd + 2.0*fc + 4.0*fe + fb);
	
	// One step of Romberg extrapolation (or Richardson extrapolation).
	sum = sum2 + (sum2 - sum1)/15.0;
	
	// Check accuracy of integral over this subinterval.
	if (fabs(sum2 - sum)  <= tol)
		return sum;
	else // Subdivide into two subintervals.
	{
		double stepLims[2];
		double stepVals[3];
		
		stepLims[0]=a;
		stepLims[1]=c;
		
		stepVals[0]=fa;
		stepVals[1]=fd;
		stepVals[2]=fc;
		
		if (inInteg)
			sum = quadStep(func, stepLims, stepVals, tol, nbEval, hmin, params, inInteg, limx);
		else
			sum = quadStep(func, stepLims, stepVals, tol, nbEval, hmin, params, NULL, NULL);
		
		stepLims[0]=c;
		stepLims[1]=b;
		
		stepVals[0]=fc;
		stepVals[1]=fe;
		stepVals[2]=fb;
		
		if (inInteg)
			sum= sum + quadStep(func, stepLims, stepVals, tol, nbEval, hmin, params, inInteg, limx);
		else
			sum= sum + quadStep(func, stepLims, stepVals, tol, nbEval, hmin, params, NULL, NULL);
	}
	
	return sum;
}



//! calculate the coefficients in the continued fraction representation of the Pade aproximant,
// func: vector of values of the function, x: vector of the input points, N: size of 'func' and 'x',
// coefficients are returned in 'coef'
inline void generique::pade_cont_frac_coef(dcomplex *func, dcomplex *x, int N, dcomplex *coef)
{
	int l,m;
	
	dcomplex *gij=new dcomplex[(N*(N+1))/2];
	
	for (l=0; l<N; l++)
		gij[(l*(l+1))/2]=func[l];
	
	for (m=1; m<N; m++)
	{
		for (l=m; l<N; l++)
			gij[m+(l*(l+1))/2]=(gij[m-1+((m-1)*m)/2]-gij[m-1+(l*(l+1))/2])/( (x[l]-x[m-1])*gij[m-1+(l*(l+1))/2] );
	}
	
	for (l=0; l<N; l++)
	{
		coef[l]=gij[l+(l*(l+1))/2];
	}
	
	delete [] gij;
}

// pade approximant for an input value x after coef is calculated with pade_cont_frac_coef (or pade_cont_frac_coef_rec),
// n: number of input points in the Pade approximant, x0: vector of the input points
dcomplex generique::pade(dcomplex x, int n, dcomplex *x0, dcomplex *coef)
{
	dcomplex A, B;
	dcomplex q;
	
	int j;
	
	A=1.0;
	B=(double) 1.0 + (x-x0[0])*coef[1];
	q=A/B;
	for (j=3; j<=n; j++)
	{
		A=(double) 1.0 + (x-x0[j-2])*coef[j-1]/A;
		B=(double) 1.0 + (x-x0[j-2])*coef[j-1]/B;
		q=q*A/B;
	}
	q=q*coef[0];
	
	return q;
}

//! recursive formula in the Pade coefficients calculation
dcomplex generique::pade_recursion(int indFunc, int indx, dcomplex *func, dcomplex *x, dcomplex *coef)
{
	if (indFunc>0)
	{
		dcomplex f0=coef[indFunc-1];
		dcomplex f1=pade_recursion(indFunc-1, indx, func, x, coef);
		
		return (f0-f1)/( (x[indx]-x[indFunc-1])*f1 );
	}
	else
		return func[indx];
}

//! find the roots of the polynomial Ax^2+Bx+C with high precision
void generique::find_roots_2pol(dcomplex coef[], dcomplex roots[])
{
	double eps=1.0e-15;
	
	dcomplex A=coef[0], B=coef[1], C=coef[2];
	dcomplex R;
	
	if ( fabs(A)>eps )
	{
		R=sqrt(B*B-((double)4.0)*A*C);
		
		roots[0]=(-B - R)/(((double)2.0)*A);
		
		roots[1]=(-B + R)/(((double)2.0)*A);
	}
	else
	{
		cout<<"find_roots_2pol(): attention, coefficient de x^2 presque nul\n";
		return;
	}
	
	int j, nbIterMax=10, iter;
	dcomplex x, y, x2, y2, y0, p, c;
	double tol=1.0e-16;
	
	for (j=0; j<2; j++)
	{
		x=roots[j];
		y=A*x*x+B*x+C;
		p=((double)2.0)*A*x+B;
		y0=y;
		
		//		cout<<"|p(x_"<<j+1<<")| avant Newton-Raphson:  "<<fabs(y)<<'\n';
		
		iter=0;
		while (fabs(p) && iter<nbIterMax && fabs(y)>tol)
		{
			x2=x;
			y2=y;
			c=y-p*x;
			x=-c/p;
			y=A*x*x+B*x+C;
			p=((double)2.0)*A*x+B;
			
			if (fabs(y2)<fabs(y))
			{
				x=((double)0.5)*(x+x2);
				y=A*x*x+B*x+C;
				p=((double)2.0)*A*x+B;
			}
			//cout<<iter<<"	"<<x<<"	"<<fabs(y)<<'\n';
			
			iter++;
		}
		if (fabs(y)<fabs(y0))	
		{
			cout<<"avant:  x: "<<roots[j]<<"   y:  "<<fabs(y0)<<"  après:  x: "<<x<<"   y: "<<fabs(y)<<endl;
			roots[j]=x;
		}
	}
}

//! find the roots of degree 3 polynomial
void generique::find_roots_3pol(dcomplex coef[], dcomplex roots[])
{
	dcomplex I(0,1);
	
	double p3=1.0/3.0;
	double p2p3=pow(2.0,p3);
	double sq3=sqrt(3.0);
	
	dcomplex A=coef[0], B=coef[1], C=coef[2], D=coef[3];
	
	dcomplex u=-B*B + 3.0*A*C;
	dcomplex v=-2.0*B*B*B + 9.0*A*B*C - 27.0*A*A*D;
	dcomplex w;
	
	if (u!=0.0) w=pow( v + sqrt(4.0*u*u*u + v*v) ,p3);
	else w=pow( 2.0*v ,p3);
	
	//	cout<<"coef: "<<A<<"  "<<B<<"  "<<C<<"  "<<D<<'\n';
	//	cout<<"u, v, w: "<<u<<"  "<<v<<"  "<<w<<'\n';
	
	roots[0]=-B/(3.*A) - (p2p3*u)/(3.*A*w) + w/(3.*p2p3*A);
	roots[1]=-B/(3.*A) + ((1.0 + I*sq3)*u)/(3.*p2p3*p2p3*A*w) - ((1.0 - I*sq3)*w)/(6.*p2p3*A);
	roots[2]=-B/(3.*A) + ((1.0 - I*sq3)*u)/(3.*p2p3*p2p3*A*w) - ((1.0 + I*sq3)*w)/(6.*p2p3*A);
}

//! find the roots of the polynomial Ax^4+Bx^3+Cx^2+Dx+F
void generique::find_roots_4pol(dcomplex coef[], dcomplex roots[])
{
	dcomplex A=coef[0], B=coef[1], C=coef[2], D=coef[3], F=coef[4];
	
	dcomplex G, H, J, K, L, M, P, Q, S, U;
	
	double p3=1.0/3.0;
	
	double eps=1.0e-13;
	if ( fabs(B)>eps || fabs(D)>eps )
	{
		G=C*C - ((double)3.0)*B*D + ((double)12.0)*A*F;
		H=((double)2.0)*C*C*C - ((double)9.0)*B*C*D + ((double)27.0)*A*D*D + ((double)27.0)*B*B*F - ((double)72.0)*A*C*F;
		
		U=pow(H + sqrt(-((double)4.0)*G*G*G + H*H),p3);
		
		Q=(pow((double)2.0,p3)*G)/(((double)3.0)*A*U);
		S=U/(((double)3.0)*pow(2,p3)*A);
		
		J=sqrt(B*B/(((double)4.0)*A*A) - (((double)2.0)*C)/(((double)3.0)*A) + Q + S);
		
		M=B*B/(((double)2.0)*A*A) - (((double)4.0)*C)/(((double)3.0)*A) - Q - S;
		P=(-(B*B*B/(A*A*A)) + (((double)4.0)*B*C)/(A*A) - (((double)8.0)*D)/A)/(((double)4.0)*J);
		
		K=sqrt(M - P);
		L=sqrt(M + P);
		
		roots[0]=-B/(((double)4.0)*A) - K/((double)2.0) - J/((double)2.0);
		
		roots[1]=-B/(((double)4.0)*A) + K/((double)2.0) - J/((double)2.0);
		
		roots[2]=-B/(((double)4.0)*A) - L/((double)2.0) + J/((double)2.0);
		
		roots[3]=-B/(((double)4.0)*A) + L/((double)2.0) + J/((double)2.0);
	}
	else
	{
		//		cout<<"find_roots_4pol(): B et D sont nuls\n";
		roots[0]=sqrt((-C+sqrt(C*C-((double)4.0)*A*F))/(((double)2.0)*A));
		roots[1]=-roots[0];
		roots[2]=sqrt((-C-sqrt(C*C-((double)4.0)*A*F))/(((double)2.0)*A));
		roots[3]=-roots[2];
	}
	
	int j, nbIterMax=10, iter;
	dcomplex x, y, x2, y2, y0, p, c;
	double tol=1.0e-15;
	double diffrel;
	
	for (j=0; j<4; j++)
	{
		x=roots[j];
		y=A*x*x*x*x+B*x*x*x+C*x*x+D*x+F;
		p=((double)4.0)*A*x*x*x+((double)3.0)*B*x*x+((double)2.0)*C*x+D;
		y0=y;
		
		//		cout<<"|p(x_"<<j+1<<")| avant Newton-Raphson:  "<<fabs(y)<<'\n';
		
		iter=0;
		while (fabs(p) && iter<nbIterMax && fabs(y)>tol)
		{
			x2=x;
			y2=y;
			c=y-p*x;
			x=-c/p;
			y=A*x*x*x*x+B*x*x*x+C*x*x+D*x+F;
			p=((double)4.0)*A*x*x*x+((double)3.0)*B*x*x+((double)2.0)*C*x+D;
			
			if (fabs(y2)<fabs(y))
			{
				x=((double)0.5)*(x+x2);
				y=A*x*x*x*x+B*x*x*x+C*x*x+D*x+F;
				p=((double)4.0)*A*x*x*x+((double)3.0)*B*x*x+((double)2.0)*C*x+D;
			}
			//			cout<<iter<<"	"<<x<<"	"<<fabs(y)<<'\n';
			
			iter++;
		}
		if (fabs(y)<fabs(y0))	roots[j]=x;
		
		//		x=roots[j];
		//		y=A*x*x*x*x+B*x*x*x+C*x*x+D*x+F;
		//		cout<<"|p(x_"<<j+1<<")| apres Newton-Raphson:  "<<fabs(y)<<'\n';
	}
	
	int l;
	for (j=0; j<3; j++)
		for (l=j+1; l<4; l++)
		{
			diffrel=2*fabs(roots[j]-roots[l])/(fabs(roots[j])+fabs(roots[l]));
			if (diffrel<1.0e-6) cout<<"find_roots_4pol(): attention, racine double possible\n";
		}
}
