/*
 file OmegaMaxEnt_data.h
 definition of class "OmegaMaxEnt_data", the main class of the program OmegaMaxEnt that performs the analytic continuation of numerical Matsubara Green and correlation functions.
 (other source files: OmegaMaxEnt_data.cpp, OmegaMaxEnt_main.cpp, graph_2D.h, graph_2D.cpp, generique.h, generique.cpp)
 
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

#ifndef OMEGAMAXENT_DATA_H
#define OMEGAMAXENT_DATA_H

#include "includeDef.h"
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include "graph_2D.h"
#include "generique.h"
#include "OmegaMaxEnt_license.h"

using namespace arma;

//MAIN INPUT FILE PARAMETERS

static string default_input_params_file_name("OmegaMaxEnt_input_params.dat");
static string template_input_params_file_name("OmegaMaxEnt_input_params_template.dat");
static string other_params_file_name("OmegaMaxEnt_other_params.dat");

static string data_file_param("data file:");

enum Data_params_name {BOSON, TAU_GF, TEM, G_INF_FINITE, G_INF, NORM_A, M_1, ERR_M1, M_2, ERR_M2, M_3, ERR_M3, TRUNC_FREQ};

static map<Data_params_name, string> Data_params( {
	{BOSON,"bosonic data (yes/[no]):"},
	{TAU_GF,"imaginary time data (yes/[no]):"},
	{TEM,"temperature (in energy units, k_B=1):"},
	{G_INF_FINITE,"finite value at infinite frequency (yes/[no]):"},
	{G_INF,"value at infinite frequency:"},
	{NORM_A,"norm of spectral function:"},
	{M_1,"1st moment:"},
	{ERR_M1,"1st moment error:"},
	{M_2,"2nd moment:"},
	{ERR_M2,"2nd moment error:"},
	{M_3,"3rd moment:"},
	{ERR_M3,"3rd moment error:"},
	{TRUNC_FREQ,"truncation frequency:"}} );

enum Intput_files_params_name {INPUT_DIR, COL_GR, COL_GI, ERROR_FILE, COL_ERR_GR, COL_ERR_GI, COVAR_RE_RE_FILE, COVAR_IM_IM_FILE, COVAR_RE_IM_FILE, COL_G_TAU, COL_ERR_G_TAU, COVAR_TAU_FILE, NOISE_PARAMS};

static map<Intput_files_params_name, string> Input_files_params( {
	{INPUT_DIR, "input directory:"},
	{COL_GR, "Re(G) column in data file (default: 2):"},
	{COL_GI, "Im(G) column in data file (default: 3):"},
	{ERROR_FILE, "error file:"},
	{COL_ERR_GR, "Re(G) column in error file (default: 2):"},
	{COL_ERR_GI, "Im(G) column in error file (default: 3):"},
	{COVAR_RE_RE_FILE, "re-re covariance file:"},
	{COVAR_IM_IM_FILE, "im-im covariance file:"},
	{COVAR_RE_IM_FILE, "re-im covariance file:"},
	{COL_G_TAU, "column of G(tau) in data file (default: 2):"},
	{COL_ERR_G_TAU, "column of G(tau) error in error file (default: 2):"},
	{COVAR_TAU_FILE, "imaginary time covariance file:"},
	{NOISE_PARAMS, "added noise relative error (s1 s2 ...) (default: 0):"}} );


enum Grig_params_name {CUTOFF_WN, SPECTR_FUNC_WIDTH, SPECTR_FUNC_CENTER, W_ORIGIN, STEP_W, GRID_W_FILE, NON_UNIFORM_GRID, USE_GRID_PARAMS, PARAM_GRID_PARAMS,OUTPUT_GRID_PARAMS};

static map<Grig_params_name, string> Grid_params( {
	{CUTOFF_WN, "Matsubara frequency cutoff (in energy units, k_B=1):"},
	{SPECTR_FUNC_WIDTH, "spectral function width:"},
	{SPECTR_FUNC_CENTER, "spectral function center:"},
	{W_ORIGIN, "real frequency grid origin:"},
	{STEP_W, "real frequency step:"},
	{GRID_W_FILE, "real frequency grid file:"},
	{NON_UNIFORM_GRID,"use non uniform grid in main spectral range (yes/[no]):"},
	{USE_GRID_PARAMS, "use parameterized real frequency grid (yes/[no]):"},
	{PARAM_GRID_PARAMS, "grid parameters (w_0 dw_0 w_1 dw_1 ... w_{N-1} dw_{N-1} w_N):"},
	{OUTPUT_GRID_PARAMS,"output real frequency grid parameters (w_min dw w_max):"}} );


enum Preproc_comp_params_name {EVAL_MOMENTS, MAX_M, DEFAULT_MODEL_CENTER, DEFAULT_MODEL_WIDTH, DEFAULT_MODEL_SHAPE, DEFAULT_MODEL_FILE, INIT_SPECTR_FUNC_FILE,COMPUTE_PADE,N_PADE,ETA_PADE};
//, INTERP_TYPE

static map<Preproc_comp_params_name, string> Preproc_comp_params( {
	{EVAL_MOMENTS, "evaluate moments (yes/[no]):"},
	{MAX_M, "maximum moment:"},
	{DEFAULT_MODEL_CENTER, "default model center (default: 1st moment):"},
	{DEFAULT_MODEL_WIDTH, "default model half width (default: standard deviation):"},
	{DEFAULT_MODEL_SHAPE, "default model shape parameter (default: 2):"},
	{DEFAULT_MODEL_FILE, "default model file:"},
	{INIT_SPECTR_FUNC_FILE, "initial spectral function file:"},
	{COMPUTE_PADE,"compute Pade result (yes/[no]):"},
	{N_PADE,"number of frequencies for Pade:"},
	{ETA_PADE,"imaginary part of frequency in Pade:"}} );
//{INTERP_TYPE, "interpolation type (spline (default), quad, lin):"}

enum Preproc_exec_params_name {PREPROSSESS_ONLY, DISPL_PREP_FIGS, DISPL_ADV_PREP_FIGS, PRINT_OTHER_PARAMS};

static map<Preproc_exec_params_name, string> Preproc_exec_params( {
	{PREPROSSESS_ONLY, "preprocess only (yes/[no]):"},
	{DISPL_PREP_FIGS, "display preprocessing figures (yes/[no]):"},
	{DISPL_ADV_PREP_FIGS, "display advanced preprocessing figures (yes/[no]):"},
	{PRINT_OTHER_PARAMS, "print other parameters (yes/[no]):"} } );


enum Output_files_params_name {OUTPUT_DIR, OUTPUT_NAME_SUFFIX, ALPHA_SAVE_MAX, ALPHA_SAVE_MIN, W_SAMPLE};

static map<Output_files_params_name, string> Output_files_params( {
	{OUTPUT_DIR, "output directory:"},
	{OUTPUT_NAME_SUFFIX, "output file names suffix:"},
	{ALPHA_SAVE_MAX, "maximum alpha for which results are saved:"},
	{ALPHA_SAVE_MIN, "minimum alpha for which results are saved:"},
	{W_SAMPLE, "spectral function sample frequencies (w_1 w_2 ... w_N):"} } );


enum Optim_comp_params_name {ALPHA_INIT, ALPHA_MIN, ALPHA_OPT_MAX, ALPHA_OPT_MIN};

static map<Optim_comp_params_name, string> Optim_comp_params( {
	{ALPHA_INIT, "initial value of alpha:"},
	{ALPHA_MIN, "minimum value of alpha:"},
	{ALPHA_OPT_MAX, "maximum optimal alpha:"},
	{ALPHA_OPT_MIN, "minimum optimal alpha:"} } );

enum Optim_exec_params_name {N_ALPHA, INITIALIZE_MAXENT, INITIALIZE_PREPROC, INTERACTIVE_MODE};

static map<Optim_exec_params_name, string> Optim_exec_params( {
	{N_ALPHA, "number of values of alpha computed in one execution:"},
	{INITIALIZE_MAXENT, "initialize maxent (yes/[no]):"},
	{INITIALIZE_PREPROC, "initialize preprocessing (yes/[no]):"},
	{INTERACTIVE_MODE, "interactive mode ([yes]/no):"} } );


enum Optim_displ_params_name {PRINT_ALPHA, SHOW_OPTIMAL_ALPHA_FIGS, SHOW_LOWEST_ALPHA_FIGS, SHOW_ALPHA_CURVES, REF_SPECTR_FILE};

static map<Optim_displ_params_name, string> Optim_displ_params( {
	{PRINT_ALPHA, "print results at each value of alpha (yes/[no]):"},
	{SHOW_OPTIMAL_ALPHA_FIGS,"show optimal alpha figures ([yes]/no):"},
	{SHOW_LOWEST_ALPHA_FIGS,"show lowest alpha figures ([yes]/no):"},
	{SHOW_ALPHA_CURVES,"show alpha dependant curves ([yes]/no):"},
	{REF_SPECTR_FILE, "reference spectral function file:"} } );

// OTHER PARAMETERS

enum Other_params_int_name {NN_MIN, NN_MAX, NW_MIN, NW_MAX, NN_FIT_MAX, NN_FIT_FIN, NN_AS_MIN, N_ITER_DA_MAX, NW_SAMP, NSMOOTH_ERRG};

static map<Other_params_int_name, string> Other_params_int( {
	{NN_MIN,"Nn_min, minimum number of Matsubara frequencies:"},
	{NN_MAX,"Nn_max, maximum number of Matsubara frequencies:"},
    {NW_MIN,"Nw_min, minimum number of real frequencies:"},
    {NW_MAX,"Nw_max, maximum number of real frequencies:"},
    {NN_FIT_MAX,"Nn_fit_max, initial maximum number of frequencies used to fit the asymptotic form during computation of moments:"},
    {NN_FIT_FIN,"Nn_fit_fin, final maximum number of frequencies used to fit the asymptotic form during computation of moments:"},
	{NN_AS_MIN,"Nn_as_min, minimum number of frequencies in the asymptotic region:"},
    {N_ITER_DA_MAX,"Niter_dA_max, maximum number of iterations in Newton's method for a given value of alpha:"},
    {NW_SAMP,"Nwsamp, default number of sample frequencies of spectral function to be save as a function of alpha:"},
	{NSMOOTH_ERRG,"Nsmooth_errG, smoothing distance for the added noise error:"}} );

static map<Other_params_int_name, int> Other_params_int_default_values( {
	{NN_MIN,20},
	{NN_MAX,500},
    {NW_MIN,50},
    {NW_MAX,800},
    {NN_FIT_MAX,200},
    {NN_FIT_FIN,300},
	{NN_AS_MIN,10},
    {N_ITER_DA_MAX,20},
    {NW_SAMP,11},
	{NSMOOTH_ERRG,0} } );

enum Other_params_fl_name {F_SW_STD_OMEGA, F_W_RANGE, RMIN_SW_DW, TOL_TEM, TOL_GINF, TOL_NORM, TOL_M1, TOL_M2, TOL_M3, DEFAULT_ERROR_G, ERR_NORM, DEFAULT_ERROR_M, TOL_MEAN_C1,TOL_STD_C1, TOL_RDW, RMIN_DW_DW, RDW_MAX, RW_GRID, RWD_GRID,  MIN_DEF_M, F_ALPHA_INIT, R_WIDTH_ASMIN, F_SMIN, DIFF_CHI2_MAX, TOL_INT_DA, R_C2_H, POW_ALPHA_STEP_INIT, POW_ALPHA_STEP_MIN, CHI2_ALPHA_SMOOTH_RANGE_2, F_SCALE_LALPHA_LCHI2, FN_FIT_TAU_W, STD_NORM_PEAK_MAX, VAR_M2_PEAK_MAX, PEAK_WEIGHT_MIN, RMAX_DLCHI2_LALPHA, F_ALPHA_MIN, SAVE_ALPHA_RANGE, R_PEAK_WIDTH_DW, R_WNCUTOFF_WR, R_DW_DW, R_SW_WR, R_WMAX_WR_MIN,WGT_MIN_SM,R_SW_G_RE_W_RANGE,R_DW_MIN_DW_DENSE, R_WKK_SW, R_SV_MIN};

static map<Other_params_fl_name, string> Other_params_fl( {
	{F_SW_STD_OMEGA, "f_SW_std_omega, ratio of main spectral range and standard deviation of spectrum:"},
	{F_W_RANGE,"f_w_range, total real frequency range=f_w_range*(spectral function width):"},
	{RMIN_SW_DW, "Rmin_SW_dw, minimum ratio of standard deviation and frequency step:"},
    {TOL_TEM,"tol_tem, relative tolerance between temperature extracted from Matsubara frequency and input temperature:"},
	{TOL_GINF,"tol_Ginf, tolerance on frequency-independent part of data:"},
    {TOL_NORM,"tol_norm, tolerance on norm extracted from high frequency:"},
    {TOL_M1,"tol_M1, tolerance between 1st moment extracted from high frequency and input one:"},
    {TOL_M2,"tol_M2, tolerance between 2nd moment extracted from high frequency and input one:"},
    {TOL_M3,"tol_M3, tolerance between 3rd moment extracted from high frequency and input one:"},
    {DEFAULT_ERROR_G,"default_error_G, default error on the input data:"},
    {ERR_NORM,"err_norm, relative error on norm:"},
    {DEFAULT_ERROR_M,"default_error_M, default error on moments:"},
    {TOL_MEAN_C1,"tol_mean_C1, tolerance on mean(M0(n)):"},
    {TOL_STD_C1,"tol_std_C1, tolerance on std(M0(n)):"},
    {TOL_RDW,"tol_rdw, tolerance on ratio of consecutive frequency step:"},
    {RMIN_DW_DW,"Rmin_Dw_dw, minimum number of steps in a grid interval:"},
    {RDW_MAX,"Rdw_max, maximum ratio of steps in consecutive grid interval:"},
    {RW_GRID,"RW_grid, grid interval vs transition region ratio:"},
    {RWD_GRID,"RWD_grid, transition region vs width parameter ratio:"},
    {MIN_DEF_M,"minDefM, minimum value of default model:"},
    {F_ALPHA_INIT,"f_alpha_init, initial ratio of entropy and chi2 contributions to the spectrum:"},
    {R_WIDTH_ASMIN,"R_width_ASmin, width of the minimum entropy spectrum relative to spectral function width:"},
    {F_SMIN,"f_Smin, minimum entropy term versus optimal chi2 ratio:"},
    {DIFF_CHI2_MAX,"diff_chi2_max, maximum relative difference between the chi2 of consecutive values of alpha:"},
    {TOL_INT_DA,"tol_int_dA, tolerance on consecutive values of the integral of |dA| in Newton's method:"},
    {R_C2_H,"rc2H, maximum ratio of the penalization parameter and the maximum eigenvalue of the hessian of chi2:"},
    {POW_ALPHA_STEP_INIT,"pow_alpha_step_init, initial value of the step in log_10(alpha):"},
    {POW_ALPHA_STEP_MIN,"pow_alpha_step_min, minimum value of the step in log_10(alpha):"},
    {CHI2_ALPHA_SMOOTH_RANGE_2,"chi2_alpha_smooth_range, chi2 vs alpha smoothing range in log10 scale:"},
    {F_SCALE_LALPHA_LCHI2,"f_scale_lalpha_lchi2, log(alpha) scale factor with respect to log(chi2) in the curvature calculation:"},
    {FN_FIT_TAU_W, "FNfitTauW, factor to determine the number of values of tau in the polynomial fit:"},
    {STD_NORM_PEAK_MAX,"std_norm_peak_max, relative tolerance for standard deviation of low frequency peak weight:"},
    {VAR_M2_PEAK_MAX,"varM2_peak_max, relative tolerance on low frequency peak variance:"},
	{PEAK_WEIGHT_MIN, "peak_weight_min, minimum value of peak weight to assume a low energy peak is present:"},
	{RMAX_DLCHI2_LALPHA,"RMAX_dlchi2_lalpha, maximum ratio of dlog(chi2)/dlog(alpha) at the lowest alpha and the maximum value:"},
	{F_ALPHA_MIN,"f_alpha_min, factor by which alpha_min is reduced when found to be too high:"},
	{SAVE_ALPHA_RANGE,"save_alpha_range, range of alpha to be saved around the optimal alpha in log10 scale:"},
	{R_PEAK_WIDTH_DW, "R_peak_width_dw, ratio of low energy peak width and low frequency step:"},
	{R_WNCUTOFF_WR, "R_wncutoff_wr, ratio of onset Matsubara frequency of asymptotic region and main spectral range maximum frequency:"},
	{R_DW_DW, "R_Dw_dw, ratio of grid interval length and step:"},
	{R_SW_WR, "R_SW_wr, ratio of main spectral range maximum frequency and spectrum standard deviation:"},
	{R_WMAX_WR_MIN, "R_wmax_wr_min, minimum ratio of grid maximum frequency and main spectral range maximum frequency:"},
	{WGT_MIN_SM, "wgt_min_sm, smallest relative weight in the smoothing of the added noise error:"},
	{R_SW_G_RE_W_RANGE, "R_SW_G_Re_w_range, ratio of total frequency range and main spectral region for the real part of G:"},
	{R_DW_MIN_DW_DENSE,"R_dw_min_dw_dense, default ratio of the minimal step in the computation grid and the step in the output grid:"},
	{R_WKK_SW,"R_wKK_SW, frequency region around zero where Re[G] is computed with Kramers-Kronig, divided by the spectral function width:"},
	{R_SV_MIN,"minimum ratio of matrix singular values in the moments computation in tau:"}} );

static map<Other_params_fl_name, double> Other_params_fl_default_values( {
	{F_SW_STD_OMEGA,3},
	{F_W_RANGE,20.0},
	{RMIN_SW_DW,50},
    {TOL_TEM,1.0e-8},
	{TOL_GINF,1.0e-3},
    {TOL_NORM,0.1},
    {TOL_M1,0.05},
    {TOL_M2,0.05},
    {TOL_M3,0.1},
    {DEFAULT_ERROR_G,1.0e-4},
    {ERR_NORM,1.0e-6},
    {DEFAULT_ERROR_M,1.0e-4},
    {TOL_MEAN_C1,0.002},
    {TOL_STD_C1,0.002},
    {TOL_RDW,1.0e-10},
    {RMIN_DW_DW,4.0},
    {RDW_MAX,5.0},
    {RW_GRID,3.0},
    {RWD_GRID,10.0},
    {MIN_DEF_M,1.0e-20},
    {F_ALPHA_INIT,1.0e3},
    {R_WIDTH_ASMIN,0.05},
    {F_SMIN,1.0},
    {DIFF_CHI2_MAX,0.2},
    {TOL_INT_DA,1.0e-12},
    {R_C2_H,1.0e12},
    {POW_ALPHA_STEP_INIT,0.2},
    {POW_ALPHA_STEP_MIN,0.001},
    {CHI2_ALPHA_SMOOTH_RANGE_2,0.2},
    {F_SCALE_LALPHA_LCHI2,0.2},
    {FN_FIT_TAU_W,4.0},
    {STD_NORM_PEAK_MAX,0.02},
    {VAR_M2_PEAK_MAX,0.02},
	{PEAK_WEIGHT_MIN,1.0e-4},
	{RMAX_DLCHI2_LALPHA,0.01},
	{F_ALPHA_MIN,100},
	{SAVE_ALPHA_RANGE,0},
	{R_PEAK_WIDTH_DW,10},
	{R_WNCUTOFF_WR,10},
	{R_DW_DW,30},
	{R_SW_WR,1},
	{R_WMAX_WR_MIN,3},
	{WGT_MIN_SM,0.2},
	{R_SW_G_RE_W_RANGE,10},
	{R_DW_MIN_DW_DENSE,5},
	{R_WKK_SW,0.01},
	{R_SV_MIN,1e-12}} );

static const char *OmegaMaxEnt_notice=R"(
OmegaMaxEnt Copyright (C) 2015 Dominic Bergeron (dominic.bergeron@usherbrooke.ca)
This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you
are welcome to redistribute it under certain conditions; For details refer to
the GNU General Public License, of which you should have received a copy
along with the program, or see <http://www.gnu.org/licenses/>.
)";

extern "C++"
{
	class OmegaMaxEnt_data: public generique
    {
    public:
        OmegaMaxEnt_data(int arg_N, char *args[]);
        ~OmegaMaxEnt_data();
		
		//main function for this class. Only function called from outside the class.
        void loop_run();

		void display_license();
		void display_notice();
        
    private:
		
		// load input parameters, like data file name, covariance file, default model, etc, from file OmegaMaxEnt_input_params.dat
        bool load_input_params();
		// load internal computation parameters from file OmegaMaxEnt_other_params.dat, which contains the default values at initialization
        bool load_other_params();
		// load data files. Used to load all arrays.
		bool load_data_file(mat &data_array, string file_name);
		// main preprocessing routine
		bool preproc();
		// define a fermionic Green function
		bool set_G_omega_n_fermions();
		// define a bosonic Green function
		bool set_G_omega_n_bosons();
		// define the Matsubara frequency covariance matrices for general Green function
		bool set_covar_G_omega_n();
		// define the Matsubara frequency covariance matrices for even bosonic data
		bool set_covar_chi_omega_n();
		// define the imaginary time covariance matrix
		bool set_covar_Gtau();
		// set the moments in the fermionic case. 
		bool set_moments_fermions();
		// set the moments in the bosonic case
		bool set_moments_bosons();
		// extract the moments from the Matsubara frequency Green function. Associated with parameter "evaluate moments" in section COMPUTATION OPTIONS of file OmegaMaxEnt_input_params.dat, but also called automatically if not enough information is provided to define the real frequency grid.
		bool compute_moments_omega_n();
		// extract the moments from a Matsubara frequency Green function which has a frequency-independant part
		bool compute_moments_omega_n_2();
		// extract moments from a real (even) Matsubara frequency correlation function
		bool compute_moments_chi_omega_n();
		//compute derivatives of G(tau) at tau=0 and tau=beta
		bool compute_dG_dtau();
		// extract moments from an imaginary time fermionic Green function
		bool compute_moments_tau_fermions();
		// extract moments from an imaginary time bosonic Green function
		bool compute_moments_tau_bosons();
		// test the presence of a sharp peak around omega=0, using a Laurent series fit, for a fermionic Matsubara frequency Green function
		bool test_low_energy_peak_fermions();
		// test the presence of a sharp peak around omega=0, using a Laurent series fit, for a bosonic Matsubara frequency Green function
		bool test_low_energy_peak_bosons();
		// test the presence of a sharp peak around omega=0, using a Laurent series fit, for a real bosonic Matsubara frequency Green function
		bool test_low_energy_peak_chi();
		// set the initial spectrum if it does not correspond to the default model. Useful if a calculation is interrupted, to restart it at the last value of alpha computed. Associated with parameter "initial spectral function file" in section COMPUTATION OPTIONS of file OmegaMaxEnt_input_params.dat
		bool set_initial_spectrum();
		// set the initial spectrum if it does not correspond to the default model. Even correlation function case.
		bool set_initial_spectrum_chi();
		// use a real frequency grid provided by the user in the main spectral region
		bool set_grid_omega_from_file();
		// use a real frequency grid provided by the user in the main spectral region. Even correlation function case.
		bool set_grid_omega_from_file_chi();
		// define the grid from parameters "grid parameters", if "use parameterized real frequency grid" is enabled, in section FREQUENCY GRID PARAMETERS of file OmegaMaxEnt_input_params.dat
		bool set_grid_from_params();
		// define the grid from parameters "grid parameters", if "use parameterized real frequency grid" is enabled. Even correlation function case.
		bool set_grid_from_params_chi();
		// define the real frequency grid in the main spectral region.
		bool set_wc();
		// define the real frequency grid in the main spectral region. Even correlation function case.
		bool set_wc_chi();
		// create the complete real frequency grid, including high frequency parts.
		bool set_omega_grid();
		// create the complete real frequency grid, including high frequency parts. Even correlation function case.
		bool set_omega_grid_chi();
		// define a reference spectrum that will be plotted along with the final result. Associated with parameter "reference spectral function file" in section DISPLAY OPTIONS of file OmegaMaxEnt_input_params.dat
		bool set_A_ref();
		// truncate the data at frequency "truncation frequency" in section DATA PARAMETERS of file OmegaMaxEnt_input_params.dat. Useful to reduce preprocessing time if there are much more frequencies in the data than needed.
		bool truncate_G_omega_n();
		// truncate the data at frequency "truncation frequency" in section DATA PARAMETERS of file OmegaMaxEnt_input_params.dat. Even correlation function case.
		bool truncate_chi_omega_n();
		// define the default model. Associated with the default model parameters in section COMPUTATION OPTIONS of file OmegaMaxEnt_input_params.dat.
		bool set_default_model();
		// define the default model. Even correlation function case.
		bool set_default_model_chi();
		// compute values of the default model by interpolation or using gaussian tails for a user-defined default model.
		bool default_model_val_G(vec x, vec x0, vec coeffs, vec gaussians_params, vec &dm);
		// compute values of the default model by interpolation or using gaussian tails for a user-defined default model. Even correlation function case.
		bool default_model_val_chi(vec x, vec x0, vec coeffs, vec gaussians_params, vec &dm);
		// compute the discrete version of the kernel in the spectral representation for fermions, using a cubic spline model for the spectrum. In that version, a piecewise linear transformation is applied to the grids in the low and high frequency regions to make them uniform and improve the conditioning of the matrix to be inverted. See appendix E in https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.023303 for some details. The calculation with the grid transformation is unpublished however.
		bool Kernel_G_fermions_grid_transf();
		// version of Kernel_G_fermions_grid_transf() for the case of a uniform grid in the main spectral region. Not currently used.
		bool Kernel_G_fermions_grid_transf_omega();
		// another unused version of Kernel_G_fermions_grid_transf().
		bool Kernel_G_fermions_grid_transf_2();
		// old version of the kernel calculation without the grid transformation.
		bool Kernel_G_fermions();
		// simplest possible kernel definition using a Riemann integral
		bool Kernel_G_fermions_Riemann_integ();
		// compute the discrete version of the kernel in the spectral representation for a general bosonic Green function. The grid transformation has not been implemented for that case yet.
		bool Kernel_G_bosons();
		// compute the discrete version of the kernel in the spectral representation for the even correlation function case. The grid transformation has not been implemented for that case yet.
		bool Kernel_chi();
		// diagonalize the covariance matrix.
		bool diagonalize_covariance();
		// diagonalize the covariance matrix. Even correlation function case.
		bool diagonalize_covariance_chi();
		// compute the Fourier transform of imaginary time data.
		bool Fourier_transform_G_tau();
		// generate the non-uniform frequency grid in the main spectral region. Called by set_grid_from_params() and set_grid_from_params_chi().
		bool non_uniform_frequency_grid(rowvec w_steps, rowvec wlims, double w0, vec R, vec &grid);
		
		// define a function as a sum of gaussians
		bool sum_gaussians(vec x, rowvec x0, rowvec s0, rowvec wgt, vec &F);
		// define an even function as a sum of gaussians
		bool sum_gaussians_chi(vec x, rowvec x0, rowvec s0, rowvec wgt, vec &F);
		// compute the values of a general normal distribution at vector x
		bool general_normal(vec x, double x0, double s0, double p, vec &F);
		// compute a single value of a general normal distribution
		double general_normal_val(double x, void *par[]);
		
		// compute the spline coefficients for a function V defined at points x0
		void spline_coeffs(double *x0, double *V, int N0, double *coeffs);
		//
		void spline_matrix(double *x0, int N0, mat &MS);
		// obtain a value of the spline computed with spline_coeffs()
		bool spline_val(vec x, vec x0, vec coeffs, vec &s);
		// compute the coefficients of a hybrid spline cubic in frequency at low frequency and cubic in u=1/(w-w_{0,s}) at high frequency (w_{0,s} defines the cutoff, s=left/right). Used by set_default_model() if a default model previously created by this software is reused by the user.
		bool spline_G_part(vec x, uvec ind_xlims, vec xs, vec F, vec &coeffs);
		// compute values of the spline defined by spline_G_part. Used by set_default_model().
		bool spline_val_G_part(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv);
		// compute a single value of the spline defined by spline_G_part.
		double spline_val_G_part(double x, vec &x0, uvec &ind_xlims, vec &xs, vec &coeffs);
		// compute values of the spline defined by spline_G_part. Used during tests only.
		bool spline_G_omega_u(vec x, uvec ind_xlims, vec xs, vec F, vec &coeffs);
		// define the hybrid spline matrix. Used by kernel_G_fermions() and kernel_G_bosons().
		bool spline_matrix_G_part(vec x, uvec ind_xlims, vec xs, mat &M);
		// define the hybrid spline matrix. Used by Kernel_G_fermions_grid_transf_omega().
		bool spline_matrix_grid_transf(vec w0, mat &M);
		// define the hybrid spline matrix. Used by Kernel_G_fermions_grid_transf().
		bool spline_matrix_grid_transf_G_part(vec x, uvec ind_xlims, vec xs, mat &M);
		//convert a matrix to LAPACK band matrix format
		void convert_matrix_to_band_format(mat M, mat &Mbf, int KL, int KU);
		// define the hybrid spline matrix. Used by Kernel_G_fermions_grid_transf_2().
		bool spline_matrix_grid_transf_G_part_2(vec x, uvec ind_xlims, vec xs, mat &M);
		// compute values of the hybrid spline created using spline_matrix_G_part and a given spectrum. Used only during tests.
		bool spline_val_grid_transf(vec x, vec x0, vec coeffs, vec &s);
		// compute the coefficients of hybrid spline for an even function F defined at points x. Used by set_default_model_chi().
		bool spline_chi_part(vec x, uvec ind_xlims, vec xs, vec F, vec &coeffs);
		// compute values of the spline defined by spline_chi_part(). Used by set_default_model_chi().
		bool spline_val_chi_part(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv);
		// compute a single value of the spline defined by spline_chi_part(). Used by KK_integ_chi() called from compute_Re_chi_omega().
		double spline_val_chi_part(double x, vec x0, uvec ind_xlims, vec xs, vec coeffs);
		// define the hybrid spline matrix for the even function cases. Used by kernel_chi().
		bool spline_matrix_chi(vec x, uvec ind_xlims, vec xs, mat &M);
		// compute values of splines. Used during tests in Kernel_G_fermions_grid_transf() and other versions of the same function.
		bool spline_val_G_part_grid_transf(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv);
		// compute values of splines. Used during tests in Kernel_G_fermions_grid_transf() and other versions of the same function.
		bool spline_val_G_part_grid_transf_1(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv);
		// compute a single value of the hybrid spline. Used for tests in Kernel_G_fermions() and Kernel_G_bosons().
		double spline_val_G_part_int(double x, void *par[]);
		
		//for imaginary time data, set the number of time slices to a power of 2
		void set_Ntau_to_pow_2();
		
		// fit a circle arc to compute a curvature
		bool fit_circle_arc(vec x, vec y, vec &arc_params);
		
		//The following functions were used to compute the classic and Bryan MaxEnt result, but are not currently used.
		double integrate_spline(vec x, vec coeffs);
		double diff_gaussian(double x, double par[]);
		double S_i(double u, double c[]);
		double P_A_alpha_val(double u, void*par[]);
		void integrate_P_A_alpha();
		double integrate_P_A_i_alpha(double alphaD_p);
		void normalize_P_alpha_G();
		void compute_Bryan_spectrum(vec &Abr);
		
		// minimization routine, including the loop over decreasing alpha
		void minimize();
		// minimization routine, with alpha increasing. Was used to verify that no hysteresis was present.
		void minimize_increase_alpha();
		
		// plot data using data in vec (armadillo) format.
		void plot(graph_2D &g, vec x, vec y, char *xl, char *yl, char *attr=NULL);
		
		// generate the default main input file OmegaMaxEnt_input_params.dat if it does not exist.
        bool create_default_input_params_file();
		// generate the default internal parameters file OmegaMaxEnt_other_params.dat if it does not exist.
        bool create_default_other_params_file();
		
		//destroy the unwanted files created during the calculation
		void remove_files();
		
		// compute the real frequency Green function with a Pade approximant
		void compute_G_with_Pade(vec wP, int NP, double eta);
		//compute the Fourier transform of the spectrum A(t)=TF[A(w)]
		void Fourier_transform_spectrum(vec wFt, vec AwFt, vec &t, cx_vec &At);
		//compute the real frequency Green function from A(t)
		void compute_G_Re_omega_from_A_t(vec t, cx_vec At, cx_vec &G_Re_omega);
		//compute the real part of the real-frequency Green function Re[G(omega)] using the Kramers-Kronig relation
		void compute_Re_G_omega(vec Ap);
		//compute the real part of the real-frequency correlation function Re[chi(omega)] that has the property chi(-omega)=chi*(omega) using the Kramers-Kronig relation
		void compute_Re_chi_omega(vec Ap);
		//set the output (uniform) real frequency grid. extr_w(0) and extr_w(1) are the extrema of the grid used in the MaxEnt computation (for which the spectrum is defined)
		void set_output_frequency_grid(vec extr_w);
		// perform the Kramers-Kronig integral 
		void KK_integrate(vec w_KK, fctPtr1 Ptr, void *par[], double Rwdw, vec &G_tmp, vec tol, vec lims);
		// perform the Kramers-Kronig integral for even bosonic functions
		void KK_integrate_chi(vec w_KK, fctPtr1 Ptr, void *par[], double Rwdw, vec &G_tmp, vec tol);
		//integrand in the Kramers-Kronig relation, fermionic case
		double KK_integ(double x, void *par[]);
		//integrand in the Kramers-Kronig relation, general bosonic case
		double KK_integ_boson(double x, void *par[]);
		//integrand using linear interpolation in the Kramers-Kronig relation
		double KK_integ_lin_interp(double x, void *par[]);
		//integrand using linear interpolation in the Kramers-Kronig relation, even bosonic case
		double KK_integ_lin_interp_chi(double x, void *par[]);
		//integrand in the Kramers-Kronig relation, symmetric bosonic case
		double KK_integ_chi(double x, void *par[]);
		
        string input_params_file_name;
        
        //! internal computation parameters
		int Nn_min, Nn_max, Nw_min, Nw_max, Nn_fit_max, Nn_fit_fin, Niter_dA_max, Nalpha_max_figs, Nwsamp, Nsmooth_errG;
		
        double f_w_range, f_SW_std_omega, f_width_grid_dens, tol_tem, tol_G_inf, tol_norm, tol_R_G0_Gbeta, tol_M1, tol_M2, tol_M3, default_error_G, err_norm, default_error_M, tol_mean_C1, tol_std_C1, tol_rdw, Rmin_Dw_dw, Rdw_max, RW_grid, RWD_grid, minDefM, f_alpha_init, R_width_ASmin, f_Smin, diff_chi2_max, tol_int_dA, rc2H, pow_alpha_step_init, pow_alpha_step_min, chi2_alpha_smooth_range, f_scale_lalpha_lchi2, FNfitTauW, std_norm_peak_max, varM2_peak_max, peak_weight_min, RMAX_dlchi2_lalpha, f_alpha_min, save_alpha_range,  Rmin_SW_dw, R_peak_width_dw, R_wncutoff_wr, R_Dw_dw, R_SW_wr, R_wmax_wr_min, wgt_min_sm, R_SW_G_Re_w_range, R_dw_min_dw_dense, R_wKK_SW, R_sv_min;
		
		//! input parameters
		string input_dir_in, input_dir, data_file_name_in, data_file_name, boson_in, tau_GF_in, tem_in, M0_in, M1_in, errM1_in, M2_in, errM2_in, M3_in, errM3_in, omega_n_trunc_in, G_omega_inf_in, col_Gr_in, col_Gi_in, error_file_in, error_file, col_errGr_in, col_errGi_in, covar_re_re_file_in, covar_re_re_file, covar_im_im_file_in, covar_im_im_file, covar_re_im_file_in, covar_re_im_file, col_Gtau_in, col_errGtau_in, covar_tau_file_in, covar_tau_file, cutoff_wn_in, SW_in, SC_in, w_origin_in, step_omega_in, grid_omega_file_in, grid_omega_file, use_grid_params_in, omega_grid_params_in, eval_moments_in, maxM_in, def_model_file_in, def_model_file, init_spectr_func_file_in, init_spectr_func_file, default_model_center_in, default_model_width_in, default_model_shape_in, non_uniform_grid_in, Ginf_finite_in, noise_params_in, output_dir_in, output_dir, output_dir_fin, output_name_suffix, output_name_format, w_sample_in, Nalpha_in, alpha_min_in, alpha_init_in, alpha_opt_max_in, alpha_opt_min_in, alpha_save_max_in, alpha_save_min_in, A_ref_file, A_ref_file_in, def_model_output_file_name, A_opt_name_format, A_opt_err_name_format, output_G_format, output_error_format, auto_corr_error_G_format, output_G_opt_format, error_G_opt_format, auto_corr_error_G_opt_format, output_moments_format, output_moments_opt_format, chi2_vs_alpha_format, Asamp_vs_alpha_format, samp_freq_format, A_opt_name, A_opt_name_rm, A_opt_err_name_rm, A_alpha_min_name, output_G_opt_rm, error_G_opt_rm, auto_corr_error_G_opt_rm, output_moments_opt_rm, G_re_omega_name, Pade_G_re_omega_name, G_re_t_name, output_grid_params_in, compute_Pade_in, N_Pade_in, eta_Pade_in;
		//interp_type, interp_type_in
		
        bool use_grid_params, use_const_dw, use_exp_step, displ_prep_figs, displ_adv_prep_figs, print_other_params, boson, tau_GF, initialize, initialize_maxent, execute_maxent, save_spec_func, print_alpha, displ_optim_figs, cov_diag, moments_provided, eval_moments, covm_diag, wc_exists, w_exists, SW_set, SC_set, peak_exists, read_params, read_other_params, params_loaded, other_params_loaded, M1_set, M2_set, main_spectral_region_set, A_ref_change, show_optimal_alpha_figs, show_lowest_alpha_figs, show_alpha_curves, preproc_complete, Du_constant, non_uniform_grid, w_origin_set, interactive_mode, Ginf_finite, alpha_min_too_high, error_provided, compute_Pade, dG_dtau_computed;
		
        double tem, cutoff_wn, SW, SC, w_origin, step_omega, signG, alpha0, alpha0_default, alpha, pow_alpha_step, alpha_min_default, alpha_min, alpha_opt_max, alpha_opt_min, M0, errM0, M1, errM1, M2, errM2, M3, errM3, std_omega, omega_n_trunc, wl, wr, w0l, w0r, dwl, dwr, dw_peak, M0t, M1n, default_model_width, default_model_center, default_model_shape, dlchi2_lalpha_min, dlchi2_lalpha_max, alpha_save_max, alpha_save_min, lchi2_lalpha_lgth, G_omega_inf, eta_Pade;
        
        uint col_Gr, col_errGr, col_errGi, col_Gtau, col_errGtau, Nalpha, Nn, Nn_all, indG_0, indG_f, NM, NMinput, NM_odd, NM_even, Nw, NwA, Nwc, Nw_dense, Nw_out, jfit, ind_cutoff_wn, NGM, Nalpha_max, NAprec, ind0, Ntau, Nn_as_min;
		uvec n, n_all, Nw_lims;
		int maxM, maxM_default, col_Gi, ind_alpha_vec, NnC, ind_curv, ind_curv0, ind_noise, N_params_noise, N_Pade;
		
		mat K, KGM, KGMw, invDw, KG_V, KM, KM_V, COV, CRR, CII, CRI, COVM, COVMfit, Ctau, Ctau_all, green_data, error_data, grid_w_data, def_data, Aw_data, Aref_data, Aprec, Aw_samp;
		rowvec omega_grid_params, w_sample, noise_params, output_grid_params;
		uvec w_sample_ind;
		vec w_out, w_dense, Gr_Re_w, Gi_Re_w, Gr_Re_w_KK, Gi_Re_w_KK, Gi_Re_w_FFT, Gr, Gi, Gchi2, G_V, GM, wn, wn_all, errGr, errGi, errG, errGtau, M, M_V, errM, M_even, M_odd, Mfit, ws, A, A0, Amin, wc, w, wA, dwS, default_model, w_ref, A_ref, chi2_vec, alpha_vec, S_vec, M_ord, Gtau, tau, dlchi2_lalpha_1, curv_lchi2_lalpha_1, grid_dens, P_alpha_G, log_P_alpha_G, dG_tau, d2G_tau, d3G_tau, t_re, dG_w;
		cx_vec G, G_all, G_t_re, GR_Pade;
		cx_mat Kcx;
		uword ind_P_alpha_G_max;
		vec integ_P_A_alpha, pow_alphaD_vec;
		
		bool wn_sign_change, wn_inverted;
		
		bool compute_P_alpha_G, uniform_grid, gaussian_grid_density;
		
		time_t *time_params_file, *time_other_params_file;
		
		mt19937 rnd_gen;
		normal_distribution<double> normal_distr;

    };
    
}

#endif
