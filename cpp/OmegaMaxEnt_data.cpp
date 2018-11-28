/*
 file OmegaMaxEnt_data.cpp
 functions definitions for class "OmegaMaxEnt_data". This is the main source file of the program OmegaMaxEnt that performs the analytic continuation of numerical Matsubara Green and correlation functions.

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

#include "OmegaMaxEnt_data.h"
#include <cstring>
#include <cmath>

extern "C"
{
	//computes the solution to system of linear equations A * X = B ( http://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f_source.html )
    void dgesv_(int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO );
	// computes the solution to a real system of linear equations A * X = B, where A is an N-by-N symmetric positive definite matrix and X and B are N-by-NRHS matrices. ( http://www.netlib.org/lapack/explore-html/d9/d6f/dposv_8f_source.html )
	void dposv_(char *UPLO, int *N, int *NRHS, double *A, int *LDA, double *B, int *LDB, int *INFO );
	//computes the singular value decomposition ( http://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f_source.html )
    void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA, double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO );
	//computes the singular value decomposition. If singular vectors are desired, uses a divide and conquer algorithm. ( http://www.netlib.org/lapack/explore-html/db/db4/dgesdd_8f_source.html )
    void dgesdd_( char *JOBZ, int *M, int *N, double *A, int *LDA, double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *IWORK, int *INFO );
	//computes the eigenvalues and, optionally, the left and/or right eigenvectors for SY matrices ( http://www.netlib.org/lapack/explore-html/dd/d4c/dsyev_8f_source.html )
	void dsyev_( char *JOBZ, char *UPLO, int *N, double *A, int *LDA, double *W, double *WORK, int *LWORK, int *INFO );
	//solves a general Gauss-Markov linear model (GLM) problem. ( http://www.netlib.org/lapack/explore-html/d3/df4/dggglm_8f_source.html )
	void dggglm_(int *N, int *M, int *P, double *A, int *LDA, double *B, int *LDB, double *D, double *X, double *Y, double *WORK, int *LWORK, int *INFO );
	//computes the solution to system of linear equations A * X = B, where A is a band matrix ( http://www.netlib.org/lapack/explore-html/dd/dc2/dgbsv_8f_source.html )
	void dgbsv_(int *N, int *KL, int *KU, int *NRHS, double *AB, int *LDAB, int *IPIV, double *B, int *LDB, int *INFO );
}

bool polyfit(vec x, vec y, int D, double x0, vec &cfs);
bool polyval(double x0, vec cfs, vec x, vec &y);
void remove_spaces_front(string &str);
void remove_spaces_back(string &str);
void remove_spaces_ends(string &str);
void pascal(int n, imat &P);

OmegaMaxEnt_data::OmegaMaxEnt_data(int arg_N, char *args[])
{
    input_params_file_name=default_input_params_file_name;
	
	interactive_mode=true;
	
    if (arg_N>1)
    {
        for (int j=1; j<arg_N; j++)
        {
		//	cout<<"option: "<<args[j]<<endl;
            if (args[j][0]=='-')
            {
                if (!strcmp(args[j],"-nd"))
				{
					graph_2D::display_figures=false;
				}
				else if (!strcmp(args[j],"-np"))
				{
					graph_2D::print_to_file=false;
				}
				else if (!strcmp(args[j],"-ni"))
				{
					interactive_mode=false;
					graph_2D::display_figures=false;
				}
                else
                    cout<<"invalid option: "<<args[j]<<'\n';
            }
            else if (args[j][0])
            {
                input_params_file_name.assign(args[j]);
                cout<<"input parameters file name: "<<input_params_file_name<<'\n';
            }
        }
    }
	
	params_loaded=false;
	other_params_loaded=false;
	initialize=true;
	preproc_complete=false;
	initialize_maxent=true;
	print_other_params=false;
	time_params_file=NULL;
	time_other_params_file=NULL;
	ind_alpha_vec=0;
	rnd_gen.seed(time(NULL));
	NAprec=5;
}

OmegaMaxEnt_data::~OmegaMaxEnt_data()
{
	if (time_params_file) delete time_params_file;
	if (time_other_params_file) delete time_other_params_file;
	graph_2D::figs_ind_file<<'\n';
	graph_2D::figs_ind_file.close();
}

int OmegaMaxEnt_data::loop_run()
{
	init_params();
	
	struct stat file_stat;
	char continue_exec;
	string buf;
	save_spec_func=true;
	bool plot_dlchi2_lalpha=false;
	compute_P_alpha_G=false;
	gaussian_grid_density=false;
	uniform_grid=false;
	success=1;
	
	alpha_save_max=DBL_MIN;
	alpha_save_min=DBL_MIN;
	
	ofstream warnings_file("warnings.txt");
	set_stream_err2(warnings_file);
	
	if (interactive_mode)
	{
		ifstream file("license_info_displayed");
		if (!file)
		{
			file.clear();
			ofstream outputfile("license_info_displayed");
			outputfile.close();
			display_license();
		}
		else
		{
			file.close();
			file.open("notice_displayed");
			if (!file)
			{
				file.clear();
				ofstream outputfile("notice_displayed");
				outputfile.close();
				display_notice();
			}
			else
			{
				file.close();
				cout<<OmegaMaxEnt_notice<<endl;
			}
		}
	}
	
	if (graph_2D::print_to_file && interactive_mode) cout<<"Figures can be displayed by executing the command \"python OmegaMaxEnt_figs_#.py\" where # is one of the indices in file figs_ind.dat\n";
	
	do
	{
		read_params=false;
		read_other_params=false;
		
		graph_2D::reset_figs_ind_file();

		stat(input_params_file_name.c_str(),&file_stat);
		if (time_params_file)
		{
			if (*time_params_file!=file_stat.st_mtime) read_params=true;
		}
		else
		{
			time_params_file=new time_t;
			read_params=true;
		}
		*time_params_file=file_stat.st_mtime;
		
		stat(other_params_file_name.c_str(),&file_stat);
		if (time_other_params_file)
		{
			if (*time_other_params_file!=file_stat.st_mtime) read_other_params=true;
		}
		else
		{
			time_other_params_file=new time_t;
			read_other_params=true;
		}
		*time_other_params_file=file_stat.st_mtime;
	
		if (read_other_params)
			other_params_loaded=load_other_params();
		
		if (read_params)
			params_loaded=load_input_params();
		
		if (print_other_params) other_params_loaded=load_other_params();
	
		if (G_omega_inf_in.size() && G_omega_inf)
		{
			Ginf_finite=true;
		}
		
		if (Ginf_finite && !eval_moments_in.size() && !G_omega_inf_in.size()) eval_moments=true;

		if (!graph_2D::display_figures && !graph_2D::print_to_file)
		{
			show_optimal_alpha_figs=false;
			show_lowest_alpha_figs=false;
			show_alpha_curves=false;
			displ_prep_figs=false;
			displ_adv_prep_figs=false;
		}

		if (params_loaded && other_params_loaded)
		{
			if (initialize)
			{
				preproc_complete=preproc();
				read_params=true;
			}
			
			if (execute_maxent && preproc_complete)
			{
				if (read_params)
				{
					if (output_dir_in.size())
					{
						output_dir_fin=output_dir;
						if (output_dir.back()!='/')
						{
							output_dir+='/';
							output_dir_fin+='/';
						}
					}
					else
					{
						output_dir=input_dir;
						output_dir_fin=input_dir;
					}
					output_dir+="OmegaMaxEnt_files";
					output_dir_fin+="OmegaMaxEnt_final_result";
					if (N_params_noise && !error_provided)
					{
						char err_rel_tmp[20];
						sprintf(err_rel_tmp,"_err_%1.1e",noise_params(ind_noise));
						output_dir+=err_rel_tmp;
						output_dir_fin+=err_rel_tmp;
					}
					output_dir+="/";
					output_dir_fin+="/";
					
//					cout<<"output directories:\n"<<output_dir<<endl<<output_dir_fin<<endl;
					
					output_name_format.assign("spectral_function");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_')
							output_name_format+='_';
						output_name_format+=output_name_suffix;
					}
					if (output_name_format.back()!='_') output_name_format+='_';
					output_name_format.insert(0,output_dir);
					output_name_format+="tem%1.4g_alpha%1.2e.dat";
//					cout<<"name format: "<<output_name_format<<endl;
					
					string def_model_output_name_format("default_model");
				 	if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') def_model_output_name_format+='_';
						def_model_output_name_format+=output_name_suffix;
					}
					if (def_model_output_name_format.back()!='_') def_model_output_name_format+='_';
					def_model_output_name_format+="tem%1.4g.dat";
					char def_model_output_name_tmp[200];
					sprintf(def_model_output_name_tmp,def_model_output_name_format.c_str(),tem);
					def_model_output_file_name.assign(def_model_output_name_tmp);
//					cout<<"def_model_output_file_name: "<<def_model_output_file_name<<endl;
					
					A_alpha_min_name="A_alpha_min";
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') A_alpha_min_name+='_';
						A_alpha_min_name+=output_name_suffix;
					}
					A_alpha_min_name+=".dat";
					
					A_opt_name="optimal_spectral_function";
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') A_opt_name+='_';
						A_opt_name+=output_name_suffix;
					}
					A_opt_name+=".dat";
					
					G_re_omega_name="real_frequency_Green_function";
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') G_re_omega_name+='_';
						G_re_omega_name+=output_name_suffix;
					}
					G_re_omega_name+=".dat";
					
					G_re_t_name="real_time_Green_function";
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') G_re_t_name+='_';
						G_re_t_name+=output_name_suffix;
					}
					G_re_t_name+=".dat";
					
					A_opt_name_rm.clear();
					//A_opt_name_rm.assign("optimal_spectral_function*");
					A_opt_name_format.assign("optimal_spectral_function");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') A_opt_name_format+='_';
						A_opt_name_format+=output_name_suffix;
					}
				 	if (A_opt_name_format.back()!='_') A_opt_name_format+='_';
					A_opt_name_format+="tem%1.4g_alpha%1.2e.dat";
//					cout<<"A_opt_name_format: "<<A_opt_name_format<<endl;
					
					A_opt_err_name_rm.clear();
					A_opt_err_name_format.assign("optimal_spectral_functions");
				 	if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') A_opt_err_name_format+='_';
					 	A_opt_err_name_format+=output_name_suffix;
				 	}
				 	if (A_opt_err_name_format.back()!='_') A_opt_err_name_format+='_';
					A_opt_err_name_format+="tem%1.4g_alpha%1.2e_%1.2e_%1.2e.dat";
//					cout<<"A_opt_err_name_format: "<<A_opt_err_name_format<<endl;

					output_G_format.assign("G_out");
				 	if (output_name_suffix.size())
					{
						if (output_G_format[0]!='_') output_G_format+='_';
				 		output_G_format+=output_name_suffix;
					}
				 	if (output_G_format.back()!='_') output_G_format+='_';
					output_G_format.insert(0,output_dir);
					output_G_format+="tem%1.4g_alpha%1.2e.dat";
//					cout<<"output_G_format: "<<output_G_format<<endl;
					
					output_error_format.assign("error_G_out");
				 	if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') output_error_format+='_';
						output_error_format+=output_name_suffix;
				 	}
				 	if (output_error_format.back()!='_') output_error_format+='_';
					output_error_format.insert(0,output_dir);
					output_error_format+="tem%1.4g_alpha%1.2e.dat";
//					cout<<"output_error_format: "<<output_error_format<<endl;
					
					auto_corr_error_G_format.assign("auto_corr_error_G_out");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') auto_corr_error_G_format+='_';
						auto_corr_error_G_format+=output_name_suffix;
					}
					if (auto_corr_error_G_format.back()!='_') auto_corr_error_G_format+='_';
					auto_corr_error_G_format.insert(0,output_dir);
					auto_corr_error_G_format+="tem%1.4g_alpha%1.2e.dat";
					
					output_G_opt_rm.clear();
					//output_G_opt_rm.assign("G_opt*");
					//output_G_opt_rm.insert(0,output_dir_fin);
					//cout<<output_G_opt_rm<<endl;
					output_G_opt_format.assign("G_opt");
					if (output_name_suffix.size())
					{
						if (output_G_format[0]!='_') output_G_opt_format+='_';
						output_G_opt_format+=output_name_suffix;
					}
					if (output_G_opt_format.back()!='_') output_G_opt_format+='_';
					output_G_opt_format.insert(0,output_dir_fin);
					output_G_opt_format+="tem%1.4g_alpha%1.2e.dat";
					
					error_G_opt_rm.clear();
					//error_G_opt_rm.assign("error_G_opt*");
					//error_G_opt_rm.insert(0,output_dir_fin);
					error_G_opt_format.assign("error_G_opt");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') error_G_opt_format+='_';
						error_G_opt_format+=output_name_suffix;
					}
					if (error_G_opt_format.back()!='_') error_G_opt_format+='_';
					error_G_opt_format.insert(0,output_dir_fin);
					error_G_opt_format+="tem%1.4g_alpha%1.2e.dat";
					
					auto_corr_error_G_opt_rm.clear();
					//auto_corr_error_G_opt_rm.assign("auto_corr_error_G_opt*");
					//auto_corr_error_G_opt_rm.insert(0,output_dir_fin);
					auto_corr_error_G_opt_format.assign("auto_corr_error_G_opt");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') auto_corr_error_G_opt_format+='_';
						auto_corr_error_G_opt_format+=output_name_suffix;
					}
					if (auto_corr_error_G_opt_format.back()!='_') auto_corr_error_G_opt_format+='_';
					auto_corr_error_G_opt_format.insert(0,output_dir_fin);
					auto_corr_error_G_opt_format+="tem%1.4g_alpha%1.2e.dat";
					
					output_moments_format.assign("moments_G_out");
				 	if (output_name_suffix.size())
					{
					 	if (output_name_suffix[0]!='_') output_moments_format+='_';
						output_moments_format+=output_name_suffix;
				 	}
				 	if (output_moments_format.back()!='_') output_moments_format+='_';
					output_moments_format.insert(0,output_dir);
					output_moments_format+="tem%1.4g_alpha%1.2e.dat";
//					cout<<"output_moments_format: "<<output_moments_format<<endl;
					
					output_moments_opt_rm.clear();
					//output_moments_opt_rm.assign("moments_optimal_spectrum*");
					//output_moments_opt_rm.insert(0,output_dir_fin);
					output_moments_opt_format.assign("moments_optimal_spectrum");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') output_moments_opt_format+='_';
						output_moments_opt_format+=output_name_suffix;
					}
					if (output_moments_opt_format.back()!='_') output_moments_opt_format+='_';
					output_moments_opt_format.insert(0,output_dir_fin);
					output_moments_opt_format+="tem%1.4g_alpha%1.2e.dat";
					
					chi2_vs_alpha_format.assign("chi2_vs_alpha");
				 	if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') chi2_vs_alpha_format+='_';
						chi2_vs_alpha_format+=output_name_suffix;
					}
					if (chi2_vs_alpha_format.back()!='_') chi2_vs_alpha_format+='_';
//					chi2_vs_alpha_format.insert(0,output_dir);
					chi2_vs_alpha_format+="tem%1.4g.dat";
					
					Asamp_vs_alpha_format.assign("Asamp_vs_alpha");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') Asamp_vs_alpha_format+='_';
						Asamp_vs_alpha_format+=output_name_suffix;
				 	}
					if (Asamp_vs_alpha_format.back()!='_') Asamp_vs_alpha_format+='_';
//					Asamp_vs_alpha_format.insert(0,output_dir);
					Asamp_vs_alpha_format+="tem%1.4g.dat";
					
					samp_freq_format.assign("sample_freq");
					if (output_name_suffix.size())
					{
						if (output_name_suffix[0]!='_') samp_freq_format+='_';
						samp_freq_format+=output_name_suffix;
					}
					if (samp_freq_format.back()!='_') samp_freq_format+='_';
					samp_freq_format+="tem%1.4g.dat";
					
					if (initialize_maxent)
					{
						if (!alpha_init_in.size()) alpha0=alpha0_default;
						if (!alpha_min_in.size()) alpha_min=alpha_min_default;
					}
					
					if (A_ref_file.size() && (A_ref_change || initialize_maxent))
					{
						if (!set_A_ref()) cout<<"warning: reference spectrum could not be defined\n";
					}
				}
				
				double pow_alpha0=log10(alpha0), pow_alpha_min=log10(alpha_min);
			
			//	if (alpha_min<alpha0)
			//		Nalpha_max=(pow_alpha0-pow_alpha_min)/pow_alpha_step_min;
			//	else
			//		Nalpha_max=10;
			
				if (!alpha_opt_max_in.size())
					alpha_opt_max=alpha0;
				
				if (!alpha_opt_min_in.size())
					alpha_opt_min=DBL_MIN;
				
				if (w_sample_in.size())
					Nwsamp=w_sample.n_cols;
				else
				{
					if (col_Gi>0)
					{
						double tol_wsamp=1e-2;
						rowvec w_sample_tmp=linspace<rowvec>(wl,wr,Nwsamp);
						int j=0;
						while (j<Nwsamp && w_sample_tmp(j)<0) j++;
						if (j<Nwsamp)
						{
							if (w_sample_tmp(j)/SW<tol_wsamp)
							{
								w_sample_tmp(j)=0;
								w_sample=w_sample_tmp;
							}
							else if (j>0 && fabs(w_sample_tmp(j-1))/SW<tol_wsamp)
							{
								w_sample_tmp(j-1)=0;
								w_sample=w_sample_tmp;
							}
							else
							{
								w_sample.zeros(Nwsamp+1);
								if (j>0)	w_sample.cols(0,j-1)=w_sample_tmp.cols(0,j-1);
								w_sample.cols(j+1,Nwsamp)=w_sample_tmp.cols(j,Nwsamp-1);
								w_sample(j)=0;
							}
						}
						else if (fabs(w_sample_tmp(j-1))/SW<tol_wsamp)
						{
							w_sample_tmp(j-1)=0;
							w_sample=w_sample_tmp;
						}
						else
							w_sample=w_sample_tmp;
						
					}
					else
						w_sample=linspace<rowvec>(0,wr,Nwsamp);
				}

				if (col_Gi>0)
					w_sample_ind.ones(Nwsamp);
				else
					w_sample_ind.zeros(Nwsamp);
				
				if (col_Gi>0)
					w_sample_ind=2*w_sample_ind;
				
				int indmax=Nw-3;
				if (col_Gi<=0)	indmax=Nw-2;
				int j;
				for (j=0; j<Nwsamp; j++)
					while (abs(w_sample(j)-w(w_sample_ind(j)+1))<abs(w_sample(j)-w(w_sample_ind(j))) && w_sample_ind(j)<indmax)	w_sample_ind(j)++;
				
				w_sample=trans(w.rows(w_sample_ind));
				
				if (col_Gi>0) w_sample_ind=w_sample_ind-1;
				
				if (initialize_maxent)
				{
					initialize_maxent=false;
					alpha_min_too_high=false;
					
					if (!alpha_init_in.size()) alpha0=alpha0_default;
					if (!alpha_min_in.size()) alpha_min=alpha_min_default;
					
					if (output_dir_in.size())
					{
						if (stat(output_dir_in.c_str(),&file_stat))
						{
							cout<<"creating output directory: "<<output_dir_in<<endl;
							mkdir(output_dir_in.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
						}
					}
					if (stat(output_dir.c_str(),&file_stat))
					{
						cout<<"creating output directory: "<<output_dir<<endl;
						mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
					}
					if (stat(output_dir_fin.c_str(),&file_stat))
					{
						cout<<"creating output directory: "<<output_dir_fin<<endl;
						mkdir(output_dir_fin.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
					}
					
					mat save_mat=zeros<mat>(Nw,2);
					save_mat.col(0)=w;
					if (col_Gi>0)
						save_mat.submat(1,1,Nw-2,1)=default_model/exp(1);
					else
						save_mat.submat(0,1,Nw-2,1)=default_model/exp(1);
					string file_name(output_dir);
					file_name+=def_model_output_file_name;
					save_mat.save(file_name.c_str(),raw_ascii);
					file_name.assign(output_dir_fin);
					file_name+=def_model_output_file_name;
					save_mat.save(file_name.c_str(),raw_ascii);
					
					A=A0;
					Aprec=A0*ones<rowvec>(NAprec);
					alpha=alpha0;
					pow_alpha_step=pow_alpha_step_init;
					
					alpha_vec.zeros(Nalpha_max);
					chi2_vec.zeros(Nalpha_max);
					S_vec.zeros(Nalpha_max);
					if (compute_P_alpha_G) log_P_alpha_G.zeros(Nalpha_max);
					Aw_samp.zeros(Nalpha_max,Nwsamp);
					ind_alpha_vec=0;
					
					lchi2_lalpha_lgth=0;
					ind_curv=0;
					ind_curv0=0;
					curv_lchi2_lalpha_1.zeros(Nalpha_max);
					dlchi2_lalpha_1.zeros(Nalpha_max);
					
					cout<<"\nStarting minimization at initial value of alpha\n";
					
					/*
					{
						graph_2D g1;
						char xl[]="$\\\\omega$";
						char yl[]="initial spectrum";
						plot(g1, wA, A, xl, yl, NULL);
						graph_2D::show_figures();
					}
					*/
				}
				
				if (read_params) copy_file(input_params_file_name, "./", output_dir_fin);
				if (read_other_params) copy_file(other_params_file_name, "./", output_dir_fin);
				
				double pow_alpha=log10(alpha);
				if (pow_alpha_min<=pow_alpha)
					Nalpha_max=(pow_alpha-pow_alpha_min)/pow_alpha_step_min;
				else
					Nalpha_max=10;
				if (!Nalpha_in.size()) Nalpha=Nalpha_max;
				
				int Nalpha_new=ind_alpha_vec+Nalpha;
				if (alpha_vec.n_rows<Nalpha_new)
				{
					alpha_vec.resize(Nalpha_new);
					chi2_vec.resize(Nalpha_new);
					S_vec.resize(Nalpha_new);
					if (compute_P_alpha_G) log_P_alpha_G.resize(Nalpha_new);
					Aw_samp.resize(Nalpha_new,Nwsamp);
					curv_lchi2_lalpha_1.resize(Nalpha_new);
					dlchi2_lalpha_1.resize(Nalpha_new);
				}
				
				minimize();
//				pow_alpha_step=pow_alpha_step_min;
//				minimize_increase_alpha();
				
				vec Acl, Abr;
				if (compute_P_alpha_G)
				{
					normalize_P_alpha_G();
					
					P_alpha_G.max(ind_P_alpha_G_max);
					
					char file_name[200];
					mat data;
					sprintf(file_name, output_name_format.c_str(),tem,alpha_vec(ind_P_alpha_G_max));
					if (data.load(file_name))
					{
						Acl=data.col(1);
					}
					else
						cout<<"spectral function was not saved at alpha= "<<alpha_vec(ind_P_alpha_G_max)<<endl;
					
					compute_Bryan_spectrum(Abr);
				}
				
				char alpha_output[100], alpha_output_format[]="alpha: % 1.4e,  Q: % 1.4e,  S: % 1.4e,  chi2: % 1.4e\n";
				double Q=chi2_vec(ind_alpha_vec-1)-alpha_vec(ind_alpha_vec-1)*S_vec(ind_alpha_vec-1);
				sprintf(alpha_output,alpha_output_format,alpha_vec(ind_alpha_vec-1),Q,S_vec(ind_alpha_vec-1),chi2_vec(ind_alpha_vec-1));
				cout<<"last output values:\n";
				cout<<alpha_output;
				
				vec lalpha=log10(alpha_vec.rows(0,ind_alpha_vec-1));
				vec lchi2=log10(chi2_vec.rows(0,ind_alpha_vec-1));
				
				char file_name[200];
				sprintf(file_name, chi2_vs_alpha_format.c_str(),tem);
				mat save_mat=join_rows(alpha_vec.rows(0,ind_alpha_vec-1),chi2_vec.rows(0,ind_alpha_vec-1));
				string name(output_dir);
				name.append(file_name);
				save_mat.save(name.c_str(), raw_ascii);
				name.assign(output_dir_fin);
				name.append(file_name);
				save_mat.save(name.c_str(), raw_ascii);
    
				sprintf(file_name, Asamp_vs_alpha_format.c_str(),tem);
				save_mat=join_rows(alpha_vec.rows(0,ind_alpha_vec-1),Aw_samp.rows(0,ind_alpha_vec-1));
				name.assign(output_dir);
				name.append(file_name);
				save_mat.save(name.c_str(), raw_ascii);
				name.assign(output_dir_fin);
				name.append(file_name);
				save_mat.save(name.c_str(), raw_ascii);
				
				sprintf(file_name, samp_freq_format.c_str(),tem);
				name.assign(output_dir);
				name.append(file_name);
//				cout<<"sample freq name: "<<name<<endl;
				w_sample.save(name.c_str(), raw_ascii);
				name.assign(output_dir_fin);
				name.append(file_name);
//				cout<<"sample freq name: "<<name<<endl;
				w_sample.save(name.c_str(), raw_ascii);

				vec DM=default_model/exp(1);
				
				if (ind_alpha_vec>3)
				{
					//cout<<"ind_alpha_vec: "<<ind_alpha_vec<<endl;
					
					double fs=f_scale_lalpha_lchi2;
					double smooth_length=chi2_alpha_smooth_range;
					int ind_curv_start, ind_curv_end;
					double la, lc, ls;
					j=1;
					la=lalpha(j-1)-lalpha(j);
					lc=lchi2(j-1)-lchi2(j);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && j<ind_alpha_vec-1)
					{
						j++;
						la=lalpha(j-1)-lalpha(j);
						lc=lchi2(j-1)-lchi2(j);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					ind_curv_start=j;
					//cout<<"ind_curv_start: "<<ind_curv_start<<endl;
					
					j=ind_alpha_vec-2;
					la=lalpha(j)-lalpha(j+1);
					lc=lchi2(j)-lchi2(j+1);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && j>0)
					{
						j--;
						la=lalpha(j)-lalpha(j+1);
						lc=lchi2(j)-lchi2(j+1);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					ind_curv_end=j;
					//cout<<"ind_curv_end: "<<ind_curv_end<<endl;
					
					int Ncurv=ind_curv_end-ind_curv_start+1;
					//cout<<"Ncurv: "<<Ncurv<<endl;
					
					vec dlchi2_lalpha=zeros<vec>(Ncurv);
					vec curv_lchi2_lalpha=zeros<vec>(Ncurv);
					
					vec arc_params;
					int jmin, jmax;
					
					double xc, yc, x1, y1;
					
					vec cfs_poly(2);
					for (j=ind_curv_start; j<=ind_curv_end; j++)
					{
						jmin=j-1;
						la=lalpha(jmin)-lalpha(jmin+1);
						lc=lchi2(jmin)-lchi2(jmin+1);
						ls=sqrt(pow(fs*la,2)+pow(lc,2));
						while (ls<fs*smooth_length && jmin>0)
						{
							jmin--;
							la=lalpha(jmin)-lalpha(jmin+1);
							lc=lchi2(jmin)-lchi2(jmin+1);
							ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
						}
						jmax=j+1;
						la=lalpha(jmax-1)-lalpha(jmax);
						lc=lchi2(jmax-1)-lchi2(jmax);
						ls=sqrt(pow(fs*la,2)+pow(lc,2));
						while (ls<fs*smooth_length && jmax<ind_alpha_vec-1)
						{
							jmax++;
							la=lalpha(jmax-1)-lalpha(jmax);
							lc=lchi2(jmax-1)-lchi2(jmax);
							ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
						}
						fit_circle_arc(fs*lalpha.rows(jmin,jmax), lchi2.rows(jmin,jmax), arc_params);
						curv_lchi2_lalpha(j-ind_curv_start)=1.0/arc_params(0);
						
						xc=arc_params(1);
						yc=arc_params(2);
						x1=fs*lalpha(j);
						y1=lchi2(j);
						
						if (xc==xc && yc==yc) dlchi2_lalpha(j-ind_curv_start)=(xc-x1)/(y1-yc);
						else
						{
							
							if (polyfit(fs*lalpha.rows(jmin,jmax), lchi2.rows(jmin,jmax), 1, 0.5*fs*(lalpha(jmin)+lalpha(jmax)), cfs_poly))
								dlchi2_lalpha(j-ind_curv_start)=cfs_poly(0);
							else
								dlchi2_lalpha(j-ind_curv_start)=0;
						}
					 
						//x1=fs*lalpha(j-1)-xc;
						//y1=lchi2(j-1)-yc;
						//x2=fs*lalpha(j)-xc;
						//y2=lchi2(j)-yc;
						//v1=sqrt(x1*x1+y1*y1);
						//v2=sqrt(x2*x2+y2*y2);
						//angle(j-1)=acos((x1*x2+y1*y2)/(v1*v2));
						//total_curv_lchi2_lalpha(j-1)=sum(sign(curv_lchi2_lalpha.rows(0,j-1)) % angle.rows(0,j-1));
					}
					
					uword ind_dlchi2_lalpha_max;
					dlchi2_lalpha_max=dlchi2_lalpha.max(ind_dlchi2_lalpha_max);
					dlchi2_lalpha_min=dlchi2_lalpha(Ncurv-1);
					
					int N_av=5;
					double dlchi2_lalpha_min_av=sum(dlchi2_lalpha.rows(Ncurv-N_av,Ncurv-1))/N_av;
					
					int ind_min_curv=0;
					if (alpha_opt_max_in.size())
					{
						while (alpha_vec(ind_min_curv)>alpha_opt_max && ind_min_curv<ind_alpha_vec-1)
							ind_min_curv++;
					}
					else
					{
						ind_min_curv=ind_dlchi2_lalpha_max+ind_curv_start;
					}
					
				//	cout<<"alpha_opt_max: "<<alpha_vec(ind_min_curv)<<endl;
					
					ind_min_curv=ind_min_curv-ind_curv_start;
					if (ind_min_curv<0) ind_min_curv=0;
					int ind_max_curv=ind_alpha_vec-1;
					while (alpha_vec(ind_max_curv)<alpha_opt_min && ind_max_curv>0)	 ind_max_curv--;
					ind_max_curv=ind_max_curv-ind_curv_start;
					if (ind_max_curv>=Ncurv) ind_max_curv=Ncurv-1;
					if (ind_max_curv<ind_min_curv) ind_max_curv=ind_min_curv;
					
					int ind_alpha_opt, ind_alpha_opt_l, ind_alpha_opt_r;
					
					double max_curv;
					if (ind_min_curv<Ncurv && ind_max_curv<Ncurv)
					{
						uword ind_alpha_opt_tmp;
						vec curv=curv_lchi2_lalpha.rows(ind_min_curv,ind_max_curv);
						max_curv=curv.max(ind_alpha_opt_tmp);
						ind_alpha_opt=ind_alpha_opt_tmp+ind_min_curv;
						
						j=ind_alpha_opt;
						while (curv_lchi2_lalpha(j)>max_curv/2 && j>0)  j--;
						ind_alpha_opt_r=j+ind_curv_start;
						j=ind_alpha_opt;
						while (curv_lchi2_lalpha(j)>max_curv/2 && j<Ncurv-1) j++;
						ind_alpha_opt_l=j+ind_curv_start;
						ind_alpha_opt=ind_alpha_opt+ind_curv_start;
					}
					else
					{
						ind_alpha_opt=ind_alpha_vec-1;
					}
					
					vec G_V_out;
					G_V_out=KG_V*A;
					
					vec M_out, M_V_out, errM_out;
					if (NM>0)
					{
						if (!covm_diag) errM=sqrt(COVM.diag());
						M_out=KM*A;
						errM_out=(M-M_out)/errM;
						M_V_out=KM_V*A;
					}
					
					graph_2D *gM[2];
					if (NM>1)
					{
						gM[0]=new graph_2D;
						gM[1]=new graph_2D;
					}

					if (show_lowest_alpha_figs)
					{
						if (NM>1)
						{
							char ttl_M[]="normalized deviation of moments at lowest $\\\\alpha$";
							char xl_M[]="moment order", yl_M[]="$(M_{in}-M_{out})/\\\\sigma_M$";
							char attrM[]="'o', color='m', markeredgecolor='m'";
							gM[0]->add_title(ttl_M);
							plot(*gM[0], M_ord, errM_out, xl_M, yl_M, attrM);
						}
						else if (NM>0)
						{
							cout<<"normalized deviation for moment of order "<<M_ord(0)<<": "<<errM_out(0)<<endl;
						}
					}
					
					vec errRe, errIm;
					if (!boson || col_Gi>0)
					{
						if (cov_diag)
						{
							uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
							errRe=G_V.rows(even_ind)-G_V_out.rows(even_ind);
							errIm=G_V.rows(even_ind+1)-G_V_out.rows(even_ind+1);
						}
						else
						{
							errRe=G_V-G_V_out;
							//errRe=G_V.rows(0,Nn-1)-G_V_out.rows(0,Nn-1);
							//errIm=G_V.rows(Nn,2*Nn-1)-G_V_out.rows(Nn,2*Nn-1);
						}
					}
					else
						errRe=G_V-G_V_out;
					
					vec eigv_ind;
					if (cov_diag)
						eigv_ind=wn;
					else
						eigv_ind=linspace<vec>(0,errRe.n_rows-1,errRe.n_rows);
					
					
					int l;
					vec CRe, CIm;
					if (!boson || col_Gi>0)
					{
						if (cov_diag)
						{
							CRe.zeros(NnC);
							CIm.zeros(NnC);
							for (j=0; j<NnC; j++)
							{
								for (l=0; l<Nn-j; l++)
								{
									CRe(j)=CRe(j)+errRe(l)*errRe(l+j);
									CIm(j)=CIm(j)+errIm(l)*errIm(l+j);
								}
								CRe(j)=CRe(j)/(Nn-j);
								CIm(j)=CIm(j)/(Nn-j);
							}
						}
						else
						{
							CRe.zeros(NnC);
							for (j=0; j<NnC; j++)
							{
								for (l=0; l<2*Nn-j; l++)
								{
									CRe(j)=CRe(j)+errRe(l)*errRe(l+j);
								}
								CRe(j)=CRe(j)/(2*Nn-j);
							}
						}
							
					}
					else
					{
						CRe.zeros(NnC);
						for (j=0; j<NnC; j++)
						{
							for (l=0; l<Nn-j; l++)
								CRe(j)=CRe(j)+errRe(l)*errRe(l+j);
							CRe(j)=CRe(j)/(Nn-j);
						}
					}
					
					int Ngraph=22;
					graph_2D *gv[Ngraph];
					for (j=0; j<Ngraph; j++) gv[j]=new graph_2D;
					
					double xlims[2], ylims[2];
					
					vec Dn;
					Dn=linspace<vec>(0,NnC-1,NnC);
					
					char xl_w[]="$\\\\omega$";
					
					char ttlchi2[]="$\\\\log_{10}(\\\\chi^2)$ vs $\\\\log_{10}(\\\\alpha)$";
					char xl0[]="$\\\\log_{10}(\\\\alpha)$", yl0[]="$\\\\log_{10}(\\\\chi^2)$";
					char attr0[]="'o', color='k'";
					
					char ttlcurv[]="curvature of $\\\\log_{10}(\\\\chi^2)$ vs $\\\\gamma \\\\log_{10}(\\\\alpha)$";
					char yl1[]="$\\\\kappa$";
					char attr1[]="'o-', color='k', markeredgecolor='k'";
					
					char ttlPalpha[]="$P(\\\\alpha,|G)$ in classic and Bryan's methods";
					char ttl_log_Palpha[]="$\\\\log[P(\\\\alpha,|G)]$";
					char yl2[]="$P(\\\\alpha,|G)$";
					char yl3[]="$\\\\log[P(\\\\alpha,|G)]$";
					char attr2[]="'o-', color='m', markeredgecolor='m'";
					
					xlims[0]=lalpha.min()-0.05*(lalpha.max()-lalpha.min());
					xlims[1]=lalpha.max()+0.05*(lalpha.max()-lalpha.min());
					ylims[0]=lchi2.min()-0.05*(lchi2.max()-lchi2.min());
					ylims[1]=lchi2.max()+0.05*(lchi2.max()-lchi2.min());
					
					int ind_fig=0;
					if (show_alpha_curves)
					{
						gv[ind_fig]->add_data(lalpha.memptr(),lchi2.memptr(),ind_alpha_vec);
						gv[ind_fig]->set_axes_labels(xl0, yl0);
						gv[ind_fig]->set_axes_lims(xlims,ylims);
						gv[ind_fig]->add_attribute(attr0);
						gv[ind_fig]->add_title(ttlchi2);
						gv[ind_fig]->curve_plot();
						ind_fig++;
						
						//plot(*gv[ind_fig], lalpha.rows(1,Ncurv), curv_lchi2_lalpha, xl0, yl1, attr1);
						gv[ind_fig]->add_title(ttlcurv);
						plot(*gv[ind_fig], lalpha.rows(ind_curv_start,Ncurv+ind_curv_start-1), curv_lchi2_lalpha.rows(0,Ncurv-1), xl0, yl1, attr1);
						ind_fig++;
						
						if (compute_P_alpha_G)
						{
							gv[ind_fig]->add_title(ttlPalpha);
							plot(*gv[ind_fig], lalpha, P_alpha_G.rows(0,ind_alpha_vec-1), xl0, yl2, attr2);
							ind_fig++;
							
							//	gv[ind_fig]->add_title(ttl_log_Palpha);
							//	plot(*gv[ind_fig], lalpha, log_P_alpha_G.rows(0,ind_alpha_vec-1), xl0, yl3, attr2);
							//	ind_fig++;
						}
						
						char yS[]="S";
						char ttlS[]="relative entropy";
						gv[ind_fig]->add_title(ttlS);
						plot(*gv[ind_fig], lalpha, S_vec.rows(0,ind_alpha_vec-1), xl0, yS, attr1);
						ind_fig++;
						
						
						
						//char ylc[]="total curv[log(chi2) vs log(alpha)]";
						//plot(*gv[13], lalpha.rows(1,Ncurv), total_curv_lchi2_lalpha, xl0, ylc, attr1);
					}
					
					if (plot_dlchi2_lalpha)
					{
						char attrc[]="'^-', color='r', markeredgecolor='r'";
						char attrd[]="'v-', color='b', markeredgecolor='b'";
						char attrd1[]="'s-', color='m', markeredgecolor='m'";
						
						char yld[]="dlog(chi2)/dlog(alpha)";
						plot(*gv[ind_fig], lalpha.rows(ind_curv_start,Ncurv+ind_curv_start-1), dlchi2_lalpha.rows(0,Ncurv-1), xl0, yld, attrd);
						ind_fig++;
						
						gv[ind_fig]->add_title(ttlcurv);
						plot(*gv[ind_fig], lalpha.rows(ind_curv0,ind_curv0+ind_curv), curv_lchi2_lalpha_1.rows(0,ind_curv), xl0, yl1, attrc);
						ind_fig++;
						
						plot(*gv[ind_fig], lalpha.rows(ind_curv0,ind_curv0+ind_curv), dlchi2_lalpha_1.rows(0,ind_curv), xl0, yld, attrd1);
						ind_fig++;
					}
					
					if (show_lowest_alpha_figs)
					{
						char ttl2[]="Spectrum at lowest $\\\\alpha$ computed";
						char lgd_DM[]="default model", lgd_format_A[]="$\\\\alpha=%4.2e$", lgd_A[100];
						
						if (col_Gi>0)
						{
							xlims[0]=SC-2*SW;
							xlims[1]=SC+2*SW;
						}
						else
						{
							xlims[0]=0;
							xlims[1]=2*SW;
						}
						vec A_all=join_vert(A,DM);
						
						//cout<<"SC, SW, xlims: "<<setw(20)<<SC<<setw(20)<<SW<<setw(20)<<xlims[0]<<setw(20)<<xlims[1]<<endl;
						
						gv[ind_fig]->add_to_legend(lgd_DM);
						gv[ind_fig]->add_data(wA.memptr(),DM.memptr(),NwA);
						if (A_ref_file.size() && A_ref.n_rows)
						{
							char lgd_refA[]="reference spectrum";
							gv[ind_fig]->add_to_legend(lgd_refA);
							gv[ind_fig]->add_data(w_ref.memptr(),A_ref.memptr(),w_ref.n_rows);
							A_all=join_vert(A_all,A_ref);
						}
						ylims[0]=0;
						ylims[1]=1.1*A_all.max();
						
						sprintf(lgd_A,lgd_format_A,alpha_vec(ind_alpha_vec-1));
						gv[ind_fig]->add_to_legend(lgd_A);
						gv[ind_fig]->add_data(wA.memptr(),A.memptr(),NwA);
						gv[ind_fig]->set_axes_labels(xl_w,NULL);
						gv[ind_fig]->set_axes_lims(xlims,ylims);
						gv[ind_fig]->add_title(ttl2);
						gv[ind_fig]->curve_plot();
						ind_fig++;
						
						char xl3[50], xl5[50], yl3[50], yl4[50], yl5[50], yl6[50];
						char ttl3[100], ttl4[100], ttl5[100], ttl6[100];
						if (cov_diag)
						{
							strcpy(xl3,"$\\\\omega_n$");
							strcpy(xl5,"frequency index difference");
							strcpy(yl3,"$\\\\Delta \\\\tilde{G}_{Re}$");
							strcpy(yl4,"$\\\\Delta \\\\tilde{G}_{Im}$");
							strcpy(yl5,"$AC[\\\\Delta \\\\tilde{G}_{Re}]$");
							strcpy(yl6,"$AC[\\\\Delta \\\\tilde{G}_{Im}]$");
							strcpy(ttl3,"normalized deviation of $Re[G]$ at lowest $\\\\alpha$");
							strcpy(ttl4,"normalized deviation of $Im[G]$ at lowest $\\\\alpha$");
							strcpy(ttl5,"autocorrelation of $\\\\Delta \\\\tilde{G}_{Re}$ at lowest $\\\\alpha$");
							strcpy(ttl6,"autocorrelation of $\\\\Delta \\\\tilde{G}_{Im}$ at lowest $\\\\alpha$");
						}
						else
						{
							strcpy(xl3,"covar. eigen. index");
							strcpy(xl5,"covar. eigen. index difference");
							strcpy(yl3,"$\\\\Delta \\\\tilde{G}$");
							strcpy(yl5,"$AC[\\\\Delta \\\\tilde{G}]$");
							strcpy(ttl3,"normalized deviation of $G$ at lowest $\\\\alpha$");
							strcpy(ttl5,"autocorrelation of $\\\\Delta \\\\tilde{G}$ at lowest $\\\\alpha$");
						}
						char attr3[]="'o-', color='b', markeredgecolor='b'", attr4[]="'o-', color='r', markeredgecolor='r'";
						
						if (cov_diag)
						{
							gv[ind_fig]->add_title(ttl3);
							plot(*gv[ind_fig], wn, errRe, xl3, yl3, attr3);
							ind_fig++;
							gv[ind_fig]->add_title(ttl5);
							plot(*gv[ind_fig], Dn, CRe, xl5, yl5, attr3);
							ind_fig++;
							if (col_Gi>0)
							{
								gv[ind_fig]->add_title(ttl4);
								plot(*gv[ind_fig], wn, errIm, xl3, yl4, attr4);
								ind_fig++;
								gv[ind_fig]->add_title(ttl6);
								plot(*gv[ind_fig], Dn, CIm, xl5, yl6, attr4);
								ind_fig++;
							}
						}
						else
						{
							gv[ind_fig]->add_title(ttl3);
							plot(*gv[ind_fig], eigv_ind, errRe, xl3, yl3, attr3);
							ind_fig++;
							gv[ind_fig]->add_title(ttl5);
							plot(*gv[ind_fig], Dn, CRe, xl5, yl5, attr3);
							ind_fig++;
						}
					}
					
					char ttl7[]="spectrum at sample frequencies", yl7[]="$A(\\\\omega_{samp})$";
					char lgd_tmp[20], lgd_samp_format[]="$\\\\omega=%4.2e$";
					char attr7[]="'o-'";
					vec Asamp_tmp;
					
					xlims[0]=lalpha.min()-0.05*(lalpha.max()-lalpha.min());
					xlims[1]=lalpha.max()+0.05*(lalpha.max()-lalpha.min());
					ylims[0]=0;
					ylims[1]=1.05*max(max(Aw_samp));
					
					//cout<<"Nwsamp: "<<Nwsamp<<endl;
					//cout<<"Aw_samp.n_cols: "<<Aw_samp.n_cols<<endl;
					if (show_alpha_curves)
					{
						gv[ind_fig]->legend_loc=1;
						gv[ind_fig]->add_title(ttl7);
						gv[ind_fig]->set_axes_labels(xl0,yl7);
						for (j=0; j<Nwsamp; j++)
						{
							Asamp_tmp=Aw_samp.col(j);
							Asamp_tmp=Asamp_tmp.rows(0,ind_alpha_vec-1);
							sprintf(lgd_tmp,lgd_samp_format,w_sample(j));
							gv[ind_fig]->add_to_legend(lgd_tmp);
							gv[ind_fig]->add_attribute(attr7);
							gv[ind_fig]->add_data(lalpha.memptr(),Asamp_tmp.memptr(),ind_alpha_vec);
						}
						gv[ind_fig]->set_axes_lims(xlims,ylims);
						gv[ind_fig]->curve_plot();
						ind_fig++;
					}
					
					//if ( (chi2_vec(ind_alpha_opt)<chi2save || alpha_vec(ind_alpha_opt)<alpha_save) && D2curv_max<0 && ind_alpha_opt_l<Ncurv)
					if (max_curv>0 && (ind_alpha_opt_l-ind_curv_start)<Ncurv-1)
					{
						double f_alpha_save=pow(10,save_alpha_range);
						//cout<<"f_alpha_save: "<<f_alpha_save<<endl;
						if (!alpha_save_max_in.size()) alpha_save_max=f_alpha_save*alpha_vec(ind_alpha_opt);
						if (!alpha_save_min_in.size()) alpha_save_min=alpha_vec(ind_alpha_opt)/f_alpha_save;
						
						cout<<"optimal value of alpha: "<<alpha_vec(ind_alpha_opt)<<endl;
						cout<<"chi2 at optimal alpha: "<<chi2_vec(ind_alpha_opt)<<endl;
						if (compute_P_alpha_G)
						{
							cout<<"optimal value of alpha for classic MaxEnt (alpha_cl): "<<alpha_vec(ind_P_alpha_G_max)<<endl;
							cout<<"chi2 at alpha_cl: "<<chi2_vec(ind_P_alpha_G_max)<<endl;
						}
						
						mat data;
						vec A_opt, A_opt_l, A_opt_r;
						A_opt.reset();
						A_opt_l.reset();
						A_opt_r.reset();
						
						sprintf(file_name, output_name_format.c_str(),tem,alpha_vec(ind_alpha_opt));
						if (data.load(file_name))
							A_opt=data.col(1);
						else
							cout<<"spectral function was not saved at alpha= "<<alpha_vec(ind_alpha_opt)<<endl;
						
						sprintf(file_name, output_name_format.c_str(),tem,alpha_vec(ind_alpha_opt_r));
						if (data.load(file_name))
							A_opt_r=data.col(1);
						else
							cout<<"spectral function was not saved at alpha= "<<alpha_vec(ind_alpha_opt_r)<<endl;
						
						sprintf(file_name, output_name_format.c_str(),tem,alpha_vec(ind_alpha_opt_l));
						if (data.load(file_name))
							A_opt_l=data.col(1);
						else
							cout<<"spectral function was not saved at alpha= "<<alpha_vec(ind_alpha_opt_l)<<endl;
						
						if (A_opt.n_rows && A_opt_l.n_rows && A_opt_r.n_rows)
						{
							string file_name_str=output_dir_fin;
							file_name_str+=A_opt_name;
							remove(file_name_str.c_str());
							data.col(1)=A_opt;
							data.save(file_name_str,raw_ascii);
							
							if (col_Gi) compute_Re_G_omega(A_opt);
							else compute_Re_chi_omega(A_opt);
							
							string name_format(output_dir);
							if (A_opt_name_rm.size())
							{
								name_format+=A_opt_name_rm;
								remove(name_format.c_str());
								name_format=output_dir_fin;
								name_format+=A_opt_name_rm;
								remove(name_format.c_str());
								name_format=output_dir;
							}
							name_format+=A_opt_name_format;
							sprintf(file_name, name_format.c_str(),tem,alpha_vec(ind_alpha_opt));
							data.col(1)=A_opt;
							data.save(file_name,raw_ascii);
							name_format=output_dir_fin;
							name_format+=A_opt_name_format;
							sprintf(file_name, name_format.c_str(),tem,alpha_vec(ind_alpha_opt));
							data.save(file_name,raw_ascii);
							sprintf(file_name, A_opt_name_format.c_str(),tem,alpha_vec(ind_alpha_opt));
							A_opt_name_rm.assign(file_name);
							
							name_format=output_dir;
							if (A_opt_err_name_rm.size())
							{
								name_format+=A_opt_err_name_rm;
								remove(name_format.c_str());
								name_format=output_dir_fin;
								name_format+=A_opt_err_name_rm;
								remove(name_format.c_str());
								name_format=output_dir;
							}
							data.resize(Nw,4);
							data.col(1)=A_opt_l;
							data.col(2)=A_opt;
							data.col(3)=A_opt_r;
							name_format+=A_opt_err_name_format;
							sprintf(file_name, name_format.c_str(),tem,alpha_vec(ind_alpha_opt_l),alpha_vec(ind_alpha_opt),alpha_vec(ind_alpha_opt_r));
							data.save(file_name,raw_ascii);
							name_format=output_dir_fin;
							name_format+=A_opt_err_name_format;
							sprintf(file_name, name_format.c_str(),tem,alpha_vec(ind_alpha_opt_l),alpha_vec(ind_alpha_opt),alpha_vec(ind_alpha_opt_r));
							data.save(file_name,raw_ascii);
							sprintf(file_name, A_opt_err_name_format.c_str(),tem,alpha_vec(ind_alpha_opt_l),alpha_vec(ind_alpha_opt),alpha_vec(ind_alpha_opt_r));
							A_opt_err_name_rm.assign(file_name);
							
							vec G_out;
							G_out=K*A_opt.rows(ind0,Nw-2);
							G_V_out=KG_V*A_opt.rows(ind0,Nw-2);
							
							if (NM>0)
							{
								M_out=KM*A_opt.rows(ind0,Nw-2);
								M_V_out=KM_V*A_opt.rows(ind0,Nw-2);
								errM_out=(M-M_out)/errM;
							}
							
							if (show_optimal_alpha_figs)
							{
								if (NM>1)
								{
									char ttl_M[]="normalized deviation of moments at optimal $\\\\alpha$";
									char xl_M[]="moment order", yl_M[]="$(M_{in}-M_{out})/\\\\sigma_M$";
									char attrM[]="'o', color='m', markeredgecolor='m'";
									gM[1]->add_title(ttl_M);
									plot(*gM[1], M_ord, errM_out, xl_M, yl_M, attrM);
								}
								else if (NM>0)
								{
									cout<<"normalized deviation of moment of order "<<M_ord(0)<<" at optimal alpha: "<<errM_out(0)<<endl;
								}
							}
							
							if (!boson || col_Gi>0)
							{
								if (cov_diag)
								{
									uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
									errRe=G_V.rows(even_ind)-G_V_out.rows(even_ind);
									errIm=G_V.rows(even_ind+1)-G_V_out.rows(even_ind+1);
								}
								else
								{
									errRe=G_V-G_V_out;
									//errRe=G_V.rows(0,Nn-1)-G_V_out.rows(0,Nn-1);
									//errIm=G_V.rows(Nn,2*Nn-1)-G_V_out.rows(Nn,2*Nn-1);
								}
							}
							else
								errRe=G_V-G_V_out;
							
							if (!boson || col_Gi>0)
							{
								if (cov_diag)
								{
									CRe.zeros(NnC);
									CIm.zeros(NnC);
									for (j=0; j<NnC; j++)
									{
										for (l=0; l<Nn-j; l++)
										{
											CRe(j)=CRe(j)+errRe(l)*errRe(l+j);
											CIm(j)=CIm(j)+errIm(l)*errIm(l+j);
										}
										CRe(j)=CRe(j)/(Nn-j);
										CIm(j)=CIm(j)/(Nn-j);
									}
								}
								else
								{
									CRe.zeros(NnC);
									for (j=0; j<NnC; j++)
									{
										for (l=0; l<2*Nn-j; l++)
										{
											CRe(j)=CRe(j)+errRe(l)*errRe(l+j);
										}
										CRe(j)=CRe(j)/(2*Nn-j);
									}
								}
							}
							else
							{
								CRe.zeros(NnC);
								for (j=0; j<NnC; j++)
								{
									for (l=0; l<Nn-j; l++)
										CRe(j)=CRe(j)+errRe(l)*errRe(l+j);
									CRe(j)=CRe(j)/(Nn-j);
								}
							}
							
							if (output_G_opt_rm.size())
								remove(output_G_opt_rm.c_str());
							mat M_save;
							if (!boson || col_Gi>0)
							{
								M_save.zeros(Nn,3);
								uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
								M_save.col(0)=wn;
								M_save.col(1)=G_out.rows(even_ind);
								M_save.col(2)=G_out.rows(even_ind+1);
							}
							else
							{
								M_save.zeros(Nn,2);
								M_save.col(0)=wn;
								M_save.col(1)=G_out;
							}
							sprintf(file_name,output_G_opt_format.c_str(),tem,alpha_vec(ind_alpha_opt));
//							cout<<file_name<<endl;
							M_save.save(file_name,raw_ascii);
							output_G_opt_rm.assign(file_name);
							
							if (error_G_opt_rm.size())
								remove(error_G_opt_rm.c_str());
							if (!boson || col_Gi>0)
							{
								if (cov_diag)
								{
									M_save.zeros(Nn,3);
									M_save.col(0)=wn;
									M_save.col(1)=errRe;
									M_save.col(2)=errIm;
								}
								else
								{
									M_save.zeros(2*Nn,2);
									M_save.col(0)=eigv_ind;
									M_save.col(1)=errRe;
								}
							}
							else
							{
								M_save.zeros(Nn,2);
								M_save.col(0)=eigv_ind;
								M_save.col(1)=errRe;
							}
							sprintf(file_name,error_G_opt_format.c_str(),tem,alpha_vec(ind_alpha_opt));
//							cout<<file_name<<endl;
							M_save.save(file_name,raw_ascii);
							error_G_opt_rm.assign(file_name);
							
							if (auto_corr_error_G_opt_rm.size())
								remove(auto_corr_error_G_opt_rm.c_str());
							if (!boson || col_Gi>0)
							{
								if (cov_diag)
								{
									M_save.zeros(NnC,3);
									M_save.col(0)=Dn;
									M_save.col(1)=CRe;
									M_save.col(2)=CIm;
								}
								else
								{
									M_save.zeros(NnC,2);
									M_save.col(0)=Dn;
									M_save.col(1)=CRe;
								}
							}
							else
							{
								M_save.zeros(NnC,2);
								M_save.col(0)=Dn;
								M_save.col(1)=CRe;
							}
							sprintf(file_name,auto_corr_error_G_opt_format.c_str(),tem,alpha_vec(ind_alpha_opt));
//							cout<<file_name<<endl;
							M_save.save(file_name,raw_ascii);
							auto_corr_error_G_opt_rm.assign(file_name);
							
							if (NM>0)
							{
								if (output_moments_opt_rm.size())
									remove(output_moments_opt_rm.c_str());
								sprintf(file_name,output_moments_opt_format.c_str(),tem,alpha_vec(ind_alpha_opt));
//								cout<<file_name<<endl;
								M_save.zeros(NM,2);
								M_save.col(0)=M;
								M_save.col(1)=M_out;
								M_save.save(file_name,raw_ascii);
								output_moments_opt_rm.assign(file_name);
							}
							
							if (show_optimal_alpha_figs)
							{
								char xl1[50], xl3[50], yl1[50], yl2[50], yl3[50], yl4[50];
								char ttl1[100], ttl2[100], ttl3[100], ttl4[100];
								if (cov_diag)
								{
									strcpy(xl1,"$\\\\omega_n$");
									strcpy(xl3,"frequency index difference");
									strcpy(yl1,"$\\\\Delta \\\\tilde{G}_{Re}$");
									strcpy(yl2,"$\\\\Delta \\\\tilde{G}_{Im}$");
									strcpy(yl3,"$AC[\\\\Delta \\\\tilde{G}_{Re}]$");
									strcpy(yl4,"$AC[\\\\Delta \\\\tilde{G}_{Im}]$");
									strcpy(ttl1,"normalized deviation of $Re[G]$ at optimal $\\\\alpha$");
									strcpy(ttl2,"normalized deviation of $Im[G]$ at optimal $\\\\alpha$");
									strcpy(ttl3,"autocorrelation of $\\\\Delta \\\\tilde{G}_{Re}$ at optimal $\\\\alpha$");
									strcpy(ttl4,"autocorrelation of $\\\\Delta \\\\tilde{G}_{Im}$ at optimal $\\\\alpha$");
								}
								else
								{
									strcpy(xl1,"covar. eigen. index");
									strcpy(xl3,"covar. eigen. index difference");
									strcpy(yl1,"$\\\\Delta \\\\tilde{G}$");
									strcpy(yl3,"$AC[\\\\Delta \\\\tilde{G}]$");
									strcpy(ttl1,"normalized deviation of $G$ at optimal $\\\\alpha$");
									strcpy(ttl3,"autocorrelation of $\\\\Delta \\\\tilde{G}$ at optimal $\\\\alpha$");
								}
								char attr1[]="'o-', color='b', markeredgecolor='b'", attr2[]="'o-', color='r', markeredgecolor='r'";
								
								if (cov_diag)
								{
									gv[ind_fig]->add_title(ttl1);
									plot(*gv[ind_fig], wn, errRe, xl1, yl1, attr1);
									ind_fig++;
									gv[ind_fig]->add_title(ttl3);
									plot(*gv[ind_fig], Dn, CRe, xl3, yl3, attr1);
									ind_fig++;
									if (col_Gi>0)
									{
										gv[ind_fig]->add_title(ttl2);
										plot(*gv[ind_fig], wn, errIm, xl1, yl2, attr2);
										ind_fig++;
										gv[ind_fig]->add_title(ttl4);
										plot(*gv[ind_fig], Dn, CIm, xl3, yl4, attr2);
										ind_fig++;
									}
								}
								else
								{
									gv[ind_fig]->add_title(ttl1);
									plot(*gv[ind_fig], eigv_ind, errRe, xl1, yl1, attr1);
									ind_fig++;
									gv[ind_fig]->add_title(ttl3);
									plot(*gv[ind_fig], Dn, CRe, xl3, yl3, attr1);
									ind_fig++;
								}
								
								double Lt_dw_min=1;
								
								vec Dw=w.rows(1,Nw-1)-w.rows(0,Nw-2);
								double dw_min=Dw.min();
								
								xlims[0]=0;
								xlims[1]=Lt_dw_min/dw_min;
								
						//		xlims[1]=t_re(t_re.n_rows-1)/5;
								
								vec Gr_t_re=real(G_t_re);
								vec Gi_t_re=imag(G_t_re);
								
								char title_Gt[]="Retarded Green function in time";
								char attrGrt[]="'b-'";
								char attrGit[]="'r-'";
								char lgdGrt[]="Re$[G^R(t)]$";
								char lgdGit[]="Im$[G^R(t)]$";
								char xlGt[]="$t$";
								
								gv[ind_fig]->add_data(t_re.memptr(),Gr_t_re.memptr(),Nw_dense);
								gv[ind_fig]->add_attribute(attrGrt);
								gv[ind_fig]->add_to_legend(lgdGrt);
								gv[ind_fig]->add_data(t_re.memptr(),Gi_t_re.memptr(),Nw_dense);
								gv[ind_fig]->add_attribute(attrGit);
								gv[ind_fig]->add_to_legend(lgdGit);
								gv[ind_fig]->set_axes_labels(xlGt,NULL);
								gv[ind_fig]->set_axes_lims(xlims,NULL);
								gv[ind_fig]->add_title(title_Gt);
								gv[ind_fig]->curve_plot();
								ind_fig++;
								
								xlims[0]=SC-3*SW;
								xlims[1]=SC+3*SW;
								
								if (xlims[0]<w_out(0)) xlims[0]=w_out(0);
								if (xlims[1]>w_out(Nw_out-1)) xlims[1]=w_out(Nw_out-1);
						
								char title_G_Re_w[]="Retarded Green function in frequency";
								char attrG1[]="'-',color='b'";
								char attrG2[]="'-',color='r'";
							//	char attrG3[]="'-',color='m'";
							//	char attrG4[]="'-',color='c'";
								char lgdG1[]="Re$[G^R(\\omega)]$";
								char lgdG2[]="Im$[G^R(\\omega)]$";
							//	char lgdG3[]="Re$[G_{KK}(\\omega)]$";
							//	char lgdG4[]="Im$[G_{FFT}(\\omega)]$";
								char xlG[]="$\\omega$";
								
							/*
								double G_Re_w_max=abs(Gr_Re_w).max();
							//	vec Gr_Re_w_inf=1.0/(w_dense);
								
								xlims[0]=w(0);
								xlims[1]=w(Nw-1);
							*/
								
								gv[ind_fig]->add_data(w_out.memptr(), Gr_Re_w.memptr(), Nw_out);
								gv[ind_fig]->add_attribute(attrG1);
								gv[ind_fig]->add_to_legend(lgdG1);
							//	gv[ind_fig]->add_data(w_dense.memptr(), Gr_Re_w_KK.memptr(), Nw_dense);
							//	gv[ind_fig]->add_attribute(attrG3);
							//	gv[ind_fig]->add_to_legend(lgdG3);
								gv[ind_fig]->add_data(w_out.memptr(), Gi_Re_w.memptr(), Nw_out);
								gv[ind_fig]->add_attribute(attrG2);
								gv[ind_fig]->add_to_legend(lgdG2);
							//	gv[ind_fig]->add_data(w_dense.memptr(), Gi_Re_w_FFT.memptr(), Nw_dense);
							//	gv[ind_fig]->add_attribute(attrG4);
							//	gv[ind_fig]->add_to_legend(lgdG4);
								gv[ind_fig]->set_axes_labels(xlG,NULL);
								gv[ind_fig]->set_axes_lims(xlims,NULL);
								gv[ind_fig]->add_title(title_G_Re_w);
								gv[ind_fig]->curve_plot();
								ind_fig++;
								
								if (col_Gi>0)
								{
									xlims[0]=SC-2*SW;
									xlims[1]=SC+2*SW;
								}
								else
								{
									xlims[0]=0;
									xlims[1]=2*SW;
								}
								
								char title_opt[]="Spectrum at optimal $\\\\alpha$";
								char lgd_opt[100], lgd_opt_r[100], lgd_opt_l[100], lgd_DM[]="default model", lgd_ref[]="reference spectrum";
								char lgd_Pade[]="Pade";
								sprintf(lgd_opt,"$\\\\alpha_{opt}=%4.2e$",alpha_vec(ind_alpha_opt));
								sprintf(lgd_opt_r,"$\\\\alpha=%4.2e$",alpha_vec(ind_alpha_opt_r));
								sprintf(lgd_opt_l,"$\\\\alpha=%4.2e$",alpha_vec(ind_alpha_opt_l));
								
								vec A_all=join_vert(DM,A_opt);
								
								gv[ind_fig]->add_data(wA.memptr(),DM.memptr(),NwA);
								gv[ind_fig]->add_to_legend(lgd_DM);
								if (A_ref_file.size() && A_ref.n_rows)
								{
									gv[ind_fig]->add_data(w_ref.memptr(),A_ref.memptr(),w_ref.n_rows);
									gv[ind_fig]->add_to_legend(lgd_ref);
									A_all=join_vert(A_all,A_ref);
								}
								if (compute_Pade && GR_Pade.n_rows)
								{
									vec imG_Pade=-2*imag(GR_Pade);
									gv[ind_fig]->add_data(w_out.memptr(),imG_Pade.memptr(),Nw_out);
									gv[ind_fig]->add_to_legend(lgd_Pade);
									A_all=join_vert(A_all,imG_Pade);
								}
								if (A_opt_l.n_rows)
								{
									gv[ind_fig]->add_data(w.memptr(),A_opt_l.memptr(),Nw);
									gv[ind_fig]->add_to_legend(lgd_opt_l);
									A_all=join_vert(A_all,A_opt_l);
								}
								if (A_opt_r.n_rows)
								{
									gv[ind_fig]->add_data(w.memptr(),A_opt_r.memptr(),Nw);
									gv[ind_fig]->add_to_legend(lgd_opt_r);
									A_all=join_vert(A_all,A_opt_r);
								}
								ylims[0]=0;
								ylims[1]=1.1*A_all.max();
								
								gv[ind_fig]->add_data(w.memptr(),A_opt.memptr(),Nw);
								gv[ind_fig]->add_to_legend(lgd_opt);
								gv[ind_fig]->set_axes_labels(xl_w,NULL);
								gv[ind_fig]->set_axes_lims(xlims,ylims);
								gv[ind_fig]->add_title(title_opt);
								gv[ind_fig]->curve_plot();
								ind_fig++;
								
								if (compute_P_alpha_G)
								{
									char title_cl_Br[]="Spectra from classic and Bryan's methods";
									char lgd_cl[100];
									char lgd_Br[]="Bryan spectrum";
									sprintf(lgd_cl,"$\\\\alpha_{cl}=%4.2e$",alpha_vec(ind_P_alpha_G_max));
									
									
									A_all=DM;
									
									gv[ind_fig]->add_data(wA.memptr(),DM.memptr(),NwA);
									gv[ind_fig]->add_to_legend(lgd_DM);
									if (A_ref_file.size() && A_ref.n_rows)
									{
										gv[ind_fig]->add_data(w_ref.memptr(),A_ref.memptr(),w_ref.n_rows);
										gv[ind_fig]->add_to_legend(lgd_ref);
										A_all=join_vert(A_all,A_ref);
									}
									
									gv[ind_fig]->add_data(w.memptr(),Acl.memptr(),Nw);
									gv[ind_fig]->add_to_legend(lgd_cl);
									A_all=join_vert(A_all,Acl);
									
									gv[ind_fig]->add_data(w.memptr(),Abr.memptr(),Nw);
									gv[ind_fig]->add_to_legend(lgd_Br);
									A_all=join_vert(A_all,Abr);
									
									ylims[0]=0;
									ylims[1]=1.1*A_all.max();
									
									gv[ind_fig]->set_axes_labels(xl_w,NULL);
									gv[ind_fig]->set_axes_lims(xlims,ylims);
									gv[ind_fig]->add_title(title_cl_Br);
									gv[ind_fig]->curve_plot();
									ind_fig++;
								}
							}
						}
						success=0;
					}
					else if (alpha<=alpha_min && !alpha_min_in.size())
					{
						cout<<"minimum value of alpha reached but optimal spectrum has not been found. Reducing alpha_min by a factor "<<f_alpha_min<<".\n";
						alpha_min=alpha_min/f_alpha_min;
						alpha_min_too_high=true;
						cout<<"new value of alpha_min: "<<alpha_min<<endl;
					}
					else
					{
						cout<<"optimal spectrum has not been found. The real frequency grid might not be adapted to the spectrum\n";
					}
					
					if (alpha<=alpha_min && dlchi2_lalpha_min_av/dlchi2_lalpha_max>RMAX_dlchi2_lalpha && !alpha_min_in.size())
					{
						cout<<"The minimum value of alpha may not be small enough. Reducing alpha_min by a factor "<<f_alpha_min<<".\n";
						alpha_min=alpha_min/f_alpha_min;
						alpha_min_too_high=true;
						cout<<"new value of alpha_min: "<<alpha_min<<endl;
					}
					
					if (show_lowest_alpha_figs || show_optimal_alpha_figs || show_alpha_curves)
					{
						if (graph_2D::display_figures) cout<<"close the figures to resume or stop execution\n";
						graph_2D::show_figures();
					}
					
					if (NM>1) for (j=0; j<2; j++) delete gM[j];
					for (j=0; j<Ngraph; j++) delete gv[j];
					
				}
			}
			
			if (interactive_mode)
			{
				cin.clear();
				if (!execute_maxent) cout<<"Note that \"preprocess only\", in section PREPROCESSING EXECUTION OPTIONS, is set to 'yes'. Disable that option if you want to continue with the actual calculation.\n";
				
			//	if (N_params_noise && !error_provided && ind_noise<(N_params_noise-1))
			//		cout<<"continue with current relative error ([y]/n)?:\n";
			//	else
				
				cout<<"continue execution? ([y]/n):\n";
					
				cin.get(continue_exec);
				if (continue_exec!='\n')
				{
					cin.clear();
					getline(cin,buf);
				}
				
				if (!(continue_exec=='y' || continue_exec=='\n')) remove_files();
				
				if (execute_maxent && N_params_noise && !error_provided && !(continue_exec=='y' || continue_exec=='\n') && ind_noise<(N_params_noise-1))
				{
					cout<<"proceed to the next added noise magnitude? ([y]/n):\n";
					cin.get(continue_exec);
					if (continue_exec!='\n')
					{
						cin.clear();
						getline(cin,buf);
					}
					ind_noise++;
					initialize=true;
				}
			}
			else if (N_params_noise && !error_provided)
			{
				ind_noise++;
				initialize=true;
				remove_files();
			}
			else
			{
				remove_files();
			}
		}
		else
		{
			continue_exec='n';
			remove_files();
		}
	
	} while (((continue_exec=='y' || continue_exec=='\n') && interactive_mode) || (!interactive_mode && N_params_noise && !error_provided && ind_noise<N_params_noise));
	
	
	/*
	int j;
	char file_name[200];
	if (ind_alpha_vec>0 && save_spec_func)
	{
		//cout<<"alpha_save_max: "<<alpha_save_max<<endl;
		//cout<<"alpha_save_min: "<<alpha_save_min<<endl;
		j=0;
		while (alpha_vec(j)>alpha_save_max && j<ind_alpha_vec-1)
		{
			alpha=alpha_vec(j);
			sprintf(file_name,output_name_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_G_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_error_format.c_str(),tem,alpha);
			remove(file_name);
			if (NM>0)
			{
				sprintf(file_name,output_moments_format.c_str(),tem,alpha);
				remove(file_name);
			}
			j++;
		}
		while (alpha_vec(j)>=alpha_save_min && j<ind_alpha_vec-1) j++;
		while (j<=ind_alpha_vec-1)
		{
			alpha=alpha_vec(j);
			sprintf(file_name,output_name_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_G_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_error_format.c_str(),tem,alpha);
			remove(file_name);
			if (NM>0)
			{
				sprintf(file_name,output_moments_format.c_str(),tem,alpha);
				remove(file_name);
			}
			j++;
		}
	}
	 */
	
	return success;
}

void OmegaMaxEnt_data::remove_files()
{
	int j;
	char file_name[200];
	if (ind_alpha_vec>0 && save_spec_func)
	{
		j=0;
		while (alpha_vec(j)>alpha_save_max && j<ind_alpha_vec-1)
		{
			alpha=alpha_vec(j);
			sprintf(file_name,output_name_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_G_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_error_format.c_str(),tem,alpha);
			remove(file_name);
			if (NM>0)
			{
				sprintf(file_name,output_moments_format.c_str(),tem,alpha);
				remove(file_name);
			}
			j++;
		}
		while (alpha_vec(j)>=alpha_save_min && j<ind_alpha_vec-1) j++;
		while (j<ind_alpha_vec-1)
		{
			alpha=alpha_vec(j);
			sprintf(file_name,output_name_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_G_format.c_str(),tem,alpha);
			remove(file_name);
			sprintf(file_name,output_error_format.c_str(),tem,alpha);
			remove(file_name);
			if (NM>0)
			{
				sprintf(file_name,output_moments_format.c_str(),tem,alpha);
				remove(file_name);
			}
			j++;
		}
	}
}

bool OmegaMaxEnt_data::preproc()
{
	initialize_maxent=true;
	initialize=false;
	bool init_spectrum_exists=false, file_grid_set=false, param_grid_set=false;
	wc_exists=false;
	w_exists=false;
	peak_exists=false;
	main_spectral_region_set=false;
	
	bool use_Riemann_integ=false;
	
	vec extr_w(2);
	
	if (!boson) maxM_default=3;
	else maxM_default=1;
	
	if (!maxM_in.size())
		maxM=maxM_default;
	
	int j;
	
	cout<<"\nPREPROCESSING\n\n";
	
	vec data_col1=abs(green_data.col(0));
	if (data_col1.min()==0 && !boson && !tau_GF)
	{
		cout<<"warning: there is a value equal to zero in the first column of your data file. If your data is bosonic, set parameter \"bosonic data\" to \"yes\" in subsection DATA PARAMETERS.\n";
		cout<<"If your data is a function of imaginary time, set \"imaginary time data\" to \"yes\".\n";
		//return false;
	}
	data_col1.reset();
	
	if (!boson)
	{
		ind0=1;
		if (!tau_GF)
		{
			cout<<"fermionic Green function given in Matsubara frequency\n";
			if (!set_G_omega_n_fermions())
			{
				cout<<"Green function definition failed\n";
				return false;
			}
			if (!set_covar_G_omega_n())
			{
				cout<<"covariance definition failed\n";
				return false;
			}
			if (!set_moments_fermions())
			{
				cout<<"moments definition failed\n";
				return false;
			}
			jfit=0;

//			cout<<"maxM: "<<maxM<<endl;
//			cout<<"moments_provided: "<<moments_provided<<endl;
//			cout<<"eval_moments: "<<eval_moments<<endl;
			if ( (!moments_provided && maxM>=0) || eval_moments )
			{
				if (!Ginf_finite || G_omega_inf_in.size())
				{
					if (!compute_moments_omega_n())
					{
						cout<<"Computation of moments failed. You need to either provide moments or grid parameters to allow the calculation to proceed.\n";
						return false;
					}
				}
				else
				{
					if (!compute_moments_omega_n_2())
					{
						cout<<"Computation of moments failed. You need to either provide moments or grid parameters to allow the calculation to proceed.\n";
						return false;
					}
				}
					
			}
			if (std_omega) cout<<"standard deviation of spectrum: "<<std_omega<<endl;
		}
		else
		{
			cout<<"fermionic Green function given in imaginary time\n";
			if (!M1_in.size() || !M2_in.size())
			{
				cout<<"For a Green function given in imaginary time, the first two moments are required to obtain the Matsubara frequency Green function.\n";
				cout<<"The program will try to extract those moments from the behavior of the Green function around tau=0 and tau=beta.\n";
			}
			
			tau=green_data.col(0);
			Ntau=tau.n_rows-1;
			Gtau=green_data.col(col_Gtau-1);
			if (Gtau.max()*Gtau.min()<0)
			{
				cout<<"warning: sign change in G(tau).\n";
	//			cout<<"error: G(tau) must not change sign.\n";
	//			return false;
			}
			if (Gtau(0)>0) Gtau=-Gtau;
			Gchi2=Gtau;
			
			cout<<"Number of imaginary time slices: "<<Ntau<<endl;
			
			if (displ_prep_figs)
			{
				graph_2D g1;
				
				char xl[]="$\\\\tau$", yl[]="$G(\\\\tau)$";
				plot(g1, tau, Gtau, xl, yl, NULL);
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			M0t=-Gtau(Ntau)-Gtau(0);
			if (M0_in.size())
			{
				M0=stod(M0_in);
				if (abs(M0t-M0)/M0>tol_norm)
					cout<<"warning: -G(0)-G(beta) not equal to provided norm.\n";
			}
			cout<<"-G(0)-G(beta): "<<M0t<<endl;
			
			if (!set_moments_fermions())
			{
				cout<<"moments definition failed\n";
				return false;
			}
			
			if (!tem_in.size())
			{
				tem=1.0/tau(Ntau);
			}
			else
			{
				if (abs(tem-1.0/tau(Ntau))>tol_tem)
					cout<<"warning: provided temperature does not match imaginary time in data file.\n";
				tau=linspace<vec>(0,Ntau,Ntau+1)/(Ntau*tem);
			}
			
			cout<<"temperature: "<<tem<<endl;
			
			if (!set_covar_Gtau())
			{
				cout<<"imaginary-time covariance matrix definition failed\n";
				return false;
			}
			
			if ( !moments_provided || eval_moments )
			{
				if (!compute_moments_tau_fermions())
				{
					cout<<"Computation of moments failed.\n";
					return false;
				}
			}
			if (std_omega) cout<<"standard deviation of spectrum: "<<std_omega<<endl;
			
			/*
			double Ntau_pow=log2(Ntau);
			if (!error_provided && floor(Ntau_pow)!=Ntau_pow )
			{
				set_Ntau_to_pow_2();
				Gchi2=Gtau;
				if (!set_covar_Gtau())
				{
					cout<<"imaginary-time covariance matrix definition failed\n";
					return false;
				}
			}
			 */
			
			if (!Fourier_transform_G_tau())
			{
				cout<<"error: unable to Fourier transform G(tau)\n";
				return false;
			}
			
		}
		
		if (maxM!=maxM_default)
		{
			if (maxM<M_ord(NM-1) && maxM>=0)
			{
				int j=NM-1;
				while (j>0 && M_ord(j)>maxM) j--;
				NM=j+1;
				M=M.rows(0,j);
				COVM=COVM.submat(0,0,NM-1,NM-1);
				if (covm_diag)	errM=errM.rows(0,NM-1);
				M_ord=M_ord.rows(0,NM-1);
				if (NM<3)
				{
					errM=sqrt(COVM.diag());
					covm_diag=true;
				}
			}
			else if (maxM<0)
			{
				M.reset();
				M_ord.reset();
				errM.reset();
				COVM.reset();
				NM=0;
				covm_diag=true;
			}
	//		cout<<"maxM!=maxM_default\n";
		}
		cout<<"number of moments imposed to the spectral function (including normalization): "<<NM<<endl;

		if (SW_set && SC_set)
		{
			wl=SC-SW/2;
			wr=SC+SW/2;
			main_spectral_region_set=true;
		}
		else if (SW_set && M1_set)
		{
			wl=M1-SW/2;
			wr=M1+SW/2;
			main_spectral_region_set=true;
		}
		else if (SC_set && std_omega)
		{
			wl=SC-std_omega;
			wr=SC+std_omega;
			main_spectral_region_set=true;
		}
		else if (std_omega && M1_set)
		{
			wl=M1-std_omega;
			wr=M1+std_omega;
			main_spectral_region_set=true;
		}
		else if (std_omega)
		{
			wl=-std_omega;
			wr=std_omega;
			main_spectral_region_set=true;
		}
		
		test_low_energy_peak_fermions();
		
		Du_constant=false;
		
		if (init_spectr_func_file.size())
		{
			init_spectrum_exists=set_initial_spectrum();
		}
		else if (grid_omega_file.size())
		{
			file_grid_set=set_grid_omega_from_file();
		}
		else if (use_grid_params && omega_grid_params.n_cols>2)
		{
			param_grid_set=set_grid_from_params();
		}
		
		if (!w_exists)
		{
			if (!wc_exists)
			{
				if (!set_wc())
				{
					cout<<"Real frequency grid definition failed.\n";
					return false;
				}
			}
			
			if (!set_omega_grid())
			{
				cout<<"Real frequency grid definition failed.\n";
				return false;
			}
			
		}
		
		j=0;
		while (abs(w_origin-w(j+1))<abs(w_origin-w(j)) && j<Nw-2) j++;
		int jc=j;
		
		cout<<"minimum frequency: "<<w(0)<<endl;
		cout<<"maximum frequency: "<<w(Nw-1)<<endl;
		cout<<"boundaries of main spectral range: "<<wl<<", "<<wr<<endl;
		cout<<"grid origin: "<<w_origin<<endl;
		cout<<"frequency step at the grid origin: "<<w(jc+1)-w(jc)<<endl;
		cout<<"number of real frequencies in the grid: "<<Nw<<endl;
		
		extr_w(0)=w(0);
		extr_w(1)=w(Nw-1);
		
		if (displ_prep_figs)
		{
			vec diff_w=w.rows(1,Nw-1)-w.rows(0,Nw-2);
			vec w_int=(w.rows(1,Nw-1)+w.rows(0,Nw-2))/2.0;
			graph_2D g1;
			char xl[]="$\\\\omega$";
			char yl[]="$1/(\\\\omega_{i+1}-\\\\omega_i)$";
			char title1[]="frequency grid density";
			char attr[]="marker='o', markerfacecolor='b', markeredgecolor='b'";
			g1.add_title(title1);
			plot(g1,w_int,1.0/diff_w,xl,yl,attr);
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		
	//	if (Nw<Nw_max)
	//	{
	//		cout<<"number of real frequencies in the grid: "<<Nw<<endl;
	//	}
	//	else
		if (Nw>Nw_max)
		{
		//	cout<<"number of frequencies Nw="<<Nw<<" larger than Nw_max="<<Nw_max<<endl;
			cout<<"number of real frequencies larger than maximum allowed Nw_max="<<Nw_max<<endl;
			cout<<"Use section \"FREQUENCY GRID PARAMETERS\" in parameter file to modify the number of real frequencies or increase parameter \"Nw_max\" in file \"OmegaMaxEnt_other_params.dat\".\n";
			return false;
		}
		
		if (cutoff_wn_in.size()) // && cutoff_wn<wn(Nn-1))
		{
		//	cout<<"cutoff_wn: "<<cutoff_wn<<endl;
			j=Nn-1;
			while (wn(j)>cutoff_wn && j>1) j--;
			ind_cutoff_wn=j;
		}
		else if (jfit)
		{
			ind_cutoff_wn=jfit;
			if ((ind_cutoff_wn+1)<Nn_min) ind_cutoff_wn=Nn_min-1;
		}
		else
			ind_cutoff_wn=Nn-1;
		
		G_all=G;
		n_all=n;
		wn_all=wn;
		Nn_all=Nn;
		if (ind_cutoff_wn<Nn-1 || ind_cutoff_wn+1>Nw)
		{
			if (!truncate_G_omega_n())
			{
				cout<<"warning: truncation of G failed.\n";
				return false;
			}
		}
		cout<<"Number of Matsubara frequencies used in chi2: "<<Nn<<endl;
		
		if (N_params_noise && !error_provided)
		{
			for (j=0; j<2*Nn; j++)
			{
				Gchi2(j)=Gchi2(j)+errG(j)*normal_distr(rnd_gen);
			}
		}
		
		if (!set_default_model())
		{
			cout<<"definition of default model failed\n";
			return false;
		}
		
		dwS=(w.rows(2,Nw-1)-w.rows(0,Nw-3))/2;
		
		if (gaussian_grid_density)
		{
			default_model.zeros(Nw);
			default_model.rows(1,Nw-2)=1/dwS;
		}
	
		if (!uniform_grid)
		{
			if (use_Riemann_integ) Kernel_G_fermions_Riemann_integ();
			else Kernel_G_fermions_grid_transf();
		}
		else
			Kernel_G_fermions_grid_transf_omega();

		mat norm_DM_tmp=KM.row(0)*default_model;
		default_model=M0*default_model/norm_DM_tmp(0);
		
		diagonalize_covariance();
		
	//	invDw=diagmat(1.0/dwS);
	//	KGMw=2*PI*KGM*invDw;
		//HKw=KGMw.t()*KGMw;
		
		default_model=exp(1)*(default_model.rows(ind0,Nw-2)+minDefM);
		
		vec A0_default=default_model/exp(1);
		if (!init_spectrum_exists) A0=A0_default;

		vec Pd=sqrt(4*PI*A0_default/dwS);
		mat P=diagmat(Pd);
		mat KGj=KG_V*P;
		mat U, V;
		vec sK, sK2;
		if (!svd(U,sK,V,KGj))
		{
			if (!svd(U,sK,V,KGj,"std"))
			{
				cout<<"preproc(): svd error\n";
				return false;
			}
		}
		sK2=pow(sK,2);
		
		alpha0_default=f_alpha_init*sK2.max();
		
		rowvec w0ASmin(2);
		rowvec s0ASmin;
		rowvec wgtASmin=ones<rowvec>(2);
		if (std_omega)
		{
			w0ASmin(0)=SC-std_omega;
			w0ASmin(1)=SC+std_omega;
			s0ASmin=R_width_ASmin*std_omega*ones<rowvec>(2);
		}
		else
		{
			w0ASmin(0)=SC-SW/2;
			w0ASmin(1)=SC+SW/2;
			s0ASmin=R_width_ASmin*(SW/2)*ones<rowvec>(2);
		}
		vec ASmin;
		sum_gaussians(w, w0ASmin, s0ASmin, wgtASmin, ASmin);
		ASmin=M0*ASmin.rows(1,Nw-2)+minDefM;
		
		double Smin=-sum(ASmin % dwS % log(ASmin/default_model))/(2*PI);
		
		if (Smin<0)
			alpha_min_default=2*Nn/(f_Smin*abs(Smin));
		else
			alpha_min_default=2*Nn/f_Smin;
		
		wA=w.rows(1,Nw-2);
		NwA=Nw-2;
		
		if (cov_diag) NnC=Nn/2;
		else NnC=Nn;
	
//		integrate_P_A_alpha();
		
		if (displ_adv_prep_figs)
		{
			graph_2D g1;
			
			char xl[]="$\\\\omega$";
			double xlims[2], ylims[2];
			char lgd1[]="default model", lgd2[]="minimum entropy spectrum";
			
			xlims[0]=SC-2*SW;
			xlims[1]=SC+2*SW;
			ylims[0]=0;
			ylims[1]=0;
			ylims[1]=1.1*ASmin.max();
			
			vec DM=default_model/exp(1);
			
			g1.add_to_legend(lgd1);
			g1.add_data(wA.memptr(),DM.memptr(),Nw-2);
			g1.add_to_legend(lgd2);
			g1.add_data(wA.memptr(),ASmin.memptr(),ASmin.n_rows);
			g1.set_axes_labels(xl,NULL);
			g1.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
	}
	else if (col_Gi>0)
	{
		ind0=1;
		if (!tau_GF)
		{
			cout<<"bosonic Green function given in Matsubara frequency\n";
			if (!set_G_omega_n_bosons())
			{
				cout<<"Green function definition failed\n";
				return false;
			}
			if (!set_covar_G_omega_n())
			{
				cout<<"covariance definition failed\n";
				return false;
			}
			if (!set_moments_bosons())
			{
				cout<<"moments definition failed\n";
				return false;
			}
			jfit=0;
			if ( (!moments_provided && maxM>0) || eval_moments )
			{
				if (!Ginf_finite || G_omega_inf_in.size())
				{
					if (!compute_moments_omega_n())
					{
						cout<<"Computation of moments failed. You need to either provide moments or grid parameters to allow the calculation to proceed.\n";
						return false;
					}
				}
				else
				{
					if (!compute_moments_omega_n_2())
					{
						cout<<"Computation of moments failed. You need to either provide moments or grid parameters to allow the calculation to proceed.\n";
						return false;
					}
				}
				/*
				if (!compute_moments_omega_n())
				{
					cout<<"computation of moments failed\n";
					return false;
				}
				 */
			}
			if (std_omega) cout<<"standard deviation of spectrum: "<<std_omega<<endl;
		}
		else
		{
			cout<<"bosonic Green function given in imaginary time\n";
			if (!M1_in.size() || !M2_in.size())
			{
				cout<<"For a Green function given in imaginary time, the first two moments are required to obtain the Matsubara frequency Green function.\n";
				cout<<"The program will try to extract those moments from the behavior of the Green function around tau=0 and tau=beta.\n";
			}
			
			tau=green_data.col(0);
			Ntau=tau.n_rows-1;
			Gtau=green_data.col(col_Gtau-1);
			if (Gtau.max()*Gtau.min()<0)
			{
				cout<<"warning: sign change in G(tau).\n";
		//		cout<<"error: G(tau) must not change sign.\n";
		//		return false;
			}
			if (Gtau(0)>0) Gtau=-Gtau;
			Gchi2=Gtau;
			
			if (!tem_in.size())
			{
				tem=1.0/tau(Ntau);
			}
			else
			{
				if (abs(tem-1.0/tau(Ntau))>tol_tem)
				cout<<"warning: provided temperature does not match imaginary time in data file.\n";
				tau=linspace<vec>(0,Ntau,Ntau+1)/(Ntau*tem);
			}
			
			cout<<"Number of imaginary time slices: "<<Ntau<<endl;
			cout<<"temperature: "<<tem<<endl;
			
			if (displ_prep_figs)
			{
				graph_2D g1;
				
				char xl[]="$\\\\tau$", yl[]="$G(\\\\tau)$";
				plot(g1, tau, Gtau, xl, yl, NULL);
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}

			M0t=Gtau(Ntau)-Gtau(0);
			if (M0_in.size())
			{
				M0=stod(M0_in);
				if (M0t)
				{
					if (abs(M0t-M0)/M0t>tol_norm)
						cout<<"warning: norm of spectral function is different from provided one.\n";
				}
			}
			cout<<"G(beta)-G(0): "<<M0t<<endl;
			
			M1n=abs(Gtau(0)/2+sum(Gtau.rows(1,Ntau-1))+Gtau(Ntau)/2)/(Ntau*tem);
			
			if (!set_moments_bosons())
			{
				cout<<"moments definition failed\n";
				return false;
			}
			
			if (!set_covar_Gtau())
			{
				cout<<"imaginary-time covariance matrix definition failed\n";
				return false;
			}

			if ( !moments_provided  || eval_moments )
			{
				if (!compute_moments_tau_bosons())
				{
					cout<<"Computation of moments failed.\n";
					return false;
				}
			}

			if (!Fourier_transform_G_tau())
			{
				cout<<"error: unable to Fourier transform G(tau)\n";
				return false;
			}

 			M1n=abs(Gr(0));
			if (!std_omega)
			{
				double var_omega=M1/M1n-pow(M0/M1n,2);
				std_omega=sqrt(var_omega);
			}
			
			if (!SC_set)
			{
				SC=M0/M1n;
				SC_set=true;
			}
			if (!SW_set)
			{
				SW=f_SW_std_omega*std_omega;
				SW_set=true;
			}
 			if (std_omega) cout<<"standard deviation of spectrum: "<<std_omega<<endl;

		}
		
		if (maxM!=maxM_default)
		{
			if (maxM<M_ord(NM-1) && maxM>=0)
			{
				int j=NM-1;
				while (j>0 && M_ord(j)>maxM) j--;
				NM=j+1;
				M=M.rows(0,j);
				COVM=COVM.submat(0,0,NM-1,NM-1);
				if (covm_diag)	errM=errM.rows(0,NM-1);
				M_ord=M_ord.rows(0,NM-1);
				if (NM<3)
				{
					errM=sqrt(COVM.diag());
					covm_diag=true;
				}
			}
			else if (maxM<0)
			{
				M.reset();
				M_ord.reset();
				errM.reset();
				COVM.reset();
				NM=0;
			}
		}
		cout<<"number of moments imposed to the spectral function: "<<NM<<endl;
		
		if (SW_set && SC_set)
		{
			wl=SC-SW/2;
			wr=SC+SW/2;
			main_spectral_region_set=true;
		}
		else if (SC_set && std_omega)
		{
			wl=SC-std_omega;
			wr=SC+std_omega;
			main_spectral_region_set=true;
		}
		else if (std_omega)
		{
			wl=-std_omega;
			wr=std_omega;
			main_spectral_region_set=true;
		}
		
		test_low_energy_peak_bosons();
		
		Du_constant=false;
		
		if (init_spectr_func_file.size())
		{
			init_spectrum_exists=set_initial_spectrum();
		}
		else if (grid_omega_file.size())
		{
			file_grid_set=set_grid_omega_from_file();
		}
		else if (use_grid_params && omega_grid_params.n_cols>2)
		{
			param_grid_set=set_grid_from_params();
		}
		
		if (!wc_exists)
		{
			if (!set_wc())
			{
				cout<<"Real frequency grid definition failed.\n";
				return false;
			}
		}
		
		if (!w_exists)
		{
			if (!set_omega_grid())
			{
				cout<<"Real frequency grid definition failed.\n";
				return false;
			}
		}
		
		j=0;
		while (abs(w_origin-w(j+1))<abs(w_origin-w(j)) && j<Nw-2) j=j+1;
		int jc=j;
		
		cout<<"minimum frequency: "<<w(0)<<endl;
		cout<<"maximum frequency: "<<w(Nw-1)<<endl;
		cout<<"boundaries of main spectral range: "<<wl<<", "<<wr<<endl;
		cout<<"frequency step at the grid origin: "<<w(jc+1)-w(jc)<<endl;
		cout<<"number of real frequencies in the grid: "<<Nw<<endl;
		
		extr_w(0)=w(0);
		extr_w(1)=w(Nw-1);
		
		if (displ_prep_figs)
		{
			vec diff_w=w.rows(1,Nw-1)-w.rows(0,Nw-2);
			vec w_int=(w.rows(1,Nw-1)+w.rows(0,Nw-2))/2.0;
			graph_2D g1;
			char xl[]="$\\\\omega$";
			char yl[]="$1/(\\\\omega_{i+1}-\\\\omega_i)$";
			char title1[]="frequency grid density";
			char attr[]="marker='o', markerfacecolor='b', markeredgecolor='b'";
			g1.add_title(title1);
			plot(g1,w_int,1/diff_w,xl,yl,attr);
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		
	//	if (Nw<Nw_max)
	//	{
	//		cout<<"number of real frequencies in the grid: "<<Nw<<endl;
	//	}
	//	else
		if (Nw>Nw_max)
		{
		//	cout<<"number of frequencies Nw="<<Nw<<" larger than Nw_max="<<Nw_max<<endl;
			cout<<"number of real frequencies larger than maximum allowed Nw_max="<<Nw_max<<endl;
			cout<<"Use section \"FREQUENCY GRID PARAMETERS\" in parameter file to modify the number of real frequencies or increase parameter \"Nw_max\" in file \"OmegaMaxEnt_other_params.dat\".\n";
			return false;
		}
		
		if (cutoff_wn_in.size()) // && cutoff_wn<wn(Nn-1))
		{
			j=Nn-1;
			while (wn(j)>cutoff_wn && j>1) j--;
			ind_cutoff_wn=j;
		}
		else if (jfit)
		{
			ind_cutoff_wn=jfit;
			if ((ind_cutoff_wn+1)<Nn_min) ind_cutoff_wn=Nn_min-1;
		}
		else
			ind_cutoff_wn=Nn-1;
		
		G_all=G;
		n_all=n;
		wn_all=wn;
		Nn_all=Nn;
		if (ind_cutoff_wn<Nn-1 || ind_cutoff_wn+1>Nw)
		{
			if (!truncate_G_omega_n())
			{
				cout<<"warning: truncation of G failed.\n";
				return false;
			}
		}
		cout<<"Number of Matsubara frequencies used in chi2: "<<Nn<<endl;
		
		if (N_params_noise && !error_provided)
		{
			for (j=0; j<2*Nn; j++)
			{
				Gchi2(j)=Gchi2(j)+errG(j)*normal_distr(rnd_gen);
			}
		}
		
		if (!set_default_model())
		{
			cout<<"definition of default model failed\n";
			return false;
		}
		
		Kernel_G_bosons();
		
		mat norm_DM_tmp=K.row(0)*default_model;
		default_model=M1n*default_model/abs(norm_DM_tmp(0));
		
		diagonalize_covariance();
		
		default_model=exp(1)*(default_model.rows(ind0,Nw-2)+minDefM);
		
		dwS=(w.rows(2,Nw-1)-w.rows(0,Nw-3))/2;
		
	//	invDw=diagmat(1.0/dwS);
	//	KGMw=2*PI*KGM*invDw;
	//	HKw=KGMw.t()*KGMw;
		
		vec A0_default=default_model/exp(1);
		if (!init_spectrum_exists) A0=A0_default;
		
		vec Pd=sqrt(4*PI*A0_default/dwS);
		mat P=diagmat(Pd);
		mat KGj=KG_V*P;
		mat U, V;
		vec sK, sK2;
		if (!svd(U,sK,V,KGj))
		{
			if (!svd(U,sK,V,KGj,"std"))
			{
				cout<<"preproc(): svd error\n";
				return false;
			}
		}
		sK2=pow(sK,2);
		
		alpha0_default=f_alpha_init*sK2.max();
		
		//vec Rchi2_S=(HK*default_model)/(2*dwS);
		//alpha0_default=f_alpha_init*max(abs(Rchi2_S));
		
		rowvec w0ASmin(2);
		rowvec s0ASmin;
		rowvec wgtASmin=ones<rowvec>(2);
		if (std_omega)
		{
			w0ASmin(0)=SC-std_omega;
			w0ASmin(1)=SC+std_omega;
			s0ASmin=R_width_ASmin*std_omega*ones<rowvec>(2);
		}
		else
		{
			w0ASmin(0)=SC-SW/2;
			w0ASmin(1)=SC+SW/2;
			s0ASmin=R_width_ASmin*(SW/2)*ones<rowvec>(2);
		}
		vec ASmin;
		sum_gaussians(w, w0ASmin, s0ASmin, wgtASmin, ASmin);
		ASmin=M1n*ASmin.rows(ind0,Nw-2)+minDefM;
		
		double Smin=-sum(ASmin % dwS % log(ASmin/default_model))/(2*PI);
		
		if (Smin<0)
			alpha_min_default=2*Nn/(f_Smin*abs(Smin));
		else
			alpha_min_default=2*Nn/f_Smin;
		
		wA=w.rows(1,Nw-2);
		NwA=Nw-2;
		
		if (cov_diag) NnC=Nn/2;
		else NnC=Nn;
		
		if (displ_adv_prep_figs)
		{
			graph_2D g1;
			
			char xl[]="$\\\\omega$";
			double xlims[2], ylims[2];
			char lgd1[]="default model", lgd2[]="minimum entropy spectrum";
			
			xlims[0]=SC-2*SW;
			xlims[1]=SC+2*SW;
			ylims[0]=0;
			ylims[1]=0;
			ylims[1]=1.1*ASmin.max();
			
			vec DM=default_model/exp(1);
			
			g1.add_to_legend(lgd1);
			g1.add_data(wA.memptr(),DM.memptr(),Nw-2);
			g1.add_to_legend(lgd2);
			g1.add_data(wA.memptr(),ASmin.memptr(),ASmin.n_rows);
			g1.set_axes_labels(xl,NULL);
			g1.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
	}
	else
	{
		w_origin=0;
		w_origin_set=true;
		ind0=0;
		if (!tau_GF)
		{
			cout<<"bosonic Green function given in Matsubara frequency\n";
			if (!set_G_omega_n_bosons())
			{
				cout<<"Green function definition failed\n";
				return false;
			}
			if (!set_covar_chi_omega_n())
			{
				cout<<"covariance definition failed\n";
				return false;
			}
			if (!set_moments_bosons())
			{
				cout<<"moments definition failed\n";
				return false;
			}
			
			jfit=0;
			if ( (!moments_provided && maxM>0) || eval_moments )
			{
				if (!compute_moments_chi_omega_n())
				{
					cout<<"Computation of moments failed. You need to either provide moments or grid parameters to allow the calculation to proceed.\n";
					return false;
				}
			}
			if (std_omega) cout<<"standard deviation of spectrum: "<<std_omega<<endl;
		}
		else
		{
			cout<<"bosonic Green function given in imaginary time\n";
			if (!M1_in.size() || !M2_in.size())
			{
				cout<<"For an even Green function given in imaginary time, the first moments is required to obtain the Matsubara frequency Green function.\n";
				cout<<"The program will try to extract the moments from the behavior of the Green function around tau=0 and tau=beta.\n";
			}
			
			tau=green_data.col(0);
			Ntau=tau.n_rows-1;
			Gtau=green_data.col(col_Gtau-1);
			if (Gtau.max()*Gtau.min()<0)
			{
				cout<<"warning: sign change in G(tau).\n";
		//		cout<<"error: G(tau) must not change sign.\n";
		//		return false;
			}
			if (Gtau(0)>0) Gtau=-Gtau;
			Gchi2=Gtau;
			
			if (!tem_in.size())
			{
				tem=1.0/tau(Ntau);
			}
			else
			{
				if (abs(tem-1.0/tau(Ntau))>tol_tem)
				cout<<"warning: provided temperature does not match imaginary time in data file.\n";
				tau=linspace<vec>(0,Ntau,Ntau+1)/(Ntau*tem);
			}
			
			cout<<"Number of imaginary time slices: "<<Ntau<<endl;
			cout<<"temperature: "<<tem<<endl;
			
			if (displ_prep_figs)
			{
				graph_2D g1;
				
				char xl[]="$\\\\tau$", yl[]="$G(\\\\tau)$";
				plot(g1, tau, Gtau, xl, yl, NULL);
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			M0t=Gtau(Ntau)-Gtau(0);
			if (M0_in.size())
			{
				M0=stod(M0_in);
				if (M0t)
				{
					if (abs(M0t-M0)/M0t>tol_norm)
						cout<<"warning: norm of spectral function is different from provided one.\n";
				}
			}
			cout<<"G(0)-G(beta): "<<M0t<<endl;
			
			M1n=abs(Gtau(0)/2+sum(Gtau.rows(1,Ntau-1))+Gtau(Ntau)/2)/(Ntau*tem);
			
			if (!set_moments_bosons())
			{
				cout<<"moments definition failed\n";
				return false;
			}
			
			if (!set_covar_Gtau())
			{
				cout<<"imaginary-time covariance matrix definition failed\n";
				return false;
			}
		
			if ( !moments_provided || eval_moments )
			{
				if (!compute_moments_tau_bosons())
				{
					cout<<"Computation of moments failed.\n";
					return false;
				}
			}
			
			if (!Fourier_transform_G_tau())
			{
				cout<<"error: unable to Fourier transform G(tau)\n";
				return false;
			}
			
			M1n=abs(Gr(0));
			
			if (!std_omega)
			{
				double var_omega=M1/M1n-pow(M0/M1n,2);
				std_omega=sqrt(var_omega);
			}
			
			if (!SC_set)
			{
				SC=0;
				SC_set=true;
			}
			if (!SW_set)
			{
				SW=f_SW_std_omega*std_omega;
				SW_set=true;
			}
			cout<<"standard deviation of spectrum: "<<std_omega<<endl;
		}
		
		if (maxM!=maxM_default)
		{
			if (maxM>0 && maxM<M_ord(NM-1))
			{
				int j=NM-1;
				while (j>0 && M_ord(j)>maxM) j--;
				NM=j+1;
				M=M.rows(0,j);
				COVM=COVM.submat(0,0,NM-1,NM-1);
				if (covm_diag)	errM=errM.rows(0,NM-1);
				M_ord=M_ord.rows(0,NM-1);
				if (NM<3)
				{
					errM=sqrt(COVM.diag());
					covm_diag=true;
				}
			}
			else if (maxM<=0)
			{
				M.reset();
				M_ord.reset();
				errM.reset();
				COVM.reset();
				NM=0;
			}
		}
		cout<<"number of moments imposed to the spectral function: "<<NM<<endl;
		
		if (SW_set)
		{
			wr=SW/2;
			main_spectral_region_set=true;
		}
		else if (std_omega)
		{
			wr=std_omega/2;
			SW=2*wr;
			main_spectral_region_set=true;
		}
		
		test_low_energy_peak_chi();
		
		Du_constant=false;
		
		if (init_spectr_func_file.size())
		{
			init_spectrum_exists=set_initial_spectrum_chi();
		}
		else if (grid_omega_file.size())
		{
			file_grid_set=set_grid_omega_from_file_chi();
		}
		else if (use_grid_params && omega_grid_params.n_cols>2)
		{
			param_grid_set=set_grid_from_params_chi();
		}
		
		if (!wc_exists)
		{
			if (!set_wc_chi())
			{
				cout<<"Real frequency grid definition failed.\n";
				return false;
			}
		}
		
		if (!w_exists)
		{
			if (!set_omega_grid_chi())
			{
				cout<<"Real frequency grid definition failed.\n";
				return false;
			}
		}
		
		cout<<"minimum frequency: "<<w(0)<<endl;
		cout<<"maximum frequency: "<<w(Nw-1)<<endl;
		cout<<"boundaries of main spectral range:  0, "<<wr<<endl;
		cout<<"frequency step at the grid origin: "<<w(1)-w(0)<<endl;
		cout<<"number of real frequencies in the grid: "<<Nw<<endl;
		
		extr_w(0)=-w(Nw-1);
		extr_w(1)=w(Nw-1);
		
		if (displ_prep_figs)
		{
			vec diff_w=w.rows(1,Nw-1)-w.rows(0,Nw-2);
			vec w_int=(w.rows(1,Nw-1)+w.rows(0,Nw-2))/2.0;
			graph_2D g1;
			char xl[]="$\\\\omega$";
			char yl[]="$1/(\\\\omega_{i+1}-\\\\omega_i)$";
			char title1[]="frequency grid density";
			char attr[]="marker='o', markerfacecolor='b', markeredgecolor='b'";
			g1.add_title(title1);
			plot(g1,w_int,1/diff_w,xl,yl,attr);
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		
	//	if (Nw<Nw_max)
	//	{
	//		cout<<"number of real frequencies in the grid: "<<Nw<<endl;
	//	}
	//	else
		if (Nw>Nw_max)
		{
		//	cout<<"number of frequencies Nw="<<Nw<<" larger than Nw_max="<<Nw_max<<endl;
			cout<<"number of real frequencies larger than maximum allowed Nw_max="<<Nw_max<<endl;
			cout<<"Use section \"FREQUENCY GRID PARAMETERS\" in parameter file to modify the number of real frequencies or increase parameter \"Nw_max\" in file \"OmegaMaxEnt_other_params.dat\".\n";
			return false;
		}
		
		if (cutoff_wn_in.size()) // && cutoff_wn<wn(Nn-1))
		{
			j=Nn-1;
			while (wn(j)>cutoff_wn && j>1) j--;
			ind_cutoff_wn=j;
		}
		else if (jfit)
		{
			ind_cutoff_wn=jfit;
			if ((ind_cutoff_wn+1)<Nn_min) ind_cutoff_wn=Nn_min-1;
		}
		else
			ind_cutoff_wn=Nn-1;
		
		G_all=G;
		n_all=n;
		wn_all=wn;
		Nn_all=Nn;
		if (ind_cutoff_wn<Nn-1 || ind_cutoff_wn+1>Nw)
		{
			if (!truncate_chi_omega_n())
			{
				cout<<"warning: truncation of G failed.\n";
				return false;
			}
		}
		cout<<"Number of Matsubara frequencies used in chi2: "<<Nn<<endl;
		
		if (N_params_noise && !error_provided)
		{
			for (j=0; j<Nn; j++)
			{
				Gchi2(j)=Gchi2(j)+errG(j)*normal_distr(rnd_gen);
			}
		}
		
		if (!set_default_model_chi())
		{
			cout<<"definition of default model failed\n";
			return false;
		}
		
		Kernel_chi();
		
		mat norm_DM_tmp=K.row(0)*default_model;
		default_model=M1n*default_model/abs(norm_DM_tmp(0));
		
		diagonalize_covariance_chi();
		
		default_model=exp(1)*(default_model.rows(ind0,Nw-2)+minDefM);
		
		dwS.zeros(Nw-1);
		dwS.rows(1,Nw-2)=w.rows(2,Nw-1)-w.rows(0,Nw-3);
		dwS(0)=w(1)-w(0);
		
	//	invDw=diagmat(1.0/dwS);
	//	KGMw=2*PI*KGM*invDw;
	//	HKw=KGMw.t()*KGMw;
		
		vec A0_default=default_model/exp(1);
		if (!init_spectrum_exists) A0=A0_default;
		
		vec Pd=sqrt(4*PI*A0_default/dwS);
		mat P=diagmat(Pd);
		mat KGj=KG_V*P;
		mat U, V;
		vec sK, sK2;
		if (!svd(U,sK,V,KGj))
		{
			if (!svd(U,sK,V,KGj,"std"))
			{
				cout<<"preproc(): svd error\n";
				return false;
			}
		}
		sK2=pow(sK,2);
		
		alpha0_default=f_alpha_init*sK2.max();
		
		//vec Rchi2_S=(HK*default_model)/(2*dwS);
		//alpha0_default=f_alpha_init*max(abs(Rchi2_S));

		rowvec w0ASmin(1);
		rowvec s0ASmin(1);
		rowvec wgtASmin(1);
		wgtASmin(0)=1;
		if (std_omega)
		{
			w0ASmin(0)=std_omega;
			s0ASmin(0)=R_width_ASmin*std_omega;
		}
		else
		{
			w0ASmin(0)=SW/2;
			s0ASmin(0)=R_width_ASmin*(SW/2);
		}
		vec ASmin;
		sum_gaussians_chi(w, w0ASmin, s0ASmin, wgtASmin, ASmin);
		ASmin=M1n*ASmin.rows(ind0,Nw-2)+minDefM;
		
		double Smin=-sum(ASmin % dwS % log(ASmin/default_model))/(2*PI);
		
		if (Smin<0)
			alpha_min_default=Nn/(f_Smin*abs(Smin));
		else
			alpha_min_default=Nn/f_Smin;
		
		wA=w.rows(ind0,Nw-2);
		NwA=Nw-1;
		
		if (cov_diag) NnC=Nn/2;
		else NnC=Nn;
		
		if (displ_adv_prep_figs)
		{
			graph_2D g1;
			
			char xl[]="$\\\\omega$";
			double xlims[2], ylims[2];
			char lgd1[]="default model", lgd2[]="minimum entropy spectrum";
			
			xlims[0]=0;
			xlims[1]=2*SW;
			ylims[0]=0;
			ylims[1]=0;
			ylims[1]=1.1*ASmin.max();
			
			vec DM=default_model/exp(1);
			
			g1.add_to_legend(lgd1);
			g1.add_data(wA.memptr(),DM.memptr(),Nw-2);
			g1.add_to_legend(lgd2);
			g1.add_data(wA.memptr(),ASmin.memptr(),ASmin.n_rows);
			g1.set_axes_labels(xl,NULL);
			g1.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
	}

	set_output_frequency_grid(extr_w);
	
	if (compute_Pade)
	{
		if (A_ref_file.size())
		{
			if (!set_A_ref()) cout<<"warning: reference spectrum could not be defined\n";
		}
		
		Pade_G_re_omega_name="Pade_Green_function";
		if (output_name_suffix.size())
		{
			if (output_name_suffix[0]!='_') Pade_G_re_omega_name+='_';
			Pade_G_re_omega_name+=output_name_suffix;
		}
		Pade_G_re_omega_name+=".dat";
		
		if (!N_Pade_in.size() || N_Pade>Nn) N_Pade=Nn;
		if (!eta_Pade_in.size()) eta_Pade=tem/100;
		
		compute_G_with_Pade(w_out, N_Pade, eta_Pade);
	}
	
	return true;
}

void OmegaMaxEnt_data::compute_G_with_Pade(vec wP, int NP, double eta)
{
	cout<<"computing real frequency Green function with Pade\n";
	
//	double tol_sv=1e-4;
	
	int j;
	int NwP=wP.n_rows;
	
//	cout<<"N_Pade: "<<N_Pade<<endl;
//	cout<<"eta: "<<eta<<endl;
//	cout<<"NwP: "<<NwP<<endl;
	
	cx_vec coeffs(NP,fill::zeros);
	cx_vec iwn(Nn,fill::zeros);
	cx_vec cx_wP(NwP,fill::zeros);
	vec eta_v=eta*ones<vec>(NwP);
	
	cx_wP.set_real(wP);
	cx_wP.set_imag(eta_v);
	
	iwn.set_imag(wn);
	
	pade_cont_frac_coef_rec(G.memptr(), iwn.memptr(), NP, coeffs.memptr());

//	cout<<"Pade coefficients computed\n";
	
	GR_Pade.zeros(NwP);
	
	for (j=0; j<NwP; j++)
		GR_Pade(j)=pade(cx_wP(j), NP, iwn.memptr(), coeffs.memptr());
	
//	cout<<"retarded Green function computed\n";
	
	vec G_Pade_re=real(GR_Pade);
	vec G_Pade_im=imag(GR_Pade);
	
	vec imG=-2*G_Pade_im;
	
	if (Ginf_finite) G_Pade_re=G_Pade_re+G_omega_inf;
	
	mat M_save(NwP,3);
	M_save.col(0)=wP;
	M_save.col(1)=G_Pade_re;
	M_save.col(2)=G_Pade_im;
	
	string file_name_str=Pade_G_re_omega_name;
	remove(file_name_str.c_str());
	M_save.save(file_name_str,raw_ascii);
	
/*
	vec A_pi(Nw,fill::zeros), sK;
	mat U, V;
	if (!svd(U,sK,V,K,"std"))
	{
		cout<<"compute_G_with_Pade(): svd of kernel failed\n";
	}
	
	int dim_wn=2*Nn;
	int dim_w=Nw-2;
	int ind0=1;
	int indf=Nw-2;
	if (col_Gi<=0)
	{
		dim_wn=Nn;
		dim_w=Nw-1;
		ind0=0;
	}
	
	mat invK(dim_w,dim_wn,fill::zeros);
	
	int dim_min=dim_wn;
	if (dim_w<dim_min) dim_min=dim_w;
	
//	cout<<setw(10)<<0<<sK(0)<<endl;
	
	invK(0,0)=1.0/sK(0);
	j=1;
	while (j<dim_min && sK(j)>tol_sv*sK(0))
	{
//		cout<<setw(10)<<j<<sK(j)<<endl;
		invK(j,j)=1.0/sK(j);
		j++;
	}
	
	A_pi.rows(ind0,indf)=V*invK*U.t()*Gchi2;
*/

	if (displ_prep_figs)
	{
		graph_2D g1, g2;
		char xl[]="$\\\\omega$";
		char lgdr[]="Re$[G(\\omega)]$";
		char lgdi[]="Im$[G(\\omega)]$";
		char lgdA[]="-2Im$[G(\\omega)]$";
		char attr_r[]="'b'";
		char attr_i[]="'r'";
		char title[]="Real frequency result from Pade";
		
		g1.add_data(wP.memptr(),G_Pade_re.memptr(),NwP);
		g1.add_attribute(attr_r);
		g1.add_to_legend(lgdr);
		g1.add_data(wP.memptr(),G_Pade_im.memptr(),NwP);
		g1.add_attribute(attr_i);
		g1.add_to_legend(lgdi);
		g1.set_axes_labels(xl,NULL);
		g1.add_title(title);
		g1.curve_plot();
		
		g2.add_data(wP.memptr(),imG.memptr(),NwP);
		g2.add_attribute(attr_i);
		g2.add_to_legend(lgdA);
//		g2.add_data(w.memptr(),A_pi.memptr(),Nw);
//		g2.add_attribute("'m'");
//		g2.add_to_legend("$A_{pinv}$");
		if (A_ref_file.size() && A_ref.n_rows)
		{
			g2.add_data(w_ref.memptr(),A_ref.memptr(),w_ref.n_rows);
			g2.add_attribute("'m'");
			g2.add_to_legend("reference spectrum");
		}
		g2.set_axes_labels(xl,NULL);
		g2.add_title("-2Im$[G(\\omega)]$ from Pade");
		g2.curve_plot();
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
}

void OmegaMaxEnt_data::compute_Re_chi_omega(vec Ap)
{
	cout<<"computing real part of the real-frequency correlation function...\n";
	
	bool compute_G_Re_w_KK=false;
	
	int j;
	
	vec coeffs;
	
	spline_chi_part(w, Nw_lims, ws, Ap, coeffs);

	int j0;
	j=0;
	while (j<Nw_dense && w_dense(j)<0) j++;
	j0=j;
	
	vec Gi_Re_w_tmp;
	spline_val_chi_part(w_dense.rows(j0,Nw_dense-1), w, Nw_lims, ws, coeffs, Gi_Re_w_tmp);
	Gi_Re_w_tmp=Gi_Re_w_tmp%w_dense.rows(j0,Nw_dense-1);
	
	Gi_Re_w.zeros(Nw_dense);
	Gi_Re_w.rows(j0,Nw_dense-1)=Gi_Re_w_tmp;
	Gi_Re_w.rows(0,j0-1)=-flipud(Gi_Re_w_tmp.rows(1,j0));
	
	cx_vec At;
	vec t;
	vec A_dense=2*Gi_Re_w;
	
	Fourier_transform_spectrum(w_dense, A_dense, t, At);
	
	t_re=t;
	G_t_re=dcomplex(0,1.0)*At;
	
	mat M_save(Nw_dense,3);
	M_save.col(0)=t_re;
	M_save.col(1)=real(G_t_re);
	M_save.col(2)=imag(G_t_re);
	
	string file_name_str=output_dir_fin;
	file_name_str+=G_re_t_name;
	remove(file_name_str.c_str());
	M_save.save(file_name_str,raw_ascii);
	
	cx_vec G_Re_w;
	
	compute_G_Re_omega_from_A_t(t, At, G_Re_w);
	Gr_Re_w=-real(G_Re_w);
	dG_w=-dG_w;
	Gi_Re_w_FFT=-imag(G_Re_w);
	
	double DwKK=R_wKK_SW*SW;
	
	int jKK_r;
	j=j0;
	while (j<Nw_dense && w_dense(j)<DwKK) j++;
	jKK_r=j;
	
	vec wKK=w_dense.rows(j0,jKK_r);
	
	int NwKK=jKK_r-j0+1;

	void *par[5];
	par[0]=&w;
	par[1]=&Nw_lims;
	par[2]=&ws;
	par[3]=&coeffs;
	
	fctPtr1 Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ_chi);
	
	double tol0=1e-4;
	double tol_r=1e-8;
	double tol_min=1e-10;
	double Rwdw=1e-10;
	vec tol;
	
	int Nint=4;
	vec lims(Nint+1);
	
	lims(0)=-w(Nw-1);
	lims(1)=-w(Nw_lims(0));
	lims(2)=0;
	lims(3)=w(Nw_lims(0));
	lims(4)=w(Nw-1);
	
	vec Gr_Re_w_tmp;
	
	Gr_Re_w_tmp.zeros(NwKK,1);
	tol=tol0*ones(NwKK,1);
	KK_integrate(wKK, Ptr, par, Rwdw, Gr_Re_w_tmp, tol, lims);
	tol=tol_r*abs(Gr_Re_w_tmp);
	for (j=0; j<NwKK; j++) if (tol(j)<tol_min) tol(j)=tol_min;
	KK_integrate(wKK, Ptr, par, Rwdw, Gr_Re_w_tmp, tol, lims);
	Gr_Re_w_tmp=-Gr_Re_w_tmp/PI;
	
	Gr_Re_w_KK.zeros(2*NwKK-1);
	Gr_Re_w_KK.rows(NwKK-1,2*NwKK-2)=Gr_Re_w_tmp;
	Gr_Re_w_KK.rows(0,NwKK-2)=flipud(Gr_Re_w_tmp.rows(1,NwKK-1));
	
	Gr_Re_w.rows(j0-NwKK+1,j0+NwKK-1)=Gr_Re_w_KK;
	
	if (Nw_out!=Nw_dense || w_out(0)!=w_dense(0) || w_out(Nw_out-1)!=w_dense(Nw_dense-1))
	{
		coeffs.zeros(4*(Nw_dense-1));
		spline_coeffs(w_dense.memptr(), Gi_Re_w.memptr(), Nw_dense, coeffs.memptr());
		spline_val(w_out, w_dense, coeffs, Gi_Re_w);
		
		coeffs.zeros();
		coeffs(0)=dG_w(0);
		coeffs(1)=dG_w(1);
		spline_coeffs(w_dense.memptr(), Gr_Re_w.memptr(), Nw_dense, coeffs.memptr());
		spline_val(w_out, w_dense, coeffs, Gr_Re_w);
	}
	
	if (Ginf_finite)
	{
		Gr_Re_w=Gr_Re_w+G_omega_inf;
	}
	
	M_save.zeros(Nw_out,3);
	M_save.col(0)=w_out;
	M_save.col(1)=Gr_Re_w;
	M_save.col(2)=Gi_Re_w;
	
	file_name_str=output_dir_fin;
	file_name_str+=G_re_omega_name;
	remove(file_name_str.c_str());
	M_save.save(file_name_str,raw_ascii);
	
	if (compute_G_Re_w_KK)
	{
	/*
		void *par[5];
		par[0]=&w;
		par[1]=&Nw_lims;
		par[2]=&ws;
		par[3]=&coeffs;
		
		fctPtr1 Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ_chi);
		
		double tol0=1e-4;
		double tol_r=1e-8;
		double tol_min=1e-8;
		double Rwdw=1e-10;
		vec tol;
		
		int Nint=4;
		vec lims(Nint+1);
		
		lims(0)=-w(Nw-1);
		lims(1)=-w(Nw_lims(0));
		lims(2)=0;
		lims(3)=w(Nw_lims(0));
		lims(4)=w(Nw-1);
		
		vec Gr_Re_w_tmp;
	*/
		Gr_Re_w_tmp.zeros(Nw_dense-j0,1);
		tol=tol0*ones(Nw_dense-j0,1);
		KK_integrate(w_dense.rows(j0,Nw_dense-1), Ptr, par, Rwdw, Gr_Re_w_tmp, tol, lims);
		//	KK_integrate_chi(w_dense, Ptr, par, Rwdw, Gr_Re_w, tol);
		tol=tol_r*abs(Gr_Re_w_tmp);
		for (j=0; j<Nw_dense-j0; j++) if (tol(j)<tol_min) tol(j)=tol_min;
		KK_integrate(w_dense.rows(j0,Nw_dense-1), Ptr, par, Rwdw, Gr_Re_w_tmp, tol, lims);
		//	KK_integrate_chi(w_dense, Ptr, par, Rwdw, Gr_Re_w, tol);
		Gr_Re_w_tmp=-Gr_Re_w_tmp/PI;
		
		Gr_Re_w_KK.zeros(Nw_dense);
		Gr_Re_w_KK.rows(j0,Nw_dense-1)=Gr_Re_w_tmp;
		Gr_Re_w_KK.rows(0,j0-1)=flipud(Gr_Re_w_tmp.rows(1,j0));
	}
	
	
	//	Gr_Re_w1.zeros(Nw_dense,1);
	//	spline_G_part(w, Nw_lims, ws, Gr_tmp, coeffs);
	//	spline_val_G_part(w_dense, w, Nw_lims, ws, coeffs, Gr_Re_w1);
	
/*
	double v=1e30;
	vec Gr_Re_w1=join_vert(flipud(Gr_Re_w.rows(1,Nw_dense-1)),Gr_Re_w);
	vec w_all=join_vert(-flipud(w_dense.rows(1,Nw_dense-1)),w_dense);
	int Nw_all=w_all.n_rows;
	vec dw=(w_all.rows(2,Nw_all-1)-w_all.rows(0,Nw_all-3))/2;
	vec dwj;
	Gi_Re_w1.zeros(Nw_dense);
//	cout<<"w_all: "<<w_all<<endl;
//	cout<<"dw: "<<dw<<endl;

	for (j=0; j<Nw_dense-1; j++)
	{
		dwj=w_dense(j)-w_all.rows(1,Nw_all-2);
		dwj(j+Nw_dense-2)=v;
		Gi_Re_w1(j)=sum((Gr_Re_w1.rows(1,Nw_all-2)%dw)/dwj)/PI;
//		cout<<"Gi_Re_w1(j): "<<Gi_Re_w1(j)<<endl;
	}
*/

/*
	//  use KK with the real part to verify that we recover the imaginary part
	Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ_lin_interp_chi);
	
	par[0]=&w_dense;
	par[1]=&Gr_Re_w;
	
	Gi_Re_w1.zeros(Nw_dense,1);
	tol=tol0*ones(Nw_dense,1);
	KK_integrate(w_dense, Ptr, par, Rwdw, Gi_Re_w1, tol, lims);
//	KK_integrate_chi(w_dense, Ptr, par, Rwdw, Gi_Re_w1, tol);
	tol=tol_r*abs(Gi_Re_w);
	for (j=0; j<Nw_dense; j++) if (tol(j)<tol_min) tol(j)=tol_min;
	KK_integrate(w_dense, Ptr, par, Rwdw, Gi_Re_w1, tol, lims);
//	KK_integrate_chi(w_dense, Ptr, par, Rwdw, Gi_Re_w1, tol);
	Gi_Re_w1=Gi_Re_w1/PI;
*/

}

void OmegaMaxEnt_data::compute_Re_G_omega(vec Ap)
{
	double tol_dw=1e-12;
	
	cout<<"computing real part of the retarded Green function...\n";
	
	bool compute_G_Re_w_KK=false;
	
	int j;
	
	vec coeffs;
	
	spline_G_part(w, Nw_lims, ws, Ap/2, coeffs);
	
	spline_val_G_part(w_dense, w, Nw_lims, ws, coeffs, Gi_Re_w);
	Gi_Re_w=-Gi_Re_w;
	if (boson) Gi_Re_w=w_dense%Gi_Re_w;
	
	cx_vec At;
	vec t;
	vec A_dense=-2*Gi_Re_w;
	
	Fourier_transform_spectrum(w_dense, A_dense, t, At);
	
	t_re=t;
	G_t_re=dcomplex(0,-1.0)*At;
	
	mat M_save(Nw_dense,3);
	M_save.col(0)=t_re;
	M_save.col(1)=real(G_t_re);
	M_save.col(2)=imag(G_t_re);
	
	string file_name_str=output_dir_fin;
	file_name_str+=G_re_t_name;
	remove(file_name_str.c_str());
	M_save.save(file_name_str,raw_ascii);
	
	cx_vec G_Re_w;
	
	compute_G_Re_omega_from_A_t(t, At, G_Re_w);
	Gr_Re_w=real(G_Re_w);
	Gi_Re_w_FFT=imag(G_Re_w);
	
// use Kramers-Kronig around omega=0
	double tol0=1e-4;
	double tol_r=1e-8;
	double tol_min=1e-10;
	double Rwdw=1e-10;
	
	double DwKK=R_wKK_SW*SW;
	
	int jKK_l, jKK_r;
	j=0;
	while (j<Nw_dense && w_dense(j)<-DwKK) j++;
	jKK_l=j-1;
	while (j<Nw_dense && w_dense(j)<DwKK) j++;
	jKK_r=j;
	
	vec wKK=w_dense.rows(jKK_l,jKK_r);
	
	int NwKK=jKK_r-jKK_l+1;
//	cout<<"w_KK_min, w_KK_max: "<<setw(20)<<w_dense(jKK_l)<<w_dense(jKK_r)<<endl;
//	cout<<"NwKK: "<<NwKK<<endl;
	
	void *par[5];
	fctPtr1 Ptr;
	
	par[0]=&w;
	par[1]=&Nw_lims;
	par[2]=&ws;
	par[3]=&coeffs;
	
	if (!boson)
		Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ);
	else if (col_Gi>0)
		Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ_boson);
	else
	{
		cout<<"compute_Re_G_omega(): wrong function! Use compute_Re_chi_omega()\n";
		return;
	}
	
	vec tol;
	
	int Nint=4;
	vec lims(Nint+1);
	
	lims(0)=w(0);
	lims(4)=w(Nw-1);
	if (w(Nw_lims(0))<0 && w(Nw_lims(1))>0)
	{
		lims(1)=w(Nw_lims(0));
		lims(2)=0;
		lims(3)=w(Nw_lims(1));
	}
	else if (w(Nw_lims(0))>0)
	{
		lims(1)=0;
		lims(2)=w(Nw_lims(0));
		lims(3)=w(Nw_lims(1));
	}
	else
	{
		lims(1)=w(Nw_lims(0));
		lims(2)=w(Nw_lims(1));
		lims(3)=0;
	}
	
	Gr_Re_w_KK.zeros(NwKK,1);
	tol=tol0*ones(NwKK,1);
	KK_integrate(wKK, Ptr, par, Rwdw, Gr_Re_w_KK, tol, lims);
	tol=tol_r*abs(Gr_Re_w_KK);
	for (j=0; j<NwKK; j++) if (tol(j)<tol_min) tol(j)=tol_min;
	
	KK_integrate(wKK, Ptr, par, Rwdw, Gr_Re_w_KK, tol, lims);
	Gr_Re_w_KK=Gr_Re_w_KK/PI;
	
	Gr_Re_w.rows(jKK_l,jKK_r)=Gr_Re_w_KK;
	
//	cout<<"real part of G computed\n";
	
	if (compute_G_Re_w_KK)
	{
	/*
		double tol0=1e-4;
		double tol_r=1e-8;
		double tol_min=1e-10;
		double Rwdw=1e-10;
		
		void *par[5];
		fctPtr1 Ptr;
		
		par[0]=&w;
		par[1]=&Nw_lims;
		par[2]=&ws;
		par[3]=&coeffs;
		
		if (!boson)
			Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ);
		else if (col_Gi>0)
			Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ_boson);
		else
		{
			cout<<"compute_Re_G_omega(): wrong function! Use compute_Re_chi_omega()\n";
			return;
		}
		
		vec tol;
		
		int Nint=4;
		vec lims(Nint+1);
	
		lims(0)=w(0);
		lims(4)=w(Nw-1);
		if (w(Nw_lims(0))<0 && w(Nw_lims(1))>0)
		{
			lims(1)=w(Nw_lims(0));
			lims(2)=0;
			lims(3)=w(Nw_lims(1));
		}
		else if (w(Nw_lims(0))>0)
		{
			lims(1)=0;
			lims(2)=w(Nw_lims(0));
			lims(3)=w(Nw_lims(1));
		}
		else
		{
			lims(1)=w(Nw_lims(0));
			lims(2)=w(Nw_lims(1));
			lims(3)=0;
		}
	*/
		Gr_Re_w_KK.zeros(Nw_dense,1);
		tol=tol0*ones(Nw_dense,1);
		KK_integrate(w_dense, Ptr, par, Rwdw, Gr_Re_w_KK, tol, lims);
		tol=tol_r*abs(Gr_Re_w_KK);
		for (j=0; j<Nw_dense; j++) if (tol(j)<tol_min) tol(j)=tol_min;
		KK_integrate(w_dense, Ptr, par, Rwdw, Gr_Re_w_KK, tol, lims);
		Gr_Re_w_KK=Gr_Re_w_KK/PI;
	}

	double dw=w_dense(1)-w_dense(0);
	if (Nw_out!=Nw_dense || fabs(w_out(0)-w_dense(0))>tol_dw*dw || fabs(w_out(Nw_out-1)-w_dense(Nw_dense-1))>tol_dw*dw)
	{
		coeffs.zeros(4*(Nw_dense-1));
		spline_coeffs(w_dense.memptr(), A_dense.memptr(), Nw_dense, coeffs.memptr());
		vec A_out;
		spline_val(w_out, w_dense, coeffs, A_out);
		Gi_Re_w=-A_out/2;
		if (boson) Gi_Re_w=w_out%Gi_Re_w;
		
		coeffs.zeros();
		coeffs(0)=dG_w(0);
		coeffs(1)=dG_w(1);
		spline_coeffs(w_dense.memptr(), Gr_Re_w.memptr(), Nw_dense, coeffs.memptr());
		spline_val(w_out, w_dense, coeffs, Gr_Re_w);
	}
	
	if (Ginf_finite)
	{
		Gr_Re_w=Gr_Re_w+G_omega_inf;
	}
	
	M_save.zeros(Nw_out,3);
	M_save.col(0)=w_out;
	M_save.col(1)=Gr_Re_w;
	M_save.col(2)=Gi_Re_w;
	
	file_name_str=output_dir_fin;
	file_name_str+=G_re_omega_name;
	remove(file_name_str.c_str());
	M_save.save(file_name_str,raw_ascii);
	
	cout<<"real part of the retarded Green function computed\n";
	
/*
	vec Atmp=-Gi_Re_w/PI;
	if (!Nw_dense%2)
	{
		Atmp.resize(Nw_dense+1);
		Atmp(Nw_dense)=0;
	}
	double intA=simpson_integ(Atmp,dw_dense);
	cout<<"int(A(w)): "<<intA<<endl;
*/
	
/*
//  use KK with the real part to verify that we recover the imaginary part
	void *par[5];
	fctPtr1 Ptr;
	int Nint=4;
	vec lims(Nint+1);
	vec tol;
	
	lims(0)=w(0);
	lims(4)=w(Nw-1);
	if (w(Nw_lims(0))<0 && w(Nw_lims(1))>0)
	{
		lims(1)=w(Nw_lims(0));
		lims(2)=0;
		lims(3)=w(Nw_lims(1));
	}
	else if (w(Nw_lims(0))>0)
	{
		lims(1)=0;
		lims(2)=w(Nw_lims(0));
		lims(3)=w(Nw_lims(1));
	}
	else
	{
		lims(1)=w(Nw_lims(0));
		lims(2)=w(Nw_lims(1));
		lims(3)=0;
	}
	
	Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::KK_integ_lin_interp);
	
	par[0]=&w_dense;
	par[1]=&Gr_Re_w;
	
	lims(0)=w_dense_min;
	lims(Nint)=w_dense_max;
	
	Gi_Re_w_KK.zeros(Nw_dense,1);
	tol=tol0*ones(Nw_dense,1);
	KK_integrate(w_dense, Ptr, par, Rwdw, Gi_Re_w_KK, tol, lims);
	tol=tol_r*abs(Gi_Re_w_KK);
	for (j=0; j<Nw_dense; j++) if (tol(j)<tol_min) tol(j)=tol_min;
	KK_integrate(w_dense, Ptr, par, Rwdw, Gi_Re_w_KK, tol, lims);
	Gi_Re_w_KK=Gi_Re_w_KK/PI;
 
//  add the contribution of the high frequency beyond the cutoff using G_hf=M0/(w-M1)
	cx_vec dGi_inf(Nw_dense,fill::zeros);
	dcomplex b, c, sq;
	for (j=0; j<Nw_dense; j++)
	{
		b=-M1-w_dense(j);
		c=M1*w_dense(j);
		sq=sqrt(4.0*c-b*b);
		dGi_inf(j)-=(2/PI)*(atan((2*w_dense_min+b)/sq)+PI/2)/sq;
		dGi_inf(j)-=(2/PI)*(PI/2-atan((2*w_dense_max+b)/sq))/sq;
	}
	vec Re_dGi=real(dGi_inf);
//	cout<<"Re_dGi: "<<Re_dGi<<endl;
//	cout<<"Gi_Re_w_KK: "<<Gi_Re_w_KK<<endl;
	Gi_Re_w_KK=Gi_Re_w_KK+M0*Re_dGi;
*/
	
}

void OmegaMaxEnt_data::set_output_frequency_grid(vec extr_w)
{
	double w_dense_min, w_dense_max, dw_dense, w_range, w_out_min, w_out_max, dw_out;
	vec Dw=w.rows(1,Nw-1)-w.rows(0,Nw-2);
	double dw_min=Dw.min()/R_dw_min_dw_dense;
	
	dw_dense=dw_min;
	w_range=R_SW_G_Re_w_range*SW;
	double w_range_tmp;
	if (output_grid_params_in.size() && output_grid_params(0)<0 && output_grid_params(2)>0 && output_grid_params(0)>extr_w(0) && output_grid_params(2)<extr_w(1))
	{
		w_range_tmp=-output_grid_params(0);
		if (output_grid_params(2)+output_grid_params(1)>w_range_tmp) w_range_tmp=output_grid_params(2);
	}
	if (w_range_tmp>w_range) w_range=w_range_tmp;
	double log2_Nw_dense=ceil(log2(w_range/dw_dense));
	int Nw_p=pow(2,log2_Nw_dense-1);
	vec w_dense_p=linspace<vec>(0,Nw_p,Nw_p+1)*dw_dense;
	vec w_dense_m=-linspace<vec>(1,Nw_p,Nw_p)*dw_dense;
	w_dense=join_vert(flipud(w_dense_m),w_dense_p);
	Nw_dense=2*Nw_p+1;
	w_dense_min=w_dense(0);
	w_dense_max=w_dense(Nw_dense-1);
	w_range=w_dense_max-w_dense_min;
	
	while (w_dense_min<extr_w(0) || w_dense_max>extr_w(1))
	{
		Nw_p=Nw_p/2;
		w_dense_p=w_dense_p.rows(0,Nw_p);
		w_dense_m=w_dense_m.rows(0,Nw_p-1);
		w_dense=join_vert(flipud(w_dense_m),w_dense_p);
		Nw_dense=2*Nw_p+1;
		w_dense_min=w_dense(0);
		w_dense_max=w_dense(Nw_dense-1);
		w_range=w_dense_max-w_dense_min;
	}
	
	if (output_grid_params_in.size() && output_grid_params(0)<0 && output_grid_params(2)>0 && output_grid_params(0)>extr_w(0) && output_grid_params(2)<extr_w(1))
	{
		w_out_min=output_grid_params(0);
		w_out_max=output_grid_params(2);
		dw_out=output_grid_params(1);
		
		w_range=w_out_max-w_out_min;
		if (dw_out>Dw.min())
		{
			cout<<"set_output_frequency_grid() warning: output frequency step might be too large to resolve the spectrum well.\n";
		}
	
//		Nw_p=(int)round(w_out_max/dw_out)+1;
//		int Nw_m=(int)round(-w_out_min/dw_out);
//		vec w_p=linspace<vec>(0,Nw_p-1,Nw_p)*dw_out;
//		vec w_m=-linspace<vec>(1,Nw_m,Nw_m)*dw_out;
//		w_out=join_vert(flipud(w_m),w_p);
//		Nw_out=Nw_m+Nw_p;
		
		Nw_out=(int)round(w_range/dw_out)+1;
		w_out=linspace<vec>(w_out_min,w_out_max,Nw_out);
		
//		Nw_out=(int)round(w_range/dw_out)+1;
//		w_out=linspace<vec>(0,Nw_out-1,Nw_out)*dw_out+w_out_min;
	}
	else
	{
		w_out=w_dense;
		Nw_out=Nw_dense;
	}
	
	cout<<"number of points in the output real frequency grid: "<<Nw_out<<endl;
	cout<<"frequency step of the output grid: "<<w_out(1)-w_out(0)<<endl;
	cout<<"minimum frequency of the output grid: "<<w_out(0)<<endl;
	cout<<"maximum frequency of the output grid: "<<w_out(Nw_out-1)<<endl;
	
//	cout<<"Nw_dense: "<<Nw_dense<<endl;
//	cout<<"w_out(0)-w_dense(0): "<<w_dense(0)-w_out(0)<<endl;
//	cout<<"w_dense(Nw_dense-1)-w_out(Nw_out-1): "<<w_dense(Nw_dense-1)-w_out(Nw_out-1)<<endl;
	
/*
	if (output_grid_params_in.size() && output_grid_params(0)<0 && output_grid_params(2)>0 && output_grid_params(0)>extr_w(0) && output_grid_params(2)<extr_w(1))
	{
		w_dense_min=output_grid_params(0);
		w_dense_max=output_grid_params(2);
		dw_dense=output_grid_params(1);
		
		w_range=w_dense_max-w_dense_min;
		if (dw_dense>2*dw_min)
		{
			cout<<"compute_Re_G_omega() warning: output frequency step might be too large to resolve the spectrum well.\n";
		//	dw_dense=dw_min;
		}
		
		int Nw_p, Nw_m;
		Nw_p=(int)floor(w_dense_max/dw_dense)+1;
		Nw_m=(int)floor(fabs(w_dense_min)/dw_dense);
		
		vec w_dense_p=linspace<vec>(0,Nw_p-1,Nw_p)*dw_dense;
		vec w_dense_m=-linspace<vec>(1,Nw_m,Nw_m)*dw_dense;
		w_dense=join_vert(flipud(w_dense_m),w_dense_p);
		Nw_dense=Nw_p+Nw_m;
	}
	else
	{
		if (output_grid_params_in.size())
		{
			cout<<"set_output_frequency_grid(): invalid output real frequency grid parameters.\n";
			if (output_grid_params(0)>0 || output_grid_params(2)<0) cout<<"omega=0 is not in the provided frequency range\n";
			if (output_grid_params(0)<extr_w(0) || output_grid_params(2)>extr_w(1)) cout<<"at least one extremum is outside the range where the spectrum is defined\n";
			cout<<"Using the default output real frequency grid.\n";
		}
		dw_dense=dw_min;
		w_range=R_SW_G_Re_w_range*SW;
		double log2_Nw_dense=ceil(log2(w_range/dw_dense));
		int Nw_p=pow(2,log2_Nw_dense-1);
		vec w_dense_p=linspace<vec>(0,Nw_p,Nw_p+1)*dw_dense;
		vec w_dense_m=-linspace<vec>(1,Nw_p,Nw_p)*dw_dense;
		w_dense=join_vert(flipud(w_dense_m),w_dense_p);
		Nw_dense=2*Nw_p+1;
		w_dense_min=w_dense(0);
		w_dense_max=w_dense(Nw_dense-1);
		w_range=w_dense_max-w_dense_min;
		
		while (w_dense_min<extr_w(0) || w_dense_max>extr_w(1))
		{
			Nw_p=Nw_p/2;
			w_dense_p=w_dense_p.rows(0,Nw_p);
			w_dense_m=w_dense_m.rows(0,Nw_p-1);
			w_dense=join_vert(flipud(w_dense_m),w_dense_p);
			Nw_dense=2*Nw_p+1;
			w_dense_min=w_dense(0);
			w_dense_max=w_dense(Nw_dense-1);
			w_range=w_dense_max-w_dense_min;
		}
	}
*/
	
//	cout<<"number of points in the output real frequency grid: "<<Nw_dense<<endl;
//	cout<<"frequency step of the output real frequency grid: "<<dw_dense<<endl;
//	cout<<"minimum frequency of the output real frequency grid: "<<w_dense_min<<endl;
//	cout<<"maximum frequency of the output real frequency grid: "<<w_dense_max<<endl;
}

void OmegaMaxEnt_data::compute_G_Re_omega_from_A_t(vec t, cx_vec At, cx_vec &G_Re_omega)
{
	cout<<"computing the Fourier transform of the real time Green function...\n";
	
	dcomplex I(0,1);
	
	int N_interv=t.n_rows-1;
	double dt=t(1);
	
	vec coeffs_r(4*N_interv,fill::zeros);
	vec coeffs_i(4*N_interv,fill::zeros);
	
	vec Ar=real(At);
	vec Ai=imag(At);
	
	coeffs_i(0)=-M1;
//	cout<<"coeffs_i(0): "<<coeffs_i(0)<<endl;
//	cout<<"dAi/dt: "<<(Ai(1)-Ai(0))/t(1)<<endl;
	
	spline_coeffs(t.memptr(), Ar.memptr(), N_interv+1, coeffs_r.memptr());
	spline_coeffs(t.memptr(), Ai.memptr(), N_interv+1, coeffs_i.memptr());
	
	uvec l=linspace<uvec>(0,N_interv-1,N_interv);
	
	cx_vec d3At(N_interv,fill::zeros);
	d3At.set_real(6*coeffs_r.rows(4*l));
	d3At.set_imag(6*coeffs_i.rows(4*l));
	
	double dw=w_dense(1)-w_dense(0);
	
	dcomplex d2At0(2*coeffs_r(1),2*coeffs_i(1));
	dcomplex d2Atmax(6*coeffs_r(4*N_interv-4)*dt+2*coeffs_r(4*N_interv-3),6*coeffs_i(4*N_interv-4)*dt+2*coeffs_i(4*N_interv-3));
	
	d3At=d3At%exp(I*w_dense(0)*t.rows(0,N_interv-1));
	
	cx_vec TF_d3At(N_interv+1,fill::zeros);
	TF_d3At.rows(0,N_interv-1)=((double)N_interv)*ifft(d3At);
	
	vec wG=w_dense;
	
	int j=0;
	while (j<Nw_dense && w_dense(j)<0) j++;
	int jw0=j;
	
	wG(jw0)=1;
	
	G_Re_omega.zeros(Nw_dense);
	
	G_Re_omega=At(0)/wG+M1/pow(wG,2)+(exp(I*wG*2.0*PI/dw)*d2Atmax-d2At0)/pow(wG,3)+I*TF_d3At%(exp(I*wG*dt)-1.0)/pow(wG,4);
	
	wG(jw0)=w_dense(jw0);
	
	G_Re_omega(jw0)=0;
	
	int m;
	double a,b,c,d, intr, inti;
	for (m=0; m<N_interv; m++)
	{
		a=coeffs_r(4*m);
		b=coeffs_r(4*m+1);
		c=coeffs_r(4*m+2);
		d=coeffs_r(4*m+3);
		intr=a*pow(dt,4)/4+b*pow(dt,3)/3+c*pow(dt,2)/2+d*dt;
		a=coeffs_i(4*m);
		b=coeffs_i(4*m+1);
		c=coeffs_i(4*m+2);
		d=coeffs_i(4*m+3);
		inti=a*pow(dt,4)/4+b*pow(dt,3)/3+c*pow(dt,2)/2+d*dt;
		G_Re_omega(jw0)+=dcomplex(inti,-intr);
	}

/*
	wG.zeros(Nw_dense+2);
	wG(0)=w_dense(0)-dw;
	wG(Nw_dense+1)=w_dense(Nw_dense-1)+dw;
	wG.rows(1,Nw_dense)=w_dense;
	w_dense=wG;
	
	vec Gw_tmp=zeros<vec>(Nw_dense+2);
	
	Gw_tmp(0)=At(0)/wG(0)+M1/pow(wG(0),2)+(exp(I*wG(0)*2.0*PI/dw)*d2Atmax-d2At0)/pow(wG(0),3);
	Gw_tmp(Nw_dense+1)=At(0)/wG(Nw_dense+1)+M1/pow(wG(Nw_dense+1),2)+(exp(I*wG(Nw_dense+1)*2.0*PI/dw)*d2Atmax-d2At0)/pow(wG(Nw_dense+1),3);
	Gw_tmp.rows(1,Nw_dense)=G_Re_omega;
	G_Re_omega=Gw_tmp;
	
	Nw_dense=Nw_dense+2;
*/
	
	dG_w.zeros(2);
	dG_w(0)=-real(At(0))/pow(wG(0),2)-2*M1/pow(wG(0),3)-3*real(exp(I*wG(0)*2.0*PI/dw)*d2Atmax-d2At0)/pow(wG(0),4);
	dG_w(1)=-real(At(0))/pow(wG(Nw_dense-1),2)-2*M1/pow(wG(Nw_dense-1),3)-3*real(exp(I*wG(Nw_dense-1)*2.0*PI/dw)*d2Atmax-d2At0)/pow(wG(Nw_dense-1),4);
	
	//	double dw_dense=w_dense(1)-w_dense(0);
	//	cout<<"dG_w(0): "<<setw(20)<<dG_w(0)<<real(G_Re_omega(1)-G_Re_omega(0))/dw_dense<<endl;
	//	cout<<"dG_w(1): "<<setw(20)<<dG_w(1)<<real(G_Re_omega(Nw_dense-1)-G_Re_omega(Nw_dense-2))/dw_dense<<endl;
	
//	vec ReGRw=real(G_Re_omega);
//	vec ImGRw=imag(G_Re_omega);
	
/*
	const char attr1[]="'b-'";
	const char attr2[]="'r-'";
	const char attr3[]="'m-'";
	const char attr4[]="'c-'";
	const char lgd1[]="$Re[G^R_{KK}(\\omega)]$";
	const char lgd2[]="$Im[G^R_{KK}(\\omega)]$";
	const char lgd3[]="$Re[G^R_{FFT}(\\omega)]$";
	const char lgd4[]="$Im[G^R_{FFT}(\\omega)]$";
	const char xl[]="$\\omega$";
	
	graph_2D g1;
	
	g1.add_data(w_dense.memptr(),Gr_Re_w.memptr(),N_interv);
	g1.add_attribute(attr1);
	g1.add_to_legend(lgd1);
	g1.add_data(w_dense.memptr(),Gi_Re_w.memptr(),N_interv);
	g1.add_attribute(attr2);
	g1.add_to_legend(lgd2);
	g1.add_data(w_dense.memptr(),ReGRw.memptr(),N_interv);
	g1.add_attribute(attr3);
	g1.add_to_legend(lgd3);
	g1.add_data(w_dense.memptr(),ImGRw.memptr(),N_interv);
	g1.add_attribute(attr4);
	g1.add_to_legend(lgd4);
	g1.set_axes_labels(xl,NULL);
	g1.curve_plot();
	graph_2D::show_figures();
*/

/*
	const char attr1[]="'b-'";
	const char attr2[]="'r-'";
	const char attr3[]="'m-'";
	const char attr4[]="'c-'";
	const char lgd1[]="$Re[G^R(\\omega)]$";
	const char lgd2[]="$Im[G^R(\\omega)]$";
	const char lgd3[]="$Re[G^R(\\omega)_{DFT}]$";
	const char lgd4[]="$Im[G^R(\\omega)_{DFT}]$";
	const char xl[]="$\\omega$";
 
	cx_vec G_R_tmp=-I*dt*((double)N_interv)*ifft(At.rows(0,N_interv-1));
	vec ReGRw1=real(G_R_tmp);
	vec ImGRw1=imag(G_R_tmp);

//	vec w2=linspace<vec>(0,N_interv-1,N_interv)*dw+wG(0);
	

	cx_vec dAt(N_interv,fill::zeros);
	cx_vec TF_dAt(N_interv,fill::zeros);
//	cx_vec d2At(N_interv,fill::zeros);
	
	//	unsigned fftflag = FFTW_MEASURE;
	unsigned fftflag = FFTW_ESTIMATE;
	
	fftw_plan fftplan=fftw_plan_dft_1d(N_interv, reinterpret_cast<fftw_complex *>(dAt.memptr()), reinterpret_cast<fftw_complex *>(TF_dAt.memptr()), FFTW_BACKWARD, fftflag);
	
//	dAt.set_real(coeffs_r.rows(4*l)*pow(dt,2)+coeffs_r.rows(4*l+1)*dt+coeffs_r.rows(4*l+2));
//	dAt.set_imag(coeffs_i.rows(4*l)*pow(dt,2)+coeffs_i.rows(4*l+1)*dt+coeffs_i.rows(4*l+2));
	dAt.set_real(coeffs_r.rows(4*l+2));
	dAt.set_imag(coeffs_i.rows(4*l+2));
	
 	fftw_execute(fftplan);

//	cx_vec TF_dAt=((double)N_interv)*ifft(dAt);

	w2(0)=1;
	
	cx_vec G_R_tmp2=A(0)/w2-I*((exp(I*w2*dt)-1.0)%TF_dAt)/pow(w2,2);
//	cx_vec G_R_tmp2=(A(0)+dt*TF_dAt)/w2;
	
	G_R_tmp2(0)=0;
	w2(0)=0;
	
//	ReGRw=real(G_R_tmp2);
//	ImGRw=imag(G_R_tmp2);
	ReGRw.zeros();
	ImGRw.zeros();
	
	ReGRw.rows(0,Nw_dense-jw0-1)=real(G_Re_omega.rows(jw0,Nw_dense-1));
	ImGRw.rows(0,Nw_dense-jw0-1)=imag(G_Re_omega.rows(jw0,Nw_dense-1));
	
	vec ReGRw2(N_interv,fill::zeros);
	vec ImGRw2(N_interv,fill::zeros);

	ReGRw2.rows(0,Nw_dense-jw0-1)=Gr_Re_w.rows(jw0,Nw_dense-1);
	ImGRw2.rows(0,Nw_dense-jw0-1)=Gi_Re_w.rows(jw0,Nw_dense-1);

	vec ReGRw2(Nw_dense,fill::zeros);
	vec ImGRw2(Nw_dense,fill::zeros);
	
	ReGRw2.rows(jw0,Nw_dense-1)=ReGRw1.rows(0,Nw_dense-jw0-1);
	ReGRw2.rows(0,jw0-1)=ReGRw1.rows(Nw_dense-jw0-1,Nw_dense-2);
	ImGRw2.rows(jw0,Nw_dense-1)=ImGRw1.rows(0,Nw_dense-jw0-1);
	ImGRw2.rows(0,jw0-1)=ImGRw1.rows(Nw_dense-jw0-1,Nw_dense-2);
	
	graph_2D g1;
	
	g1.add_data(wG.memptr(),ReGRw.memptr(),Nw_dense);
	g1.add_attribute(attr1);
	g1.add_to_legend(lgd1);
	g1.add_data(wG.memptr(),ImGRw.memptr(),Nw_dense);
	g1.add_attribute(attr2);
	g1.add_to_legend(lgd2);
	g1.add_data(wG.memptr(),ReGRw2.memptr(),Nw_dense);
	g1.add_attribute(attr3);
	g1.add_to_legend(lgd3);
	g1.add_data(wG.memptr(),ImGRw2.memptr(),Nw_dense);
	g1.add_attribute(attr4);
	g1.add_to_legend(lgd4);
	g1.set_axes_labels(xl,NULL);
	g1.curve_plot();
	graph_2D::show_figures();
*/
}

void OmegaMaxEnt_data::Fourier_transform_spectrum(vec wFt, vec AwFt, vec &t, cx_vec &At)
{
	cout<<"computing the Fourier transform of the spectrum\n";
	
	dcomplex I(0,1);
	
	int N_interv=wFt.n_rows-1;
	
	vec coeffs(4*N_interv,fill::zeros);
	
	AwFt(0)=0;
	AwFt(N_interv)=0;
	
	spline_coeffs(wFt.memptr(), AwFt.memptr(), N_interv+1, coeffs.memptr());
	
	uvec l=linspace<uvec>(0,N_interv-1,N_interv);
	
	double dwFt=wFt(1)-wFt(0);
	
	cx_vec d3Aw(N_interv,fill::zeros);
	cx_vec TF_d3Aw(N_interv,fill::zeros);
	
	//	unsigned fftflag = FFTW_MEASURE;
	unsigned fftflag = FFTW_ESTIMATE;
	
	fftw_plan fftplan=fftw_plan_dft_1d(N_interv, reinterpret_cast<fftw_complex *>(d3Aw.memptr()), reinterpret_cast<fftw_complex *>(TF_d3Aw.memptr()), FFTW_FORWARD, fftflag);
	
	d3Aw.set_real(6*coeffs.rows(4*l));
	d3Aw.set_imag(zeros<vec>(N_interv));
	
	double tmax=2*PI/dwFt;
	
	vec m=linspace<vec>(0,N_interv-1,N_interv);
	t=(2.0*PI*m)/(N_interv*dwFt);
	
	cx_vec ft1=exp(-(I*2.0*PI*m)/((double)N_interv))-1.0;
	cx_vec ft=ft1 % exp(-I*wFt(0)*t);
	
	fftw_execute(fftplan);
	
	TF_d3Aw= ft % TF_d3Aw;
	
//	cx_vec TF_d3Aw= ft % fft(d3Aw);
	
	t(0)=1;
	
	double d2A_wmin=2*coeffs(1);
	double d2A_wmax=6*coeffs(4*N_interv-4)*dwFt+2*coeffs(4*N_interv-3);
	
	At=-I*(d2A_wmax*exp(-I*t*wFt(N_interv))-d2A_wmin*exp(-I*t*wFt(0)))/(2*PI*pow(t,3)) - TF_d3Aw/(2*PI*pow(t,4));
//	At(0)=M0;
	t(0)=0;
	
	At(0)=0;
	double a,b,c,d;
	for (int m=0; m<N_interv; m++)
	{
		a=coeffs(4*m);
		b=coeffs(4*m+1);
		c=coeffs(4*m+2);
		d=coeffs(4*m+3);
		At(0)+=a*pow(dwFt,4)/4+b*pow(dwFt,3)/3+c*pow(dwFt,2)/2+d*dwFt;
	}
	
	At(0)=At(0)/(2*PI);
	
	int Nt=t.n_rows+1;
	t.resize(Nt);
	t(Nt-1)=t(Nt-2)+t(1);
	At.resize(Nt);
	At(Nt-1)=0;

/*
 // compare with the discrete Fourier transform
	int j=0;
	while (j<=N_interv && wFt(j)<0) j++;
	
	cx_vec tmp_p(N_interv-j,fill::zeros);
	cx_vec tmp_m(j,fill::zeros);
	
	tmp_p.set_real(AwFt.rows(j,N_interv-1));
	tmp_m.set_real(AwFt.rows(0,j-1));
	
	cx_vec Aw2(N_interv,fill::zeros);
	Aw2.rows(0,N_interv-j-1)=tmp_p;
	Aw2.rows(N_interv-j,N_interv-1)=tmp_m;
	cx_vec At2=dwFt*fft(Aw2)/(2*PI);
	vec Ar2=real(At2);
	vec Ai2=imag(At2);
*/
/*
	if (show_optimal_alpha_figs)
	{
		vec Ar=real(At);
		vec Ai=imag(At);
		
	//	double dt=t(1);
	//	cout<<"-(Ai(1)-Ai(0))/dt: "<<-(Ai(1)-Ai(0))/dt<<endl;
		
		const char attr1[]="'b-'";
		const char attr2[]="'r-'";
		const char lgd1[]="Re(A(t))";
		const char lgd2[]="Im(A(t))";
		const char xl[]="t";
 
	//	const char attr3[]="'m-'";
	//	const char attr4[]="'c-'";
	//	const char lgd3[]="Re(A(t))";
	//	const char lgd4[]="Im(A(t))";
 
		graph_2D g1;
		g1.add_data(t.memptr(),Ar.memptr(),N_interv);
		g1.add_attribute(attr1);
		g1.add_to_legend(lgd1);
		g1.add_data(t.memptr(),Ai.memptr(),N_interv);
		g1.add_attribute(attr2);
		g1.add_to_legend(lgd2);
		g1.set_axes_labels(xl,NULL);
 
	//	g1.add_data(t.memptr(),Ar2.memptr(),N_interv);
	//	g1.add_attribute(attr3);
	//	g1.add_to_legend(lgd3);
	//	g1.add_data(t.memptr(),Ai2.memptr(),N_interv);
	//	g1.add_attribute(attr4);
	//	g1.add_to_legend(lgd4);
 
		g1.curve_plot();
		graph_2D::show_figures();
	}
*/
	
}

void OmegaMaxEnt_data::KK_integrate(vec w_KK, fctPtr1 Ptr, void *par[], double Rwdw, vec &G_tmp, vec tol, vec lims)
{
	double dw;
	double dw_min=1e-10;
	double inter_min=1e-10;
	double int_lims[2];
	int nbEval[1];
	double wtmp;
	
	int Nint=lims.n_rows-1;
	
	par[4]=&wtmp;
	
	int Nw_KK=w_KK.n_rows;
	
	G_tmp.zeros(Nw_KK,1);
	
	double wp, wm;
	int j,k;
	for (j=0; j<Nw_KK; j++)
	{
		wtmp=w_KK(j);
		dw=Rwdw*fabs(wtmp);
		if (dw<dw_min) dw=dw_min;
		wm=wtmp-dw;
		wp=wtmp+dw;

		k=0;
		while (k<Nint && wm>lims(k+1))
		{
			nbEval[0]=0;
			int_lims[0]=lims(k);
			int_lims[1]=lims(k+1);
			if (wtmp>int_lims[0] && wtmp<int_lims[1]) cout<<"!!  "<<setw(20)<<wtmp<<setw(20)<<int_lims[0]<<int_lims[1]<<setw(20)<<wm<<wp<<endl;
			G_tmp(j)+=quadInteg1D(Ptr, int_lims, tol(j), nbEval, par);
			k++;
		}
		if (k<Nint)
		{
			nbEval[0]=0;
			int_lims[0]=lims(k);
			int_lims[1]=wm;
			G_tmp(j)+=quadInteg1D(Ptr, int_lims, tol(j), nbEval, par);
			if (wp<lims(k+1))
			{
				nbEval[0]=0;
				int_lims[0]=wp;
				int_lims[1]=lims(k+1);
				G_tmp(j)+=quadInteg1D(Ptr, int_lims, tol(j), nbEval, par);
				k++;
			}
			else if (k<Nint-1)
			{
				k++;
				while (k<Nint-1 && wp>lims(k+1)) k++;
				nbEval[0]=0;
				int_lims[0]=wp;
				int_lims[1]=lims(k+1);
				G_tmp(j)+=quadInteg1D(Ptr, int_lims, tol(j), nbEval, par);
				k++;
			}
			while (k<Nint)
			{
				nbEval[0]=0;
				int_lims[0]=lims(k);
				int_lims[1]=lims(k+1);
				G_tmp(j)+=quadInteg1D(Ptr, int_lims, tol(j), nbEval, par);
				k++;
			}
		}
	}
	
}

double OmegaMaxEnt_data::KK_integ(double x, void *par[])
{
	vec *x0=reinterpret_cast<vec*>(par[0]);
	uvec *ind_xlims=reinterpret_cast<uvec*>(par[1]);
	vec *xs=reinterpret_cast<vec*>(par[2]);
	vec *coeffs=reinterpret_cast<vec*>(par[3]);
	double *w_ext=reinterpret_cast<double*>(par[4]);
	
//	if (fabs(*w_ext-x)<1.0e-10) cout<<"*w_ext: "<<setw(20)<<*w_ext<<"x: "<<setw(20)<<x<<"*w_ext-x: "<<*w_ext-x<<endl;
	
	return spline_val_G_part(x, *x0, *ind_xlims, *xs, *coeffs)/(*w_ext-x);
}

double OmegaMaxEnt_data::KK_integ_lin_interp(double x, void *par[])
{
	vec *x0p=reinterpret_cast<vec*>(par[0]);
	vec *Fp=reinterpret_cast<vec*>(par[1]);
	double *w_ext=reinterpret_cast<double*>(par[4]);
	
	vec x0=*x0p;
	vec F=*Fp;
	int Nx0=x0.n_rows;
	int j=0;
	while (j<Nx0-1 && x>x0(j+1)) j++;
	
	double fx=0;
	if (j<Nx0-1)
	{
		fx=F(j)+(x-x0(j))*(F(j+1)-F(j))/(x0(j+1)-x0(j));
	}
	
	return fx/(*w_ext-x);
}

double OmegaMaxEnt_data::KK_integ_lin_interp_chi(double x, void *par[])
{
	vec *x0p=reinterpret_cast<vec*>(par[0]);
	vec *Fp=reinterpret_cast<vec*>(par[1]);
	double *w_ext=reinterpret_cast<double*>(par[4]);
	
	vec x0=*x0p;
	vec F=*Fp;
	int Nx0=x0.n_rows;
	
	double xp=fabs(x);
	int j=0;
	while (j<Nx0-1 && xp>x0(j+1)) j++;
	
	double fx=0;
	if (j<Nx0-1)
	{
		fx=F(j)+(xp-x0(j))*(F(j+1)-F(j))/(x0(j+1)-x0(j));
	}
	
	return fx/(*w_ext-x);
}

double OmegaMaxEnt_data::KK_integ_boson(double x, void *par[])
{
	vec *x0=reinterpret_cast<vec*>(par[0]);
	uvec *ind_xlims=reinterpret_cast<uvec*>(par[1]);
	vec *xs=reinterpret_cast<vec*>(par[2]);
	vec *coeffs=reinterpret_cast<vec*>(par[3]);
	double *w_ext=reinterpret_cast<double*>(par[4]);
	
	return spline_val_G_part(x, *x0, *ind_xlims, *xs, *coeffs)*x/(*w_ext-x);
}

double OmegaMaxEnt_data::KK_integ_chi(double x, void *par[])
{
	vec *x0=reinterpret_cast<vec*>(par[0]);
	uvec *ind_xlims=reinterpret_cast<uvec*>(par[1]);
	vec *xs=reinterpret_cast<vec*>(par[2]);
	vec *coeffs=reinterpret_cast<vec*>(par[3]);
	double *w_ext=reinterpret_cast<double*>(par[4]);
	
	return spline_val_chi_part(x, *x0, *ind_xlims, *xs, *coeffs)*x/(*w_ext-x);
}

bool OmegaMaxEnt_data::diagonalize_covariance_chi()
{
	mat VM, WM, VG, WG;
	
	if (NM>0)
	{
		if (covm_diag)
		{
			VM.eye(NM,NM);
			WM=diagmat(1.0/errM);
		}
		else
		{
			vec COVM_eig;
			mat VMtmp;
			if (!eig_sym(COVM_eig,VMtmp,COVM.submat(0,0,NM-1,NM-1)))
			{
				cout<<"diagonalize_covariance_chi() error: diagonalization of moments covariance matrix failed\n";
				return false;
			}
			if (COVM_eig.min()<=0)
			{
				cout<<"diagonalize_covariance_chi() error: the moments covariance matrix has non-positive eigenvalues\n";
				return false;
			}
			VM.zeros(NM,NM);
			VM.submat(0,0,NM-1,NM-1)=VMtmp;
			vec errM_eig(NM);
			errM_eig.rows(0,NM-1)=sqrt(COVM_eig);
			WM=diagmat(1.0/errM_eig);
		}
		
		
		KM=KM.rows(0,NM-1);
		M_V=WM*VM.t()*M;
		KM_V=WM*VM.t()*KM;
		
		KM_V=KM_V.cols(ind0,Nw-2);
		KM=KM.cols(ind0,Nw-2);
	}
	
	if (cov_diag)
	{
		VG.eye(Nn,Nn);
		WG=diagmat(1.0/errGr);
	}
	else
	{
		vec COVG_eig;
		mat VGtmp;
		if (!eig_sym(COVG_eig,VGtmp,COV))
		{
			cout<<"diagonalize_covariance_chi() error: diagonalization of covariance matrix failed\n";
			return false;
		}
		mat PV=eye<mat>(Nn,Nn);
		PV=flipud(PV);
		COVG_eig=flipud(COVG_eig);
		VG=VGtmp*PV;
		vec GVtmp=VG.t()*Gchi2;
		mat SGV=diagmat(sign(GVtmp));
		SGV=SGV-abs(SGV)+eye<mat>(Nn,Nn);
		VG=VG*SGV;
		WG=diagmat(1.0/sqrt(COVG_eig));
		if (COVG_eig.min()<=0)
		{
			cout<<"diagonalize_covariance_chi() error: the covariance matrix has non-positive eigenvalues\n";
			return false;
		}
	}
	
	G_V=WG*VG.t()*Gchi2;
	KG_V=WG*VG.t()*K.cols(ind0,Nw-2);
	K=K.cols(ind0,Nw-2);
	
	if (NM>0)
	{
		GM=join_vert(M_V, G_V);
		KGM=join_vert(KM_V, KG_V);
	}
	else
	{
		GM=G_V;
		KGM=KG_V;
	}
	NGM=GM.n_rows;
	
	cout<<"number of terms in chi2: "<<NGM<<endl;
	
	return true;
}

bool OmegaMaxEnt_data::Kernel_chi()
{
	dcomplex i(0,1);
	
	bool use_HF_exp=true;
	double fg=1.7;
	double fi=fg;
	int pnmax=100;
	
	cout<<"defining kernel matrix...\n";
	
	Kcx.zeros(Nn,Nw);
	KM.zeros(1,Nw);
 
	int Nud=Nw-Nw_lims(0)-1;
 
	vec ud;
	if (Du_constant)
	{
		double dud=dwr/((wr-w0r)*(wr+dwr-w0r));
		vec ud_int=linspace<vec>(Nud,1,Nud);
		ud=dud*ud_int;
	}
	else
	{
		ud=1.0/(w.rows(Nw_lims(0)+1,Nw-1)-w0r);
	}
	
	mat MM;
	spline_matrix_chi(w, Nw_lims, ws, MM);

	int Nint=Nw;
	int	Nintc=Nwc-1;

	int Ncfs0=3*Nint-1;

	mat MC(Ncfs0+Nw,Nw);
	
	MC.submat(0,0,Ncfs0-1,Nw-1)=MM;
	MC.submat(Ncfs0,0,Ncfs0+Nw-1,Nw-1)=eye<mat>(Nw,Nw);

	mat	Pa_c=zeros<mat>(Nintc,Ncfs0+Nw);
	mat	Pb_c_r=zeros<mat>(Nintc,Ncfs0+Nw);
	mat	Pc_c_r=zeros<mat>(Nintc,Ncfs0+Nw);
	mat	Pd_c_r=zeros<mat>(Nintc,Ncfs0+Nw);
	
	int j;
	for (j=0; j<Nintc; j++)
	{
		Pa_c(j,3*j)=1;
		Pb_c_r(j,3*j+1)=1;
		Pc_c_r(j,3*j+2)=1;
		Pd_c_r(j,j+Ncfs0)=1;
	}

	mat W=diagmat(wc.rows(0,Nwc-2));
	
	mat Pb_c=Pb_c_r-3*W*Pa_c;
	mat Pc_c=Pc_c_r+3*pow(W,2)*Pa_c-2*W*Pb_c_r;
	mat Pd_c=Pd_c_r-pow(W,3)*Pa_c+pow(W,2)*Pb_c_r-W*Pc_c_r;
	
	int Nintd=Nud+1;
	
	mat Pa_d=zeros<mat>(Nintd,Ncfs0+Nw);
	mat Pb_d_r=zeros<mat>(Nintd,Ncfs0+Nw);
	mat Pc_d_r=zeros<mat>(Nintd,Ncfs0+Nw);
	mat Pd_d_r=zeros<mat>(Nintd,Ncfs0+Nw);
	
	for (j=0; j<Nud; j++)
	{
		Pa_d(j,3*j+3*Nintc)=1;
		Pb_d_r(j,3*j+1+3*Nintc)=1;
		Pc_d_r(j,3*j+2+3*Nintc)=1;
		Pd_d_r(j,j+Ncfs0+Nwc)=1;
	}
	
	j=Nud;
	Pa_d(j,3*j+3*Nintc)=1;
	Pb_d_r(j,3*j+1+3*Nintc)=1;
	
	mat U=zeros<mat>(Nud+1,Nud+1);
	vec ud1=zeros<vec>(Nud+1);
	ud1.rows(0,Nud-1)=ud;
	U.diag()=ud1;
	
	mat Pb_d=Pb_d_r-3*U*Pa_d;
	mat Pc_d=Pc_d_r+3*pow(U,2)*Pa_d-2*U*Pb_d_r;
	mat Pd_d=Pd_d_r-pow(U,3)*Pa_d+pow(U,2)*Pb_d_r-U*Pc_d_r;
	
/*
	//test the projectors and the spline
	rowvec x0={0, 1, 2};
	rowvec s0={0.01, 0.1, 0.4};
	rowvec wgt={1,1,1};
	
	vec test_A, test_A2;
	sum_gaussians_chi(w, x0, s0, wgt, test_A);
	
	vec coeffs0=MC*test_A;
	
	vec coeffs1(4*Nw);
	uvec ind_wc=linspace<uvec>(0,Nintc-1,Nintc);
	coeffs1(4*ind_wc)=Pa_c*coeffs0;
	coeffs1(4*ind_wc+1)=Pb_c_r*coeffs0;
	coeffs1(4*ind_wc+2)=Pc_c_r*coeffs0;
	coeffs1(4*ind_wc+3)=Pd_c_r*coeffs0;
	
	uvec ind_wd=linspace<uvec>(Nintc,Nw-1,Nintd);
	coeffs1(4*ind_wd)=Pa_d*coeffs0;
	coeffs1(4*ind_wd+1)=Pb_d_r*coeffs0;
	coeffs1(4*ind_wd+2)=Pc_d_r*coeffs0;
	coeffs1(4*ind_wd+3)=Pd_d_r*coeffs0;
	
	double dw2=wr/(4*(Nwc-1));
	double wmax2=5;
	int Nw2=wmax2/dw2+1;
	vec w2=linspace<vec>(0,wmax2,Nw2);
	sum_gaussians_chi(w2, x0, s0, wgt, test_A2);
	
	vec sv;
	spline_val_chi_part(w2, w, Nw_lims, ws, coeffs1, sv);
	
	graph_2D g1, g2;
	char attr1[]="'+-',color='r',markeredgecolor='r',markerfacecolor='none'";
	char attr2[]="'s-',color='b',markeredgecolor='b',markerfacecolor='none'";
	char attr3[]="'x-',color='k',markeredgecolor='k',markerfacecolor='none'";
	
	char attr4[]="color='r'";
	
	g1.add_data(w.memptr(), test_A.memptr(), Nw);
	g1.add_attribute(attr3);
	g1.add_data(w2.memptr(), test_A2.memptr(), Nw2);
	g1.add_attribute(attr1);
	g1.add_data(w2.memptr(), sv.memptr(), Nw2);
	g1.add_attribute(attr2);
	g1.curve_plot();
	
	plot(g2,w2,(test_A2-sv)/test_A2,NULL,NULL,attr4);
	
	graph_2D::show_figures();
*/

	mat Ka_c=zeros<mat>(Nn,Nintc);
	mat Kb_c=zeros<mat>(Nn,Nintc);
	mat Kc_c=zeros<mat>(Nn,Nintc);
	mat Kd_c=zeros<mat>(Nn,Nintc);
 
	Ka_c.row(0)=-trans(pow(wc.rows(1,Nwc-1),4)-pow(wc.rows(0,Nwc-2),4))/(8*PI);
	Kb_c.row(0)=-trans(pow(wc.rows(1,Nwc-1),3)-pow(wc.rows(0,Nwc-2),3))/(6*PI);
	Kc_c.row(0)=-trans(pow(wc.rows(1,Nwc-1),2)-pow(wc.rows(0,Nwc-2),2))/(4*PI);
	Kd_c.row(0)=-trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2))/(2*PI);
	
//	cx_vec_tmp.zeros(Nintc);
//	vec_tmp=-trans(pow(wc.rows(1,Nwc-1),4)-pow(wc.rows(0,Nwc-2),4))/(8*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Ka_c.row(0)=cx_vec_tmp;
//	vec_tmp=-trans(pow(wc.rows(1,Nwc-1),3)-pow(wc.rows(0,Nwc-2),3))/(6*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Kb_c.row(0)=cx_vec_tmp;
//	vec_tmp=-trans(pow(wc.rows(1,Nwc-1),2)-pow(wc.rows(0,Nwc-2),2))/(4*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Kc_c.row(0)=cx_vec_tmp;
//	vec_tmp=-trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2))/(2*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Kd_c.row(0)=cx_vec_tmp;
 
	mat Wnc=wn.rows(1,Nn-1)*ones<rowvec>(Nintc);
	mat Wc=ones<vec>(Nn-1)*wc.t();
 
	mat logc=log((pow(Wnc,2)+pow(Wc.cols(1,Nintc),2))/(pow(Wnc,2)+pow(Wc.cols(0,Nintc-1),2)));
	mat atanc=atan((Wnc % (Wc.cols(0,Nintc-1)-Wc.cols(1,Nintc)))/(Wc.cols(1,Nintc) % Wc.cols(0,Nintc-1)+pow(Wnc,2)));
	
	mat dWc=Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1);
	mat dWc2=pow(Wc.cols(1,Nintc),2)-pow(Wc.cols(0,Nintc-1),2);
	mat dWc3=pow(Wc.cols(1,Nintc),3)-pow(Wc.cols(0,Nintc-1),3);
	mat Wnc2=pow(Wnc,2);
	mat Wnc3=pow(Wnc,3);
	mat Wnc4=pow(Wnc,4);
 
	Ka_c.rows(1,Nn-1)=real(-i*( -Wnc3 % dWc + i*(Wnc2 % dWc2)/2 + (Wnc % dWc3)/3 -i*(pow(Wc.cols(1,Nintc),4)-pow(Wc.cols(0,Nintc-1),4))/4 - Wnc4 % atanc - i*Wnc4 % logc/2 ))/(2*PI);
	Kb_c.rows(1,Nn-1)=real(-i*( i*Wnc2 % dWc + (Wnc % dWc2)/2 - i*dWc3/3 + i*Wnc3 % atanc - Wnc3 % logc/2 ))/(2*PI);
	Kc_c.rows(1,Nn-1)=real(-i*( Wnc % dWc - i*dWc2/2 + Wnc2 % atanc + i*Wnc2 % logc/2 ))/(2*PI);
	Kd_c.rows(1,Nn-1)=real(-i*( -i*dWc -i*Wnc % atanc + Wnc % logc/2 ))/(2*PI);
	
	int Pmax=pnmax+4;
	imat MP;
	pascal(Pmax+1,MP);
	int pnmax2=pnmax/2;
	
	double wtmp, wj, dw1;
	vec dwp;
	int p,l, jni;
	for (j=0; j<Nwc-1; j++)
	{
		wtmp=abs(wc(j+1));
		if (abs(wc(j))>wtmp) wtmp=abs(wc(j));
		jni=1;
		while (jni<Nn && wn(jni)<fi*wtmp) jni++;
		while (jni<Nn && pow(wn(jni),pnmax)==0) jni++;
		
		if (jni<Nn)
		{
			wj=wc(j);
			dw1=wc(j+1)-wj;
			dwp.zeros(Pmax);
			dwp(0)=dw1;
			dwp(1)=pow(dw1,2)+2*wj*dw1;
			for (p=3; p<=Pmax; p++)
			{
				dwp(p-1)=0;
				for (l=0; l<p; l++)
				{
					dwp(p-1)=dwp(p-1)+MP(l,p-l)*pow(dw1,p-l)*pow(wj,l);
				}
			}
			
			Ka_c.submat(jni,j,Nn-1,j)=zeros<vec>(Nn-jni);
			Kb_c.submat(jni,j,Nn-1,j)=zeros<vec>(Nn-jni);
			Kc_c.submat(jni,j,Nn-1,j)=zeros<vec>(Nn-jni);
			Kd_c.submat(jni,j,Nn-1,j)=zeros<vec>(Nn-jni);
			//				for (p=pnmax; p>=1; p--)
			for (l=pnmax2; l>=1; l--)
			{
				p=2*l;
				Ka_c.submat(jni,j,Nn-1,j)=Ka_c.submat(jni,j,Nn-1,j) + pow(-1,l)*dwp(p+3)/(2*PI*(p+4)*pow(wn.rows(jni,Nn-1),p));
				Kb_c.submat(jni,j,Nn-1,j)=Kb_c.submat(jni,j,Nn-1,j) + pow(-1,l)*dwp(p+2)/(2*PI*(p+3)*pow(wn.rows(jni,Nn-1),p));
				Kc_c.submat(jni,j,Nn-1,j)=Kc_c.submat(jni,j,Nn-1,j) + pow(-1,l)*dwp(p+1)/(2*PI*(p+2)*pow(wn.rows(jni,Nn-1),p));
				Kd_c.submat(jni,j,Nn-1,j)=Kd_c.submat(jni,j,Nn-1,j) + pow(-1,l)*dwp(p)/(2*PI*(p+1)*pow(wn.rows(jni,Nn-1),p));
			}
			
		}
	}
	
	mat Ka_d=zeros<mat>(Nn,Nintd);
	mat Kb_d=zeros<mat>(Nn,Nintd);
	mat Kc_d=zeros<mat>(Nn,Nintd);
	mat Kd_d=zeros<mat>(Nn,Nintd);
	
	rowvec ud2(Nud+1);
	ud2.cols(1,Nud)=ud.t();
	ud2(0)=1.0/(wr-w0r);
	
	mat Wnd=wn.rows(1,Nn-1)*ones<rowvec>(Nintd-1);
	mat Ud=ones<vec>(Nn-1)*ud2;
	
	rowvec wd=1/ud2+w0r;
	mat Wd=ones<vec>(Nn-1)*wd;

	Ka_d.submat(0,0,0,Nintd-2)=-(pow(ud2.cols(1,Nintd-1),2)-pow(ud2.cols(0,Nintd-2),2))/(4*PI);
	Kb_d.submat(0,0,0,Nintd-2)=-(ud2.cols(1,Nintd-1)-ud2.cols(0,Nintd-2))/(2*PI);
	Kc_d.submat(0,0,0,Nintd-2)=-log(ud2.cols(1,Nintd-1)/ud2.cols(0,Nintd-2))/(2*PI);
	Kd_d.submat(0,0,0,Nintd-2)=(wd.cols(1,Nintd-1)-wd.cols(0,Nintd-2))/(2*PI);
	
//	cx_vec_tmp.zeros(Nud);
//	vec_tmp=-(pow(ud2.cols(1,Nintd-1),2)-pow(ud2.cols(0,Nintd-2),2))/(4*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Ka_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
//	vec_tmp=-(ud2.cols(1,Nintd-1)-ud2.cols(0,Nintd-2))/(2*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Kb_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
//	vec_tmp=-log(ud2.cols(1,Nintd-1)/ud2.cols(0,Nintd-2))/(2*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Kc_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
//	vec_tmp=(wd.cols(1,Nintd-1)-wd.cols(0,Nintd-2))/(2*PI);
//	cx_vec_tmp.set_real(vec_tmp);
//	Kd_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
	
	
	mat atand=atan((Wnd % (Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2)))/(1+w0r*(Ud.cols(1,Nintd-1)+Ud.cols(0,Nintd-2))+(pow(w0r,2)+pow(Wnd,2)) % Ud.cols(0,Nintd-2) % Ud.cols(1,Nintd-1)));
	mat logd=log((1+2*w0r*Ud.cols(1,Nintd-1)+pow(Ud.cols(1,Nintd-1),2) % (pow(Wnd,2)+pow(w0r,2)))/(1+2*w0r*Ud.cols(0,Nintd-2)+pow(Ud.cols(0,Nintd-2),2) % (pow(Wnd,2)+pow(w0r,2))));
	
	Ka_d.submat(1,0,Nn-1,Nintd-2)=real(-i*(2*Wnd % (Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2)))/pow(Wnd + i*w0r,2) - i*(w0r*(pow(Ud.cols(1,Nintd-1),2)-pow(Ud.cols(0,Nintd-2),2)))/(Wnd + i*w0r) +  i*(2*Wnd % atand)/pow(Wnd + i*w0r,3)  - (Wnd % logd)/pow(Wnd + i*w0r,3) )/(4*PI);
	Kb_d.submat(1,0,Nn-1,Nintd-2)=real(-i*(2*w0r*(Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2)))/(Wnd + i*w0r) -(2*Wnd % atand)/pow(Wnd + i*w0r,2) -i*(Wnd % logd)/pow(Wnd + i*w0r,2) )/(4*PI);
	Kc_d.submat(1,0,Nn-1,Nintd-2)=real( -2*log(Ud.cols(1,Nintd-1)/Ud.cols(0,Nintd-2)) -i*(2*Wnd % atand)/(Wnd + i*w0r) + (Wnd % logd)/(Wnd + i*w0r) )/(4*PI);
	Kd_d.submat(1,0,Nn-1,Nintd-2)=real( 2*(Wd.cols(1,Nintd-1)-Wd.cols(0,Nintd-2)) -i*2.0*Wnd % log(Ud.cols(1,Nintd-1)/Ud.cols(0,Nintd-2)) + 2*Wnd % atand +i*Wnd % logd )/(4*PI);
	
	mat KC=(Ka_c*Pa_c+Kb_c*Pb_c+Kc_c*Pc_c+Kd_c*Pd_c)*MC;
	mat KD=-(Ka_d*Pa_d+Kb_d*Pb_d+Kc_d*Pc_d+Kd_d*Pd_d)*MC;
 
	Kcx.set_real(KC+KD);
	
	rowvec Knorm_a_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_b_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_c_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_d_c_r=zeros<rowvec>(Nintc);
	
	Knorm_a_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),4)/4);
	Knorm_b_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),3)/3);
	Knorm_c_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),2)/2);
	Knorm_d_c_r=trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2));
	
	rowvec KM0c=(Knorm_a_c_r*Pa_c+Knorm_b_c_r*Pb_c_r+Knorm_c_c_r*Pc_c_r+Knorm_d_c_r*Pd_c_r)*MC/(2*PI);
	
	rowvec Knorm_a_d=zeros<rowvec>(Nintd);
	rowvec Knorm_b_d=zeros<rowvec>(Nintd);
	rowvec Knorm_c_d=zeros<rowvec>(Nintd);
	rowvec Knorm_d_d=zeros<rowvec>(Nintd);
	
	Knorm_a_d.cols(0,Nintd-2)=-(pow(ud2.cols(1,Nud),2)-pow(ud2.cols(0,Nud-1),2))/2;
	Knorm_b_d.cols(0,Nintd-2)=-(ud2.cols(1,Nud)-ud2.cols(0,Nud-1));
	Knorm_c_d.cols(0,Nintd-2)=-log(ud2.cols(1,Nud)/ud2.cols(0,Nud-1));
	Knorm_d_d.cols(0,Nintd-2)=1.0/ud2.cols(1,Nud)-1.0/ud2.cols(0,Nud-1);
	
	Knorm_a_d(Nintd-1)=pow(ud2(Nud-1),2)/2;
	Knorm_b_d(Nintd-1)=ud2(Nud-1);
	
	rowvec KM0d=(Knorm_a_d*Pa_d+Knorm_b_d*Pb_d+Knorm_c_d*Pc_d+Knorm_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM0=KM0c+KM0d;
	
	mat Wjc=diagmat(wc.rows(0,Nintc-1));
	
	rowvec KM1_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),5)/5.0);
	
	rowvec KM1_a_c=KM1_a_c_tmp+Knorm_a_c_r*Wjc;
	rowvec KM1_b_c=Knorm_a_c_r+Knorm_b_c_r*Wjc;
	rowvec KM1_c_c=Knorm_b_c_r+Knorm_c_c_r*Wjc;
	rowvec KM1_d_c=Knorm_c_c_r+Knorm_d_c_r*Wjc;
	
	rowvec KM1c=(KM1_a_c*Pa_c+KM1_b_c*Pb_c_r+KM1_c_c*Pc_c_r+KM1_d_c*Pd_c_r)*MC/(2*PI);
	
	rowvec KM1_a_d=zeros<rowvec>(Nintd);
	rowvec KM1_b_d=zeros<rowvec>(Nintd);
	rowvec KM1_c_d=zeros<rowvec>(Nintd);
	rowvec KM1_d_d=zeros<rowvec>(Nintd);
	
	KM1_a_d.cols(0,Nintd-2)=Knorm_b_d.cols(0,Nintd-2);
	KM1_b_d.cols(0,Nintd-2)=Knorm_c_d.cols(0,Nintd-2);
	KM1_c_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM1_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),2)-1.0/pow(ud2.cols(0,Nud-1),2))/2;
	
	rowvec KM1d_tmp=(KM1_a_d*Pa_d+KM1_b_d*Pb_d+KM1_c_d*Pc_d+KM1_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM1d=w0r*KM0d+KM1d_tmp;
	
	rowvec KM1=KM1c+KM1d;
	
	rowvec KM2_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),6)/6);
	
	rowvec KM2_a_c=KM2_a_c_tmp+2*KM1_a_c_tmp*Wjc+Knorm_a_c_r*pow(Wjc,2);
	rowvec KM2_b_c=KM1_a_c_tmp+2*Knorm_a_c_r*Wjc+Knorm_b_c_r*pow(Wjc,2);
	rowvec KM2_c_c=Knorm_a_c_r+2*Knorm_b_c_r*Wjc+Knorm_c_c_r*pow(Wjc,2);
	rowvec KM2_d_c=Knorm_b_c_r+2*Knorm_c_c_r*Wjc+Knorm_d_c_r*pow(Wjc,2);
	
	rowvec KM2c=(KM2_a_c*Pa_c+KM2_b_c*Pb_c_r+KM2_c_c*Pc_c_r+KM2_d_c*Pd_c_r)*MC/(2*PI);
	
	rowvec KM2_a_d=zeros<rowvec>(Nintd);
	rowvec KM2_b_d=zeros<rowvec>(Nintd);
	rowvec KM2_c_d=zeros<rowvec>(Nintd);
	rowvec KM2_d_d=zeros<rowvec>(Nintd);
	
	KM2_a_d.cols(0,Nintd-2)=Knorm_c_d.cols(0,Nintd-2);
	KM2_b_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM2_c_d.cols(0,Nintd-2)=KM1_d_d.cols(0,Nintd-2);
	KM2_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),3)-1.0/pow(ud2.cols(0,Nud-1),3))/3;
	
	rowvec KM2d_tmp=(KM2_a_d*Pa_d+KM2_b_d*Pb_d+KM2_c_d*Pc_d+KM2_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM2d=pow(w0r,2)*KM0d+2*w0r*KM1d_tmp+KM2d_tmp;
	
	rowvec KM2=KM2c+KM2d;
	
	// the factor of 2 is because the integral in done from omega=0 to omega=inf
//	KM.row(0)=2*KM2;
//	K=2*real(Kcx);
	
	// the additional factor 2 is to use the definition sigma(\omega)=Im[G(\omega)]/\omega instead of sigma(\omega)=2Im[G(\omega)]/\omega
	KM.row(0)=4*KM2;
	K=4*real(Kcx);
	
	cout<<"kernel matrix defined.\n";

/*
	 //test K
	 rowvec x0={0, 1, 2};
	 rowvec s0={0.01, 0.1, 0.4};
	 rowvec wgt={1,1,1};
	 
	 vec test_A;
	 sum_gaussians_chi(w, x0, s0, wgt, test_A);
	 
	 vec Gr_test=K*test_A;
	
	 graph_2D g1, g2;
	 char attr1[]="'o',color='r',markerfacecolor='none'";
	 char attr2[]="'s',color='b',markerfacecolor='none'";
	 char attr3[]="'.-',color='r'";
	 char attr4[]="'.-',color='b'";
	 
	 g1.add_data(wn.memptr(), Gr.memptr(), Nn);
	 g1.add_attribute(attr1);
	 g1.add_data(wn.memptr(), Gr_test.memptr(), Nn);
	 g1.add_attribute(attr2);
	 g1.curve_plot();
	
	 plot(g2,wn,(Gr_test-Gr)/Gr,NULL,NULL,attr3);
	 
	 graph_2D::show_figures();
	
	mat M1_test=KM*test_A;
	
	cout<<"M1: "<<M1_test<<endl;
*/
	
	return true;
}

double OmegaMaxEnt_data::spline_val_chi_part(double x, vec x0, uvec ind_xlims, vec xs, vec coeffs)
{
	if (x<0) x=-x;
		
	int Nx0=x0.n_rows;
	
	int Ncfs=4*Nx0;
	if (coeffs.n_rows<Ncfs)
	{
		cout<<"spline_val_chi_part(): number of elements in vectors \"coeffs\" and \"x0\" are not consistant\n";
		return false;
	}
	
	double xr=x0(ind_xlims(0));
	
	double x0r=xs(0);
	
	double a,b,c,d, Dx, Du;
	
	double sv=0;
	
	int l;
	
	if (x>=0 && x<xr)
	{
		l=0;
		while (x>=x0(l+1) && l<ind_xlims(0)-1) l++;
		a=coeffs(4*l);
		b=coeffs(4*l+1);
		c=coeffs(4*l+2);
		d=coeffs(4*l+3);
		Dx=x-x0(l);
		sv=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
	}
	else if (x>xr)
	{
		l=ind_xlims(0);
		while (x>x0(l+1) && l<Nx0-1) l++;
		a=coeffs(4*l);
		b=coeffs(4*l+1);
		c=coeffs(4*l+2);
		d=coeffs(4*l+3);
		if (l<Nx0-1)
			Du=(x0(l+1)-x)/((x-x0r)*(x0(l+1)-x0r));
		else
			Du=1.0/(x-x0r);
		sv=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
	}
	else
	{
		l=ind_xlims(0)-1;
		a=coeffs(4*l);
		b=coeffs(4*l+1);
		c=coeffs(4*l+2);
		d=coeffs(4*l+3);
		Dx=x-x0(l);
		sv=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
	}
	
	return sv;
}

bool OmegaMaxEnt_data::spline_val_chi_part(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv)
{
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;
	
	int Ncfs=4*Nx0;
	if (coeffs.n_rows<Ncfs)
	{
		cout<<"spline_val_chi_part(): number of elements in vectors \"coeffs\" and \"x0\" are not consistant\n";
		return false;
	}
	
	//	int Nc=ind_xlims(0);
	//	int Nd=Nx0-ind_xlims(1);
	
	double xr=x0(ind_xlims(0));
	
	double x0r=xs(0);
	
	double a,b,c,d, Dx, Du;
	
	sv.zeros(Nx);
	
	int j, l;
    for (j=0; j<Nx; j++)
	{
		if (x(j)>=0 && x(j)<xr)
		{
			l=0;
			while (x(j)>=x0(l+1) && l<ind_xlims(0)-1) l++;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			Dx=x(j)-x0(l);
			sv(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
		}
		else if (x(j)>xr)
		{
			l=ind_xlims(0);
			while (x(j)>x0(l+1) && l<Nx0-1) l++;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			if (l<Nx0-1)
				Du=(x0(l+1)-x(j))/((x(j)-x0r)*(x0(l+1)-x0r));
			else
				Du=1.0/(x(j)-x0r);
			sv(j)=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
		}
		else
		{
			l=ind_xlims(0)-1;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			Dx=x(j)-x0(l);
			sv(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
		}
	}
	
	return true;
}

bool OmegaMaxEnt_data::sum_gaussians_chi(vec x, rowvec x0, rowvec s0, rowvec wgt, vec &F)
{
	int Npks=x0.n_cols;
	
	F.zeros(x.n_rows);

	double W=sum(wgt.cols(0,Npks-1)), s2;
	
	int j;
	for (j=0; j<Npks; j++)
	{
		s2=pow(s0(j),2);
		F=F+wgt(j)*(exp(-pow(x-x0(j),2)/(2*s2)) + exp(-pow(x+x0(j),2)/(2*s2)))/(2*s0(j));
	}
	
	F=sqrt(2*PI)*F/W;
	
	return true;
}

bool OmegaMaxEnt_data::spline_chi_part(vec x, uvec ind_xlims, vec xs, vec F, vec &coeffs)
{
	int j;
	
	int Nx=x.n_rows;
	
	int Nc=ind_xlims(0);
	int Nd=Nx-ind_xlims(0);
	
	double x0d=xs(0);
	
	//solve the spline in the interval [0,wr]
	int NSc=3*Nc-1;
	
	mat Aspl=zeros<mat>(NSc,NSc);
	vec Dc=zeros<vec>(NSc);
	
	double Dx=x(1)-x(0);
	
	Aspl(0,0)=pow(Dx,3);
	Aspl(0,1)=pow(Dx,2);
	Dc(0)=F(1)-F(0);
	
	Aspl(1,0)=3*pow(Dx,2);
	Aspl(1,1)=2*Dx;
	Aspl(1,4)=-1;
	
	Aspl(2,0)=6*Dx;
	Aspl(2,1)=2;
	Aspl(2,3)=-2;
	
	for (j=2; j<Nc; j++)
	{
		Dx=x(j)-x(j-1);
	 
		Dc(3*j-3)=F(j)-F(j-1);
		
		Aspl(3*j-3,3*j-4)=pow(Dx,3);
		Aspl(3*j-3,3*j-3)=pow(Dx,2);
		Aspl(3*j-3,3*j-2)=Dx;
	 
		Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
		Aspl(3*j-2,3*j-3)=2*Dx;
		Aspl(3*j-2,3*j-2)=1;
		Aspl(3*j-2,3*j+1)=-1;
	 
		Aspl(3*j-1,3*j-4)=6*Dx;
		Aspl(3*j-1,3*j-3)=2;
		Aspl(3*j-1,3*j)=-2;
	}
	
	j=Nc;
	Dx=x(j)-x(j-1);
	Dc(3*j-3)=F(j)-F(j-1);
	Aspl(3*j-3,3*j-4)=pow(Dx,3);
	Aspl(3*j-3,3*j-3)=pow(Dx,2);
	Aspl(3*j-3,3*j-2)=Dx;
	
	Dc(3*j-2)=(F(j+1)-F(j-1))/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
	Aspl(3*j-2,3*j-3)=2*Dx;
	Aspl(3*j-2,3*j-2)=1;
	
	vec Cc=solve(Aspl,Dc);
/*
	mat T=zeros<mat>(NSc,Nx);
	for (j=0; j<Nc; j++)
	{
		T(3*j,j)=-1;
		T(3*j,j+1)=1;
	}
	T(3*Nc-2,ind_xlims(0)-1)=-1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	T(3*Nc-2,ind_xlims(0)+1)=1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	
	Cc=Cc*T;
	
	mat Cc2=zeros<mat>(NSc+1,Nx);
	
	Cc2.rows(0,1)=Cc.rows(0,1);
	Cc2.rows(3,NSc)=Cc.rows(2,NSc-1);
*/
	
	//solve the spline in the interval [wr,inf
	int NSd=3*Nd-1;
	Aspl.zeros(NSd,NSd);
	vec Dd=zeros<vec>(NSd);
	
	double u=1/(x(ind_xlims(0))-x0d)-1/(x(ind_xlims(0)+1)-x0d);
	double ud=1/(x(ind_xlims(0))-x0d);
	
	Aspl(0,0)=-3*pow(ud,2)*pow(u,2);
	Aspl(0,1)=-2*pow(ud,2)*u;
	Aspl(0,2)=-pow(ud,2);
	
	j=ind_xlims(0);
	Dd(0)=(F(j+1)-F(j-1))/(x(j+1)-x(j-1));
	
	Aspl(1,0)=pow(u,3);
	Aspl(1,1)=pow(u,2);
	Aspl(1,2)=u;
	Dd(1)=F(j)-F(j+1);
	
	for (j=1; j<Nd-1; j++)
	{
		u=1/(x(ind_xlims(0)+j)-x0d)-1/(x(ind_xlims(0)+j+1)-x0d);
		
		Aspl(3*j-1,3*j-1)=-1;
		Aspl(3*j-1,3*j)=3*pow(u,2);
		Aspl(3*j-1,3*j+1)=2*u;
		Aspl(3*j-1,3*j+2)=1;
		
		Aspl(3*j,3*j-2)=-2;
		Aspl(3*j,3*j)=6*u;
		Aspl(3*j,3*j+1)=2;
	 
		Dd(3*j+1)=F(ind_xlims(0)+j)-F(ind_xlims(0)+j+1);
		Aspl(3*j+1,3*j)=pow(u,3);
		Aspl(3*j+1,3*j+1)=pow(u,2);
		Aspl(3*j+1,3*j+2)=u;
	}
	
	j=Nd-1;
	u=1/(x(ind_xlims(0)+j)-x0d);
	
	Aspl(3*j-1,3*j-1)=-1;
	Aspl(3*j-1,3*j)=3*pow(u,2);
	Aspl(3*j-1,3*j+1)=2*u;
	
	Aspl(3*j,3*j-2)=-2;
	Aspl(3*j,3*j)=6*u;
	Aspl(3*j,3*j+1)=2;
	
	Dd(3*j+1)=F(ind_xlims(0)+j);
	Aspl(3*j+1,3*j)=pow(u,3);
	Aspl(3*j+1,3*j+1)=pow(u,2);
	
	vec Cd=solve(Aspl,Dd);
	
	coeffs.zeros(4*Nx);
	coeffs(0)=Cc(0);
	coeffs(1)=Cc(1);
	coeffs(3)=F(0);
	uvec ind_tmp=linspace<uvec>(1,Nc-1,Nc-1);
	coeffs.rows(4*ind_tmp)=Cc(3*ind_tmp-1);
	coeffs.rows(4*ind_tmp+1)=Cc(3*ind_tmp);
	coeffs.rows(4*ind_tmp+2)=Cc(3*ind_tmp+1);
	coeffs.rows(4*ind_tmp+3)=F.rows(ind_tmp);
	
	ind_tmp=linspace<uvec>(0,Nd-2,Nd-1);
	coeffs.rows(4*ind_tmp+4*Nc)=Cd(3*ind_tmp);
	coeffs.rows(4*ind_tmp+1+4*Nc)=Cd(3*ind_tmp+1);
	coeffs.rows(4*ind_tmp+2+4*Nc)=Cd(3*ind_tmp+2);
	coeffs.rows(4*ind_tmp+3+4*Nc)=F.rows(ind_xlims(0)+ind_tmp+1);
	coeffs(4*Nx-4)=Cd(3*Nd-3);
	coeffs(4*Nx-3)=Cd(3*Nd-2);

/*
	T.zeros(NSd,Nx);
	T(0,ind_xlims(0)-1)=-1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	T(0,ind_xlims(0)+1)=1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	for (j=1; j<Nd; j++)
	{
		T(3*j-2,ind_xlims(0)+j-1)=1;
		T(3*j-2,ind_xlims(0)+j)=-1;
	}
	j=Nd;
	T(3*j-2,ind_xlims(0)+j-1)=1;
	
	Cd=Cd*T;
	
	int NS_tot=NSc+NSd+1;
	
	Mspl.zeros(NS_tot,Nx);
	Mspl.rows(0,NSc)=Cc2;
	Mspl.rows(NSc+1,NS_tot-1)=Cd;
*/
	return true;
}

bool OmegaMaxEnt_data::spline_matrix_chi(vec x, uvec ind_xlims, vec xs, mat &Mspl)
{
	int j;
	
	int Nx=x.n_rows;
	
	int Nc=ind_xlims(0);
	int Nd=Nx-ind_xlims(0);
	
	double x0d=xs(0);
	
	//solve the spline in the interval [0,wr]
	int NSc=3*Nc-1;
	
	mat Aspl=zeros<mat>(NSc,NSc);
	
	double Dx=x(1)-x(0);
	
	Aspl(0,0)=pow(Dx,3);
	Aspl(0,1)=pow(Dx,2);
	
	Aspl(1,0)=3*pow(Dx,2);
	Aspl(1,1)=2*Dx;
	Aspl(1,4)=-1;
	
	Aspl(2,0)=6*Dx;
	Aspl(2,1)=2;
	Aspl(2,3)=-2;
	
	for (j=2; j<Nc; j++)
	{
		Dx=x(j)-x(j-1);
	 
		Aspl(3*j-3,3*j-4)=pow(Dx,3);
		Aspl(3*j-3,3*j-3)=pow(Dx,2);
		Aspl(3*j-3,3*j-2)=Dx;
	 
		Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
		Aspl(3*j-2,3*j-3)=2*Dx;
		Aspl(3*j-2,3*j-2)=1;
		Aspl(3*j-2,3*j+1)=-1;
	 
		Aspl(3*j-1,3*j-4)=6*Dx;
		Aspl(3*j-1,3*j-3)=2;
		Aspl(3*j-1,3*j)=-2;
	}
	
	j=Nc;
	Dx=x(j)-x(j-1);
	Aspl(3*j-3,3*j-4)=pow(Dx,3);
	Aspl(3*j-3,3*j-3)=pow(Dx,2);
	Aspl(3*j-3,3*j-2)=Dx;
	
	Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
	Aspl(3*j-2,3*j-3)=2*Dx;
	Aspl(3*j-2,3*j-2)=1;
	
	mat D(NSc,NSc);
	D.eye();
	mat Cc=solve(Aspl,D);
	
	mat T=zeros<mat>(NSc,Nx);
	for (j=0; j<Nc; j++)
	{
		T(3*j,j)=-1;
		T(3*j,j+1)=1;
	}
	T(3*Nc-2,ind_xlims(0)-1)=-1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	T(3*Nc-2,ind_xlims(0)+1)=1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	
	Cc=Cc*T;
	
	mat Cc2=zeros<mat>(NSc+1,Nx);
	
	Cc2.rows(0,1)=Cc.rows(0,1);
	Cc2.rows(3,NSc)=Cc.rows(2,NSc-1);
	
	//solve the spline in the interval [wr,inf
	int NSd=3*Nd-1;
	Aspl.zeros(NSd,NSd);
	
	double u=1/(x(ind_xlims(0))-x0d)-1/(x(ind_xlims(0)+1)-x0d);
	double ud=1/(x(ind_xlims(0))-x0d);
	
	Aspl(0,0)=-3*pow(ud,2)*pow(u,2);
	Aspl(0,1)=-2*pow(ud,2)*u;
	Aspl(0,2)=-pow(ud,2);
	
	Aspl(1,0)=pow(u,3);
	Aspl(1,1)=pow(u,2);
	Aspl(1,2)=u;
	
	for (j=1; j<Nd-1; j++)
	{
		u=1/(x(ind_xlims(0)+j)-x0d)-1/(x(ind_xlims(0)+j+1)-x0d);
		
		Aspl(3*j-1,3*j-1)=-1;
		Aspl(3*j-1,3*j)=3*pow(u,2);
		Aspl(3*j-1,3*j+1)=2*u;
		Aspl(3*j-1,3*j+2)=1;
		
		Aspl(3*j,3*j-2)=-2;
		Aspl(3*j,3*j)=6*u;
		Aspl(3*j,3*j+1)=2;
	 
		Aspl(3*j+1,3*j)=pow(u,3);
		Aspl(3*j+1,3*j+1)=pow(u,2);
		Aspl(3*j+1,3*j+2)=u;
	}
	
	j=Nd-1;
	u=1/(x(ind_xlims(0)+j)-x0d);
	
	Aspl(3*j-1,3*j-1)=-1;
	Aspl(3*j-1,3*j)=3*pow(u,2);
	Aspl(3*j-1,3*j+1)=2*u;
	
	Aspl(3*j,3*j-2)=-2;
	Aspl(3*j,3*j)=6*u;
	Aspl(3*j,3*j+1)=2;
	
	Aspl(3*j+1,3*j)=pow(u,3);
	Aspl(3*j+1,3*j+1)=pow(u,2);
	
	D.eye(NSd,NSd);
	mat Cd=solve(Aspl,D);
	
	T.zeros(NSd,Nx);
	T(0,ind_xlims(0)-1)=-1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	T(0,ind_xlims(0)+1)=1/(x(ind_xlims(0)+1)-x(ind_xlims(0)-1));
	for (j=1; j<Nd; j++)
	{
		T(3*j-2,ind_xlims(0)+j-1)=1;
		T(3*j-2,ind_xlims(0)+j)=-1;
	}
	j=Nd;
	T(3*j-2,ind_xlims(0)+j-1)=1;
	
	Cd=Cd*T;
	
	int NS_tot=NSc+NSd+1;
	
	Mspl.zeros(NS_tot,Nx);
	Mspl.rows(0,NSc)=Cc2;
	Mspl.rows(NSc+1,NS_tot-1)=Cd;
	
	return true;
}

bool OmegaMaxEnt_data::set_default_model_chi()
{
	if (def_model_file.size())
	{
		cout<<"default model provided\n";
		vec w_def=def_data.col(0);
		vec def_m=def_data.col(1);
		uint Nw_def=w_def.n_rows;
		if (w_def(0)!=0)
		{
			cout<<"set_default_model_chi(): if \"Im(G) column in data file\" is smaller than 1, the first frequency of the default model must be 0\n";
			return false;
		}
		if (def_m.min()<0)
		{
			cout<<"set_default_model_chi() error: provided default model has a negative value.\n";
			return false;
		}
		double wmx=w(Nw-1);
		
		double wmx_def=w_def(Nw-1);
		int j1, j2, j3;
		j1=Nw_def-3;
		j2=Nw_def-2;
		j3=Nw_def-1;
		double w0r_def=(2*w_def(j1)*w_def(j3)-w_def(j1)*w_def(j2)-w_def(j2)*w_def(j3))/(w_def(j1)-2*w_def(j2)+w_def(j3));
		if (w0r_def>wmx_def)
		{
			if (w_def(Nw_def-1)>wr)
			{
				int j=Nw_def-1;
				while (w_def(j)>wr && j>0)	 j--;
				int jr=j+1;
				if (jr>Nw_def-1) jr=Nw_def-1;
				
				if (def_m(jr-1)>def_m(jr))
				{
					w_def=w_def.rows(0,jr);
					def_m=def_m.rows(0,jr);
					Nw_def=jr+1;
					
					double l21, l31, w1, w2, w3;
					l21=log(def_m(Nw_def-2)/def_m(Nw_def-3));
					l31=log(def_m(Nw_def-1)/def_m(Nw_def-3));
					w1=w_def(Nw_def-3);
					w2=w_def(Nw_def-2);
					w3=w_def(Nw_def-1);
					double wcd=(l21*(pow(w1,2)-pow(w3,2))-l31*(pow(w1,2)-pow(w2,2)))/(2*l31*(w2-w1)-2*l21*(w3-w1));
					double C1d=(pow(w1-wcd,2)-pow(w2-wcd,2))/l21;
					double C2d=exp(-pow(w1-wcd,2)/C1d)/def_m(Nw_def-3);
					
					vec gaussians_params(3);
					gaussians_params(0)=wcd;
					gaussians_params(1)=C1d;
					gaussians_params(2)=C2d;
					
					double dfd=-2*(w_def(Nw_def-1)-wcd)*def_m(Nw_def-1)/C1d;
					
					uint Nc=4*(Nw_def-1);
					vec coeffs_spline_def(Nc+1);
					coeffs_spline_def(0)=0;
					coeffs_spline_def(1)=dfd;
					coeffs_spline_def(Nc)=def_m(Nw_def-1);
					
					if (dfd<=0 && C1d>0 && C2d>0)
					{
						spline_coeffs(w_def.memptr(),def_m.memptr(),Nw_def,coeffs_spline_def.memptr());
						if (!default_model_val_chi(w, w_def, coeffs_spline_def, gaussians_params, default_model))
						{
							return false;
						}
					}
					else
					{
						cout<<"set_default_model_chi(): problem with the user defined default model.\n";
						if (dfd>=0)
							cout<<"The derivative does not have the correct sign at the right boundary\n";
						if ( (C1d<0 || C2d<0) && w_def(Nw_def)<w(Nw) )
							cout<<"Incorrect parameters found. Unable to extend the model to whole frequency range.\n";
						
						return false;
					}
				}
				else if (def_m(jr-1)==def_m(jr))
				{
					default_model.zeros(Nw);
					default_model(0)=def_m(0);
					int l=1;
					for (j=1; j<Nw-1; j++)
					{
						while (l<Nw_def-1 && w_def(l)<=w(j)) l++;
						
						if (w(j)>=w_def(l-1) && w(j)<w_def(l))
							default_model(j)=(w(j)-w_def(l-1))*(def_m(l)-def_m(l-1))/(w_def(l)-w_def(l-1))+def_m(l-1);
						else if (w(j)==w_def(l))
							default_model(j)=def_m(l);
					}
					//				rowvec dw_tmp(Nw-1);
					//				dw_tmp(0)=w(1)/2;
					//				dw_tmp.cols(1,Nw-2)=trans(w.rows(2,Nw-1)-w.rows(0,Nw-3))/2.0;
					//				mat int_def_m=dw_tmp*(default_model.rows(0,Nw-2))/PI;
					//				default_model=M1n*default_model/int_def_m(0);
				}
				else
				{
					cout<<"warning: the provided default model is not valid\n";
					return false;
				}
			}
			else
			{
				cout<<"warning: the grid of the user defined default model must extend beyond the frequency range of the spectral function\n";
				return false;
			}
		}
		else
		{
			cout<<"provided default model was generated by this code.\n";
			if (displ_adv_prep_figs)
			{
				vec w_def2=(w_def.rows(0,Nw_def-2)+w_def.rows(1,Nw_def-1))/2;
				vec Dw_def=1.0/(w_def.rows(1,Nw_def-1)-w_def.rows(0,Nw_def-2));
				graph_2D g1;
				char xl[]="$\\\\omega$";
				char yl[]="$1/(\\\\omega_{i+1}-\\\\omega_i)$";
				char ttl[]="provided default model grid density";
				char attr[]="'o-',markeredgecolor='b', markerfacecolor='none'";
				g1.add_title(ttl);
				plot(g1,w_def2,Dw_def,xl,yl,attr);
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			uvec ind_xlims(1);
			vec xs(1);
			xs(0)=w0r_def;
			int j;
			double dw1, dw2, rdw;
			j=Nw_def-3;
			dw1=w_def(j+1)-w_def(j);
			dw2=w_def(j+2)-w_def(j+1);
			rdw=dw2/dw1;
			while (abs(rdw-1.0)>tol_rdw)
			{
				j=j-1;
				dw2=dw1;
				dw1=w_def(j+1)-w_def(j);
				rdw=dw2/dw1;
			}
			ind_xlims(0)=j+1;
			
			vec coeffs;
			spline_chi_part(w_def, ind_xlims, xs, def_m, coeffs);
			spline_val_chi_part(w, w_def, ind_xlims, xs, coeffs, default_model);
			
			/*
			//test the spline
			rowvec x0={0, 4};
			rowvec s0={0.5, 1};
			rowvec wgt={1,1};
			
			vec test_A1, test_A2, test_A3;
			sum_gaussians_chi(w_def, x0, s0, wgt, test_A1);
			
			vec coeffs2;
			spline_chi_part(w_def, ind_xlims, xs, test_A1, coeffs2);
			
			int Nw2=1000;
			vec w2=linspace<vec>(0,wr+SW,Nw2);
			sum_gaussians_chi(w2, x0, s0, wgt, test_A2);
			
			vec sv;
			spline_val_chi_part(w2, w_def, ind_xlims, xs, coeffs2, test_A3);
			
			graph_2D g1, g2;
			char xl[]="$\\\\omega$";
			char yl[]="A";
			char attr1[]="'o-',color='r',markeredgecolor='r',markerfacecolor='none'";
			char attr2[]="'s-',color='b',markeredgecolor='b',markerfacecolor='none'";
			char attr3[]="'+-',color='m',markeredgecolor='m',markerfacecolor='none'";
			
			char attr4[]="color='r'";
			
			double xlims[2], ylims[2];
			xlims[0]=w2(0);
			xlims[1]=w2(Nw2-1);
			ylims[0]=0;
			ylims[1]=1.1*test_A2.max();
			
			g1.add_data(w_def.memptr(), test_A1.memptr(), Nw_def);
			g1.add_attribute(attr1);
			g1.add_data(w2.memptr(), test_A2.memptr(), Nw2);
			g1.add_attribute(attr2);
			g1.add_data(w2.memptr(), test_A3.memptr(), Nw2);
			g1.add_attribute(attr3);
			g2.set_axes_labels(xl,yl);
			g2.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			
			char yl2[]="Delta A";
			vec Delta_A=test_A2-test_A3;
			ylims[0]=1.1*Delta_A.min();
			ylims[1]=1.1*Delta_A.max();
			g2.add_data(w2.memptr(),Delta_A.memptr(),w2.n_rows);
			g2.add_attribute(attr1);
			g2.set_axes_labels(xl,yl2);
			g2.set_axes_lims(xlims,ylims);
			g2.curve_plot();
			
			graph_2D::show_figures();
			 */
		}
	}
	else
	{
		default_model_center=0;
		if (!default_model_shape_in.size())
			default_model_shape=2.0;
		if (!default_model_width_in.size())
		{
			if (std_omega)
				default_model_width=pow(2.0,1.0/default_model_shape)*std_omega;
			else
				default_model_width=pow(2.0,1.0/default_model_shape)*SW/2;
		}
		general_normal(w, default_model_center, default_model_width, default_model_shape, default_model);
		
		/*
		double width_DM;
		if (std_omega)
			width_DM=std_omega;
		else
			width_DM=SW/2;
		rowvec x0={0};
		rowvec s0={width_DM};
		rowvec wgt={1.0};
		sum_gaussians(w,x0,s0,wgt,default_model);
		 */
	}
	
	default_model=abs(default_model);
	
	if (displ_prep_figs)
	{
		graph_2D g1;
		char xl[]="$\\\\omega$";
		char yl[]="default model";
		g1.add_title(yl);
		plot(g1,w,default_model,xl,NULL);
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	return true;
}

bool OmegaMaxEnt_data::default_model_val_chi(vec x, vec x0, vec coeffs, vec gaussians_params, vec &dm)
{
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;
	
	double wcd=gaussians_params(0);
	double C1d=gaussians_params(1);
	double C2d=gaussians_params(2);
	
	if (x0(0)!=0)
	{
		cout<<"default_model_val_chi(): the first value of x0 must be 0\n";
		return false;
	}
	
	vec diff_x=x.rows(1,Nx-1)-x.rows(0,Nx-2);
	if (diff_x.min()<=0)
	{
		cout<<"default_model_val_chi(): values in position vector are not strictly increasing\n";
		return false;
	}
	
	dm.zeros(Nx);
	
	int j=0;
	while (x(j)<x0(Nx0-1) && j<Nx-1) j++;
	int jr=j-1;
	int Nm=jr+1;
	int Nr=Nx-Nm;
	
	if (Nm)
	{
		vec vm;
		spline_val(x.rows(0,jr), x0, coeffs, vm);
		dm.rows(0,jr)=vm;
	}
	if (Nr)
	{
		vec vr=exp(-pow(x.rows(jr+1,Nx-1)-wcd,2)/C1d)/C2d;
		dm.rows(jr+1,Nx-1)=vr;
	}
	
	return true;
}

bool OmegaMaxEnt_data::truncate_chi_omega_n()
{
	double wn_max=wn(ind_cutoff_wn);
	wn=wn.rows(0,ind_cutoff_wn);
	n=n.rows(0,ind_cutoff_wn);
	Nn=ind_cutoff_wn+1;
	if (Nn>Nn_max) // || Nn>=Nw)
	{
		cout<<"Number of Matsubara frequencies larger than the maximum allowed...\n";
		uvec diff_n=n.rows(1,Nn-1)-n.rows(0,Nn-2);
		if (diff_n.max()==1)
		{
			cout<<"Using a non-uniform Matsubara grid.\n";
			int i;
			int p=ceil(log2(Nn));
			int N0=pow(2,p);
			if (N0+1>Nn_all)
			{
				p=p-1;
				N0=pow(2,p);
			}
			int r=1;
			int N1=N0/pow(2,r);
			int N2=N1/2;
			Nn=N1+r*N2+1;
			n.zeros(Nn);
			n.rows(0,N1-1)=linspace<uvec>(0,N1-1,N1);
			uvec j=linspace<uvec>(0,Nn-N1-1,Nn-N1);
			uvec lj=j/N2;
			for (i=0; i<Nn-N1; i++) n(i+N1)=(j(i) % N2)*pow(2,lj(i)+1) + N1*pow(2,lj(i));
			wn=2*PI*tem*conv_to<vec>::from(n);
			i=Nn-1;
			while (wn(i)>wn_max && i>0) i--;
			Nn=i+1;
			while (Nn>Nn_max) //|| Nn>=Nw)
			{
				r=r+1;
				N1=N0/pow(2,r);
				N2=N1/2;
				Nn=N1+r*N2+1;
				n.zeros(Nn);
				n.rows(0,N1-1)=linspace<uvec>(0,N1-1,N1);
				uvec j=linspace<uvec>(0,Nn-N1-1,Nn-N1);
				uvec lj=j/N2;
				for (i=0; i<Nn-N1; i++) n(i+N1)=(j(i) % N2)*pow(2,lj(i)+1) + N1*pow(2,lj(i));
				wn=2*PI*tem*conv_to<vec>::from(n);
				i=Nn-1;
				while (wn(i)>wn_max && i>0) i--;
				Nn=i+1;
			}
			wn=wn.rows(0,Nn-1);
			n=n.rows(0,Nn-1);
			G=G.rows(n);
			Gr=real(G);
			Gi=imag(G);
			
			Gchi2=Gr;
			COV=COV(n,n);
			
			if (cov_diag)
			{
				errGr=errGr.rows(n);
				errG=errGr;
			}
			
			if (displ_prep_figs)
			{
				graph_2D g1;
				char xl[]="frequency number";
				char yl[]="frequency index";
				char attr[]="'o'";
				j=linspace<uvec>(1,Nn,Nn);
				vec jv=conv_to<vec>::from(j);
				vec nv=conv_to<vec>::from(n);
				plot(g1,jv,nv,xl,yl,attr);
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
		}
		else
		{
			cout<<"Your Matsubara grid is not uniform. Either make it more sparse to reduce further the number of frequencies, or increase \"Nn_max\" in the file \"OmegaMaxEnt_other_params.dat\".\n";
			return false;
		}
	}
	else
	{
		G=G.rows(0,Nn-1);
		Gr=real(G);
		Gi=imag(G);
		Gchi2=Gr;
		COV=COV.submat(0,0,Nn-1,Nn-1);
		if (cov_diag)
		{
			errGr=errG.rows(0,Nn-1);
			errG=errGr;
		}
	}
	
	return true;
}

bool OmegaMaxEnt_data::set_omega_grid_chi()
{
	if (!SW_set || !wc_exists) return false;
	
	int j;
	
	double wmax=f_w_range*SW/2.0;
	if (wr>0)
	{
		if (wmax<R_wmax_wr_min*wr) wmax=R_wmax_wr_min*wr;
	}
	w0r=wr-sqrt(dwr*(wmax-wr));
	int Nur=ceil((wr-w0r)/dwr);
	w0r=wr-Nur*dwr;
	double dur=dwr/((wr-w0r)*(wr+dwr-w0r));
	wmax=1.0/dur+w0r;
	ivec ur_int=linspace<ivec>(Nur,1,Nur);
	vec ur(Nur);
	for (j=0; j<Nur; j++)
		ur(j)=dur*ur_int(j);
	
	vec wr_vec=1.0/ur+w0r;
	Nw=Nur+Nwc;
	w.zeros(Nw);
	w.rows(0,Nwc-1)=wc;
	w.rows(Nwc,Nw-1)=wr_vec;
	w_exists=true;
	ws.zeros(1);
	ws(0)=w0r;
	Nw_lims.zeros(1);
	Nw_lims(0)=Nwc-1;
	
	Du_constant=true;
	
	return true;
}

bool OmegaMaxEnt_data::set_wc_chi()
{
	bool use_nu_grid=false;
	
	double dw_min;
	
	if (non_uniform_grid && step_omega_in.size())
	{
		dw_min=step_omega;
		
		use_nu_grid=true;
	}
	else if (non_uniform_grid && peak_exists)
	{
		dw_min=2*dw_peak/R_peak_width_dw;
		
		use_nu_grid=true;
	}
	
//	if (step_omega_in.size() || !non_uniform_grid || !peak_exists || !use_nu_grid)
	if (!use_nu_grid)
	{
		cout<<"uniform grid\n";
		
		double dw;
		
		if (SW_set && !main_spectral_region_set)
		{
			wr=SW/2;
			main_spectral_region_set=true;
		}
		else if (std_omega && !main_spectral_region_set)
		{
			SW=f_SW_std_omega*std_omega;
			SW_set=true;
			wr=SW/2;
			main_spectral_region_set=true;
		}
		else if (main_spectral_region_set)
		{
			SW=2*wr;
			SW_set=true;
		}
		else
		{
			cout<<"The central part of the grid cannot be defined. Not enough information available.\n";
			return false;
		}
		
		if (step_omega_in.size())	dw=step_omega;
		else dw=SW/(f_SW_std_omega*Rmin_SW_dw);
		
/*
		if (peak_exists && dw>dw_peak)
		{
			cout<<"Warning: the step is larger than the estimated width of the peak at low energy. You can use the parameters of section FREQUENCY GRID PARAMETERS to make the grid better adapted to the spectrum.\n";
		}
*/
		wl=0;
		Nwc=round(wr/dw);
		wr=Nwc*dw;
		Nwc=Nwc+1;
		wc=linspace<vec>(0,wr,Nwc);
		
		dwr=dw;
		wc_exists=true;
		main_spectral_region_set=true;
		
		return true;
	}
	else
	{
		if (!SW_set && !jfit && !std_omega)
		{
			cout<<"The central part of the grid cannot be defined. Not enough information available.\n";
			return false;
		}
		
		double wr_tmp;
		
		omega_grid_params.zeros(3);
		
		if (SW_in.size())
		{
			wr_tmp=SW/2;
		}
		else if (jfit)
		{
			wr_tmp=wn(jfit)/R_wncutoff_wr;
		}
		else if (SW_set)
		{
			wr_tmp=R_SW_wr*SW/2;
		}
		else
		{
			wr_tmp=R_SW_wr*f_SW_std_omega*std_omega/2;
		}
		
		if (SW_set)
		{
			double wmax_tmp=f_w_range*SW/2.0;
			if (wr_tmp>wmax_tmp/R_wmax_wr_min) wr_tmp=wmax_tmp/R_wmax_wr_min;
		}
		
		if (SW_set)
		{
			if (wr_tmp<R_SW_wr*SW/2) wr_tmp=R_SW_wr*SW/2;
		}
		
		omega_grid_params(1)=dw_min;
		
		rowvec params_tmp;
		int j=2;
		omega_grid_params(2)=R_Dw_dw*omega_grid_params(1);
		
	//	if (peak_exists && omega_grid_params(2)<4*dw_peak) omega_grid_params(2)=4*dw_peak;
		
		if (SW_set)
		{
			if (omega_grid_params(1)>SW/(f_SW_std_omega*Rmin_SW_dw))	omega_grid_params(1)=SW/(f_SW_std_omega*Rmin_SW_dw);
		}
//		cout<<"wr_tmp: "<<wr_tmp<<endl;
		
//		cout<<omega_grid_params(0)<<endl;
//		cout<<omega_grid_params(1)<<endl;
//		cout<<omega_grid_params(2)<<endl;
		while (omega_grid_params(j)<wr_tmp)
		{
			params_tmp.zeros(j+3);
			params_tmp.cols(0,j)=omega_grid_params;
			omega_grid_params=params_tmp;
			j++;
			omega_grid_params(j)=2*omega_grid_params(j-2);
//			cout<<omega_grid_params(j)<<endl;
			j++;
			omega_grid_params(j)=omega_grid_params(j-2)+R_Dw_dw*omega_grid_params(j-1);
//			cout<<omega_grid_params(j)<<endl;
		}
		
		return set_grid_from_params_chi();
	}
}

/*
bool OmegaMaxEnt_data::set_wc_chi()
{
	double dw;
	
	if (SW_set && !main_spectral_region_set)
	{
		wr=SW/2;
		main_spectral_region_set=true;
	}
	else if (std_omega && !main_spectral_region_set)
	{
		SW=f_SW_std_omega*std_omega;
		SW_set=true;
		wr=SW/2;
		main_spectral_region_set=true;
	}
	else if (!main_spectral_region_set)
	{
		cout<<"The central part of the grid cannot be defined. Not enough information available.\n";
		return false;
	}
	
	if (step_omega_in.size())
	{
		dw=step_omega;
		if (peak_exists)
		{
			if (dw>dw_peak)
			{
				cout<<"warning: step is larger than the estimated width of the peak at low energy\n";
			}
		}
	}
	else
	{
	//	dw=SW/(2.0*Nw_min);
		dw=SW/Rmin_SW_dw;
		if (peak_exists)
		{
			if (dw_peak<dw)
				dw=dw_peak;
		}
	}
	
	wl=0;
	Nwc=round(wr/dw);
	wr=Nwc*dw;
	Nwc=Nwc+1;
	wc=linspace<vec>(0,wr,Nwc);
	
	dwr=dw;
	wc_exists=true;
	main_spectral_region_set=true;
	
	return true;
}
*/

bool OmegaMaxEnt_data::set_grid_from_params_chi()
{
	
	int j;
	bool grid_set=false;
	cout<<"grid parameters provided\n";
	uint Npar_grid=omega_grid_params.n_cols;
	
	if (omega_grid_params(0)!=0)
	{
		cout<<"set_grid_from_params_chi(): if \"Im(G) column in data file\" is smaller than 1, the first parameter in grid parameters must be 0\n";
		return false;
	}
	
	if ( Npar_grid % 2 )
	{
		uint Nint=(Npar_grid-1)/2;
		uint Nlims=Nint+1;
		uvec odd_ind=linspace<uvec>(1,Npar_grid-2,Nint);
		rowvec dw_par=omega_grid_params.cols(odd_ind);
		bool Rdw_too_large=false, Rdw_too_small=false;
		for (j=0; j<Nint-1; j++)
		{
			if (dw_par(j+1)/dw_par(j)>Rdw_max) Rdw_too_large=true;
			if (dw_par(j+1)/dw_par(j)<(1.0/Rdw_max)) Rdw_too_small=true;
		}
		if (Rdw_too_large || Rdw_too_small)
		{
			cout<<"The ratio of frequency steps in adjacent intervals of \"grid parameters\" is too large. Spurious oscillations may appear in the spectral function. The maximum ratio recommended is "<<Rdw_max<<endl;
		}
		uvec even_ind=linspace<uvec>(0,Npar_grid-1,Nlims);
		rowvec wlims_par=omega_grid_params.cols(even_ind);
		
		//		cout<<"Dw: "<<dw_par<<endl;
		//		cout<<"wlims: "<<wlims_par<<endl;
		bool ordered=true;
		for (j=1; j<Nlims; j++)
		{
			if (wlims_par(j)<wlims_par(j-1)) ordered=false;
		}
		if (ordered)
		{
			bool grid_par_ok=true;
			bool grid_step_pos=true;
			for (j=1; j<Nlims; j++)
			{
				if (dw_par(j-1)>=(wlims_par(j)-wlims_par(j-1))/Rmin_Dw_dw) grid_par_ok=false;
				if (dw_par(j-1)<0)
				{
					grid_par_ok=false;
					grid_step_pos=false;
				}
			}
			if (grid_par_ok)
			{
				double w0tmp=(wlims_par(0)+wlims_par(Nlims-1))/2.0;
				j=0;
				while (w0tmp>=wlims_par(j+1))
				{
					j=j+1;
				}
				double w0_par=0;
				vec R(2);
				R(0)=RW_grid;
				R(1)=RWD_grid;
				if (non_uniform_frequency_grid(dw_par, wlims_par, w0_par, R, wc))
				{
					wc_exists=true;
					Nwc=wc.n_rows;
					wr=wc(Nwc-1);
					main_spectral_region_set=true;
					dwr=wc(Nwc-1)-wc(Nwc-2);
					if (!SW_set)
					{
						SW=2*wr;
						SW_set=true;
					}
					if (!SC_set)
					{
						SC=0;
						SC_set=true;
					}
					grid_set=true;
				}
				else
				{
					cout<<"parameterized grid definition failed\n";
				}
			}
			else
			{
				if (grid_step_pos)
					cout<<"Error: the frequency step in \"grid parameters\" is too large in at least one interval. The step must be at least "<<Rmin_Dw_dw<<" times smaller than the interval size.\n";
				else
					cout<<"Error: negative step value found in \"grid parameters\"\n";
			}
		}
		else
		{
			cout<<"Error: interval boundaries in \"grid parameters\" are not strictly increasing values\n";
		}
	}
	else
	{
		cout<<"Error: \"grid parameters\" must have an odd number of elements\n";
	}
	
//	cout<<"grid_set: "<<grid_set<<endl;
	
	return grid_set;
}

bool OmegaMaxEnt_data::set_grid_omega_from_file_chi()
{
	bool grid_set=false;
	
	int j, jr;
	cout<<"real frequency grid file provided\n";
	
	Nw=grid_w_data.n_rows;
	vec grid_w=grid_w_data.col(0);
	if (Nw<Nw_min)
	{
		cout<<"warning: size of privided real frequency grid must be larger than "<<Nw_min<<endl;
	}
	double wmin=grid_w(0), wmax=grid_w(Nw-1);
	int j1, j2, j3;
	j1=Nw-3;
	j2=Nw-2;
	j3=Nw-1;
	w0r=(2*grid_w(j1)*grid_w(j3)-grid_w(j1)*grid_w(j2)-grid_w(j2)*grid_w(j3))/(grid_w(j1)-2*grid_w(j2)+grid_w(j3));
	if (w0r>wmax)
	{
		cout<<"The real frequency grid in file "<<grid_omega_file<<" was not generated by this code,\n";
		if (!main_spectral_region_set)
			wr=grid_w(Nw-1)-EPSILON;
		
		cout<<"only the part in the main spectral region will be used.\n";
		
		if (grid_w(0)<=0 && grid_w(Nw-1)>wr)
		{
			j=0;
			while (grid_w(j)<wr && j<Nw-1)  j++;
			jr=j;
			if ((grid_w(j)-wr)>(wr-grid_w(j-1)))  jr=j-1;
			wr=grid_w(jr);
			main_spectral_region_set=true;
			dwr=grid_w(jr)-grid_w(jr-1);
			wc=grid_w.rows(0,jr);
			wc_exists=true;
			Nwc=wc.n_rows;
			
			if (!SW_set)
			{
				SW=2*wr;
				SW_set=true;
			}
			if (!SC_set)
			{
				SC=0;
				SC_set=true;
			}
			grid_set=true;
		}
		else
		{
			cout<<"The provided real frequency grid must extend beyond the spectral function frequency range.\n";
			cout<<"The grid cannot be used.\n";
		}
	}
	else
	{
		Nw_lims.zeros(1);
		ws.zeros(1);
		ws(0)=w0r;
		w=grid_w;
		int j;
		double dw1, dw2, rdw;
		j=Nw-3;
		dw1=w(j+1)-w(j);
		dw2=w(j+2)-w(j+1);
		rdw=dw2/dw1;
		while (abs(rdw-1.0)>tol_rdw)
		{
			j=j-1;
			dw2=dw1;
			dw1=w(j+1)-w(j);
			rdw=dw2/dw1;
		}
		Nw_lims(0)=j+1;
		dwr=dw1;
		wr=w(j+1);
		wc=w.rows(0,Nw_lims(0));
		Nwc=wc.n_rows;
		main_spectral_region_set=true;
		if (!SW_set)
		{
			SW=2*wr;
			SW_set=true;
		}
		if (!SC_set)
		{
			SC=0;
			SC_set=true;
		}
		w_exists=true;
		wc_exists=true;
		grid_set=true;
		Du_constant=true;
	}
	
	return grid_set;
}

bool OmegaMaxEnt_data::set_initial_spectrum_chi()
{
	Nw_lims.zeros(1);
	ws.zeros(1);
	
	bool A0_loaded=false;
	
	cout<<"initial spectral function provided\n";
	
	w=Aw_data.col(0);
	Nw=w.n_rows;
	A0=Aw_data.col(1);
	A0=A0.rows(0,Nw-2);
	double wmin=w(0), wmax=w(Nw-1);
	if (w(0)!=0)
	{
		cout<<"set_initial_spectrum_chi(): if \"Im(G) column in data file\" is smaller than 1, the first frequency in initial spectrum file must be equal to 0\n";
		return false;
	}
	
	int j1, j2, j3;
	j1=Nw-3;
	j2=Nw-2;
	j3=Nw-1;
	w0r=(2*w(j1)*w(j3)-w(j1)*w(j2)-w(j2)*w(j3))/(w(j1)-2*w(j2)+w(j3));
	if (w0r>wmax)
	{
		cout<<"the real frequency grid in file "<<init_spectr_func_file<<" was not generated by this code\n";
		cout<<"the spectrum in that file cannot be used as the initial one in the calculation\n";
	}
	else
	{
		int j;
		double dw1, dw2, rdw;
		j=Nw-3;
		dw1=w(j+1)-w(j);
		dw2=w(j+2)-w(j+1);
		rdw=dw2/dw1;
		while (abs(rdw-1.0)>tol_rdw)
		{
			j=j-1;
			dw2=dw1;
			dw1=w(j+1)-w(j);
			rdw=dw2/dw1;
		}
		Nw_lims(0)=j+1;
		dwr=dw1;
		wr=w(j+1);
		wc=w.rows(0,Nw_lims(0));
		A0_loaded=true;
		Nwc=wc.n_rows;
		if (!SW_set)
		{
			SW=2*wr;
			SW_set=true;
		}
		if (!SC_set)
		{
			SC=0;
			SC_set=true;
		}
		ws(0)=w0r;
		
		if (displ_prep_figs)
		{
			graph_2D g1;
			
			vec x=w.rows(1,Nw-2);
			vec y=A0;
			
			char xl[]="$\\\\omega$", yl[]="A0";
			plot(g1, x, y, xl, yl);
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		w_exists=true;
		wc_exists=true;
		main_spectral_region_set=true;
		Du_constant=true;
	}
	
	return A0_loaded;
}

bool OmegaMaxEnt_data::test_low_energy_peak_chi()
{
	int Nn_min_pk=12;
	
	peak_exists=false;
	
	if (Nn<Nn_min_pk) return false;
	
	cout<<"Looking for a peak in the spectral function at low energy...\n";
	
	int nmax=n(Nn-1);
	
	int NCnmin=1;
	int NCnmax=3;
	int NNCn=NCnmax-NCnmin+1;
	ivec NCn=linspace<ivec>(NCnmin,NCnmax,NNCn);
	
	int DNwn=2;
	
	int NCpmin=1;
	int NCpmax=15;
	if (NCpmax>(nmax-NCn.max()-2))
	{
		NCpmax=nmax-NCn.max()-DNwn-1;
	}
	int NNCp=NCpmax-NCpmin+1;
	ivec NCp=linspace<ivec>(NCpmin,NCpmax,NNCp);
	
//	cout<<"NCpmax: "<<NCpmax<<endl;
//	cout<<"NNCp: "<<NNCp<<endl;
	
	int p0_min=2;
	int p0_max=2;
	int Np=p0_max-p0_min+1;
	ivec p0=linspace<ivec>(p0_min,p0_max,Np);
	
	imat NCmin(Np,2);
	
	vec M0_inc_opt(Np,fill::zeros);
	vec M2_pk_opt(Np,fill::zeros);
	vec varM2_pk_opt(Np,fill::zeros);
	
	mat X, CG, P, invCG, AM, BM, AMP, BMP, MP, Mtmp, chi2tmp;
	mat M0_inc(NNCp,NNCn), M2_pk(NNCp,NNCn), chi2_pk(NNCp,NNCn);
	int j, l, m, p, Nfit;
	uword q;
	vec Gtmp, Glftmp, diffG;
	rowvec maxX;
	double vartmp;
	
	for (q=0; q<Np; q++)
	{
		M0_inc.zeros();
		M2_pk.zeros();
		chi2_pk.zeros();
		
		for (l=0; l<NNCp; l++)
		{
			for (m=0; m<NNCn; m++)
			{
				Nfit=NCp(l)+NCn(m)+1+DNwn;
				
				//	cout<<"Nfit: "<<Nfit<<endl;
				
				X.zeros(Nfit,NCp(l)+NCn(m)+1);
				
				for (j=NCp(l); j>=-NCn(m); j--)
				{
					for (p=p0(q); p<=Nfit+p0(q)-1; p++)
					{
						X(p-p0(q),NCp(l)-j)=pow(wn(p-1),2*j);
					}
				}
				
				Gtmp=Gchi2.rows(p0(q)-1,Nfit+p0(q)-2);
				CG=COV.submat(p0(q)-1,p0(q)-1,Nfit+p0(q)-2,Nfit+p0(q)-2);
				
				invCG=inv(CG);
				//	invCG=inv_sympd(CG);
				AM=(X.t())*invCG*X;
				BM=(X.t())*invCG*Gtmp;
				
				Mtmp=solve(AM,BM);
				
				Glftmp=X*Mtmp;
				
				diffG=Gtmp-Glftmp;
				
				chi2tmp=((diffG.t())*invCG*diffG)/Nfit;
				
				chi2_pk(l,m)=chi2tmp(0,0);
				M0_inc(l,m)=-Mtmp(NCp(l));
				M2_pk(l,m)=-Mtmp(NCp(l)+1);
			}
		}
		
		uint NvM=2;
		uint NvarM=NNCp-2*NvM;
		mat varM2_pk(NvarM,NNCn,fill::zeros);
		for (j=NvM; j<NNCp-NvM; j++)
			varM2_pk.row(j-NvM)=var(M2_pk.rows(j-NvM,j+NvM));
		
		uword indpM0, indnM0;
		
		double varM2_pk_min=varM2_pk.min(indpM0,indnM0);
		
		l=indpM0+NvM;
		m=indnM0;
		
		NCmin(q,0)=l;
		NCmin(q,1)=m;
		
		M0_inc_opt(q)=M0_inc(l,m);
		M2_pk_opt(q)=M2_pk(l,m);
		varM2_pk_opt(q)=varM2_pk_min;
		
		if (displ_adv_prep_figs)
		{
			graph_2D g1, g2, g3;
			
			vec x(NNCp), y;
			
			for (l=0; l<NNCp; l++)
				x(l)=NCp(l);
			
			char xl[]="NCp";
			char yl[]="norm_peak";
			char yl1[]="Mpk_2";
			char yl2[]="chi2_LS";
			
			double xlims[2], ylims[2], ymin, ymax;
			
			xlims[0]=x.min();
			xlims[1]=x.max();
			
			char lgd_entry[100];
			
			for (m=0; m<NNCn; m++)
			{
				sprintf(lgd_entry,"NCn=%d",(int)NCn(m));
				
				y=-Gr(0)-M0_inc.col(m);
				ymax=y.max();
				ymin=y.min();
				ylims[0]=ymin-0.1*(ymax-ymin);
				ylims[1]=ymax+0.1*(ymax-ymin);
				g1.add_data(x.memptr(),y.memptr(),x.n_rows);
				g1.add_to_legend(lgd_entry);
				
				y=M2_pk.col(m);
				ymax=y.max();
				ymin=y.min();
				ylims[0]=ymin-0.1*(ymax-ymin);
				ylims[1]=ymax+0.1*(ymax-ymin);
				g2.add_data(x.memptr(),y.memptr(),x.n_rows);
				g2.add_to_legend(lgd_entry);
				
				y=chi2_pk.col(m);
				ymax=y.max();
				ymin=y.min();
				ylims[0]=ymin-0.1*(ymax-ymin);
				ylims[1]=ymax+0.1*(ymax-ymin);
				g3.add_data(x.memptr(),y.memptr(),x.n_rows);
				g3.add_to_legend(lgd_entry);
			}
			g1.set_axes_labels(xl,yl);
			g1.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			
			g2.set_axes_labels(xl,yl1);
			g2.set_axes_lims(xlims,ylims);
			g2.curve_plot();
			
			g3.set_axes_labels(xl,yl2);
			g3.set_axes_lims(xlims,ylims);
			g3.curve_plot();
			
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
			
		}
	}
	
	double varM2_min=varM2_pk_opt.min(q);
	
	double peak_weight=-Gr(0)-M0_inc_opt(q);
	double var_peak=M2_pk_opt(q)/peak_weight;
	double peak_width=0;
	if (var_peak>0)	peak_width=sqrt(var_peak);
	double m2_lf_min=M2_pk_opt(q);
	
	if (peak_width>100*EPSILON && varM2_min/M2_pk_opt(q)<varM2_peak_max && peak_weight>peak_weight_min*M1n && var_peak>0)
	{
		peak_exists=true;
		
		l=NCmin(q,0);
		m=NCmin(q,1);
		
		Nfit=NCp(l)+NCn(m)+1+DNwn;
		
		X.zeros(Nfit,NCp(l)+NCn(m)+1);
		
		for (j=NCp(l); j>=-NCn(m); j--)
		{
			for (p=p0(q); p<=Nfit+p0(q)-1; p++)
			{
				X(p-p0(q),NCp(l)-j)=pow(wn(p-1),2*j);
			}
		}
		
		Gtmp=Gchi2.rows(p0(q)-1,Nfit+p0(q)-2);
		CG=COV.submat(p0(q)-1,p0(q)-1,Nfit+p0(q)-2,Nfit+p0(q)-2);
		
		invCG=inv(CG);
		//	invCG=inv_sympd(CG);
		AM=(X.t())*invCG*X;
		AM=0.5*(AM.t()+AM);
		BM=(X.t())*invCG*Gtmp;
		
		Mtmp=solve(AM,BM);
		
		Glftmp=X*Mtmp;
		
		mat COVpeak=inv(AM);
		
		double err_norm_peak=sqrt(COVpeak(NCp(l),NCp(l)));
		double err_M2_peak=sqrt(COVpeak(NCp(l)+1,NCp(l)+1));
		double err_std_peak=(-err_norm_peak*m2_lf_min/peak_weight+err_M2_peak/peak_weight)/(2*peak_width);
		
		cout<<"Peak detected\n";
		cout<<"peak width: "<<peak_width<<endl;
		//cout<<"error on width: "<<err_std_peak<<endl;
		cout<<"peak weight: "<<peak_weight<<endl;
		//cout<<"error on weight: "<<err_norm_peak<<endl;
		
		
		if (displ_adv_prep_figs)
		{
			graph_2D g1;
			
			vec x=wn.rows(p0(q)-1,Nfit+p0(q)-2), y;
			
			char xl[]="$\\\\omega_n$";
			char yl[]="Gr";
			char lgd1[]="data";
			char lgd2[]="fit";
			char attr1[]="'o', markeredgecolor='r', markerfacecolor='none'";
			char attr2[]="'s', markeredgecolor='b', markerfacecolor='none'";
			
			double xlims[2], ylims[2], ymin, ymax;
			
			xlims[0]=x.min();
			xlims[1]=x.max();
			y=Gr.rows(p0(q)-1,Nfit+p0(q)-2);
			ymax=y.max();
			ymin=y.min();
			ylims[0]=ymin-0.1*(ymax-ymin);
			ylims[1]=ymax+0.1*(ymax-ymin);
			g1.add_data(x.memptr(),y.memptr(),Nfit);
			g1.add_to_legend(lgd1);
			g1.add_attribute(attr1);
			y=Glftmp.rows(0,Nfit-1);
			g1.add_data(x.memptr(),y.memptr(),Nfit);
			g1.add_to_legend(lgd2);
			g1.add_attribute(attr2);
			g1.set_axes_labels(xl,yl);
			g1.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		
		dw_peak=peak_width/2.0;
	}
	else
	{
		cout<<"no peak found\n";
	}
	
	return peak_exists;
}

bool OmegaMaxEnt_data::compute_moments_chi_omega_n()
{
	int j, NC=3;
	
	cout<<"COMPUTING MOMENTS\n";
	
	vec C2b(Nn-2), C4b(Nn-2);
	double wn1, wn2, reG1, reG2, denom2;
	for (j=1; j<Nn-1; j++)
	{
		wn1=wn(j);
		wn2=wn(j+1);
		reG1=Gr(j);
		reG2=Gr(j+1);
		denom2=wn1*wn1 - wn2*wn2;
		C2b(j-1)=-(reG1*pow(wn1,4) - reG2*pow(wn2,4))/denom2;
		C4b(j-1)=(-reG1*pow(wn1,4)*pow(wn2,2) + reG2*pow(wn1,2)*pow(wn2,4))/denom2;
	}
	
	if (displ_adv_prep_figs)
	{
		graph_2D g1;
		
		vec x=wn.rows(1,Nn-2);
		char xl[]="$\\\\omega_n$";
		char yl[]="$M_1$";
		
		plot(g1, x, C2b, xl, yl);
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	int p, Nfitmin, jfitmin, jfitmax, NNfit, Nfit;
	jfitmin=2;
	Nfitmin=2*NC+4;
	jfitmax=Nn-Nfitmin;
	NNfit=jfitmax-jfitmin+1;
	
//	cout<<"jfitmax: "<<jfitmax<<endl;
	
	if (jfitmax<jfitmin) return false;
	
	vec M1v(NNfit), M3v(NNfit);
	mat invCG, A, X, CG;
	vec Mtmp;
	
	//	char UPLO='U';
	//	int NA=NC, NRHS=1, INFO;
	
//	mat LC, invLC;
	for (jfit=jfitmin; jfit<=jfitmax; jfit++)
	{
		Nfit=Nn_fit_max;
		if ((Nn-jfit+1)<Nn_fit_max)
			Nfit=Nn-jfit+1;
		
		X.zeros(Nfit,NC);
		
		for (j=1; j<=NC; j++)
		{
			for (p=jfit; p<=jfit+Nfit-1; p++)
			{
				X((p-jfit),j-1)=pow(-1,j)/pow(wn(p-1),2*j);
			}
		}
		
		CG=COV.submat(jfit-1,jfit-1,jfit+Nfit-2,jfit+Nfit-2);
		
		//		if (jfit==jfitmin) cout<<CG.submat(0,0,10,10);
		
		//		LC=chol(CG);
		//		invLC=inv(LC);
		//		invCG=invLC*invLC.t();
		
		invCG=inv(CG);
//		invCG=inv_sympd(CG);
		A=trans(X)*invCG*X;
		A=0.5*(A+A.t());
		Mtmp=trans(X)*invCG*Gchi2.rows(jfit-1,jfit+Nfit-2);
		
		//		dposv_(&UPLO, &NA, &NRHS, A.memptr(), &NA, Mtmp.memptr(), &NA, &INFO);
		Mtmp=solve(A,Mtmp);
		
		M1v(jfit-jfitmin)=Mtmp(0);
		M3v(jfit-jfitmin)=Mtmp(1);
		
	}
	
	int Nv=Nn/16;
	if (Nv<2) Nv=2;
	
	vec varM1(NNfit-2*Nv), varM3(NNfit-2*Nv);
	
	for (j=Nv; j<NNfit-Nv; j++)
	{
		varM1(j-Nv)=var(M1v.rows(j-Nv,j+Nv));
		varM3(j-Nv)=var(M3v.rows(j-Nv,j+Nv));
	}
	
	if (displ_adv_prep_figs)
	{
		graph_2D g1, g2;
		
		vec x=wn.rows(Nv,NNfit-Nv-1);
		
		char xl[]="$\\\\omega_n$";
		char yl1[]="varM1";
		char yl2[]="varM3";
		
		plot(g1, x, varM1, xl, yl1);
		plot(g2, x, varM3, xl, yl2);
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	uword j1, j3;
	varM1.min(j1);
	varM3.min(j3);
	
	j1=j1+Nv;
	j3=j3+Nv;
	
	Mfit.zeros(2);
	Mfit(0)=mean(M1v.rows(j1-Nv,j1+Nv));
	Mfit(1)=mean(M3v.rows(j3-Nv,j3+Nv));
	
//	cout<<"frequency range used to determine the moments: "<<wn(j1+jfitmin-Nv-1)<<" to "<<wn(j1+jfitmin+Nv-1)<<" (indices "<<j1+jfitmin-Nv-1<<" to "<<j1+jfitmin+Nv-1<<")"<<endl;
	
	int jfit0;
	j=Nv;
	while (j<j1 && (abs(mean(M1v.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(M1v.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) && j<Nn-Nv-3) j++;
	jfit0=j;
	
	/*
	int jfit0;
	j=Nv;
	while ((abs(mean(C2b.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(C2b.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) && j<Nn-Nv-3) j++;
	jfit0=j;
	*/
	
	jfit=j1+jfitmin-1;
	Nfit=Nn_fit_fin;
	if (Nfit>Nn-jfit+1)
		Nfit=Nn-jfit+1;
	
	X.zeros(Nfit,NC);
	
	for (j=1; j<=NC; j++)
	{
		for (p=jfit; p<=jfit+Nfit-1; p++)
		{
			X((p-jfit),j-1)=pow(-1,j)/pow(wn(p-1),2*j);
		}
	}
	
	CG=COV.submat(jfit-1,jfit-1,jfit+Nfit-2,jfit+Nfit-2);
	
	//	LC=chol(CG);
	//	invLC=inv(LC);
	//	invCG=invLC*invLC.t();
	
	invCG=inv(CG);
//		invCG=inv_sympd(CG);
	A=trans(X)*invCG*X;
	A=0.5*(A+A.t());
	mat COVMtmp=inv(A);
	COVMfit=COVMtmp.submat(0,0,0,0);
	
//	jfit=jfit0;
	jfit=jfit0+jfitmin-1;
	
	cout<<"frequency range of asymptotic behavior: "<<wn(jfit-1)<<" to "<<wn(Nn-1)<<" (indices "<<jfit-1<<" to "<<Nn-1<<")"<<endl;
	
	cout<<"1st moment extracted from high frequencies: "<<Mfit(0)<<endl;
	cout<<"3rd moment extracted from high frequencies: "<<Mfit(1)<<endl;
	
	double std_omega_tmp;
	double var_omega;
		
	var_omega=Mfit(0)/M1n;
	if (var_omega>0)
		std_omega_tmp=sqrt(var_omega);
	else
	{
		cout<<"Negative variance found during computation of moments.\n";
		return false;
	}
		
	if (!moments_provided)
	{
		M=Mfit.rows(0,0);
		NM=1;
		COVM=COVMfit.submat(0,0,0,0);
		M1=M(0);
		covm_diag=true;
		M1_set=true;
		M_ord.zeros(1);
		M_ord(0)=1;
		errM.zeros(NM);
		errM(0)=sqrt(COVM(0,0));
	}
	else if (abs((M1-Mfit(0))/Mfit(0))>tol_M1)
		cout<<"warning: first moment different from provided one\n";
	
	if (!std_omega)
	{
		var_omega=M1/M1n;
		if (var_omega>0)
			std_omega=sqrt(var_omega);
		else
		{
			cout<<"Negative variance found during computation of moments.\n";
			return false;
		}
	}
	if (!SC_set)
	{
		SC=0;
		SC_set=true;
	}
	if (!SW_set)
	{
		SW=f_SW_std_omega*std_omega;
		SW_set=true;
	}
	
	if (Nn-jfit<Nn_as_min)
	{
		jfit=0;
		if (!moments_provided) maxM=0;
	}
	
	//	if (displ_adv_prep_figs)
	if (displ_prep_figs)
	{
		graph_2D g1, g2;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="$\\\\omega_n$";
		char yl1[]="$M_1$";
		char yl2[]="$M_3$";
		
		plot(g1, x, M1v, xl, yl1);
		plot(g2, x, M3v, xl, yl2);
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	return true;
}

bool OmegaMaxEnt_data::set_covar_chi_omega_n()
{
//	cout<<"Nsmooth_errG: "<<Nsmooth_errG<<endl;
//	cout<<"wgt_min_sm: "<<wgt_min_sm<<endl;
	
	error_provided=false;
	
	COV.zeros(Nn,Nn);
	cov_diag=true;
	if (error_file.size())
	{
		if (error_data.n_rows<Nn)
		{
			cout<<"number of lines is too small in file "<<error_file<<endl;
			return false;
		}
		cout<<"error file provided\n";
		errGr=error_data.col(col_errGr-1);
		if (wn_sign_change)
		{
			errGr=errGr.rows(indG_0,indG_f);
		}
		if (wn_inverted)
		{
			errGr=flipud(errGr);
		}
		errG=errGr;
		COV.diag()=square(errG);
		error_provided=true;
	}
	else if ( covar_re_re_file.size() )
	{
		if (CRR.n_rows<Nn || CRR.n_cols<Nn)
		{
			cout<<"number of lines and/or columns in covariance file(s) is too small\n";
			return false;
		}
		cout<<"covariance matrix provided\n";
		cov_diag=false;
		COV=CRR.submat(0,0,Nn-1,Nn-1);
		error_provided=true;
	}
	else if (N_params_noise)
	{
		int j,k;
		
		cout<<"added noise relative error: "<<noise_params(ind_noise)<<endl;
		
		errGr=noise_params(ind_noise)*abs(Gr);
		
		if (Nsmooth_errG)
		{
			double b=-log(wgt_min_sm)/Nsmooth_errG;
			
			vec wgt_sm(2*Nsmooth_errG+1);
			for (j=0; j<Nsmooth_errG; j++) wgt_sm(j)=exp(-b*(Nsmooth_errG-j));
			wgt_sm(Nsmooth_errG)=1;
			for (j=Nsmooth_errG+1; j<=2*Nsmooth_errG; j++) wgt_sm(j)=wgt_sm(2*Nsmooth_errG-j);
			wgt_sm=wgt_sm/sum(wgt_sm);
		//	cout<<"wgt_sm:\n";
		//	for (j=0; j<=2*Nsmooth_errG; j++) cout<<setw(20)<<wgt_sm(j);
		//	cout<<endl;
		//	cout<<"sum(wgt_sm): "<<sum(wgt_sm)<<endl;
			
			vec errG_tmp=errGr;
			
			double wgt_tmp;
			for (j=0; j<Nsmooth_errG; j++)
			{
				errGr(j)=0;
				wgt_tmp=0;
				for (k=Nsmooth_errG-j; k<=2*Nsmooth_errG; k++)
				{
					errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGr(j)=errGr(j)/wgt_tmp;
			}
			for (j=Nsmooth_errG; j<Nn-Nsmooth_errG; j++)
			{
				errGr(j)=0;
				for (k=0; k<=2*Nsmooth_errG; k++)
				{
					errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
				}
			}
			for (j=Nn-Nsmooth_errG; j<Nn; j++)
			{
				errGr(j)=0;
				wgt_tmp=0;
				for (k=0; k<=Nsmooth_errG+Nn-1-j; k++)
				{
					errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGr(j)=errGr(j)/wgt_tmp;
			}
			/*
			vec errG_tmp(Nn+2*Nsmooth_errG);
			errG_tmp.rows(Nsmooth_errG,Nn+Nsmooth_errG-1)=errGr;
			errG_tmp.rows(0,Nsmooth_errG-1)=flipud(errGr.rows(1,Nsmooth_errG));
			errG_tmp.rows(Nn+Nsmooth_errG,Nn+2*Nsmooth_errG-1)=flipud(errGr.rows(Nn-Nsmooth_errG-1,Nn-2));
			for (j=0; j<Nn; j++)
			{
				errGr(j)=0;
				for (k=0; k<=2*Nsmooth_errG; k++)
				{
					errGr(j)+=wgt_sm(k)*errG_tmp(j+k);
				}
			}
			*/
		}
		
		errG=errGr;
		COV.diag()=square(errG);
	}
	else
	{
		cout<<"no errors provided\nusing a constant error\n";
		
		double Gr_max=max(abs(Gr));
		errGr=default_error_G*Gr_max*ones<vec>(Nn);
		errG=errGr;
		COV.diag()=square(errG);
	}
	
	
	if (displ_prep_figs && cov_diag)
	{
		graph_2D g1;
		
		char ttl[]="error on $G$";
		char xl[]="$\\\\omega_n$";
		char yl[]="$\\\\sigma_G$";
		char attr[]="'o-', markeredgecolor='b', markerfacecolor='b'";
		
		g1.add_title(ttl);
		plot(g1, wn, errGr, xl, yl, attr);
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	COV=0.5*(COV+COV.t());
	
	return true;
}

bool OmegaMaxEnt_data::compute_moments_tau_bosons()
{
	cout<<"COMPUTING MOMENTS\n";
	
	int Nfitmax_max=50;
	int Nv=1;
	int NvN=1;
	int npmin=2;
	int npmax=5;
	int Np=npmax-npmin+1;
	int DNfitmin=0;
	int DNfitmax=Ntau-npmax-1;
	if (16-npmax-1<DNfitmax) DNfitmax=16-npmax-1;
	int NDN=DNfitmax-DNfitmin+1;
	
	mat M0tmp=zeros<mat>(NDN,Np);
	mat M1tmp=zeros<mat>(NDN,Np);
	mat M2tmp=zeros<mat>(NDN,Np);
	
	vec Gtmp, pp;
	int DNfit, np, Nfit;
	for (DNfit=DNfitmin; DNfit<=DNfitmax; DNfit++)
	{
		for (np=npmin; np<=npmax; np++)
		{
			Nfit=np+1+DNfit;
			Gtmp=Gtau.rows(0,Nfit-1)+flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			polyfit(tau.rows(0,Nfit-1),Gtmp,np,0,pp);
			M1tmp(DNfit-DNfitmin,np-npmin)=pp(np-1);
			
			Gtmp=Gtau.rows(0,Nfit-1)-flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			polyfit(tau.rows(0,Nfit-1),Gtmp,np,0,pp);
			M0tmp(DNfit-DNfitmin,np-npmin)=-pp(np);
			M2tmp(DNfit-DNfitmin,np-npmin)=-2*pp(np-2);
		}
	}
	
	mat M0m=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat M1m=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat M2m=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat varM0=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat varM1=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat varM2=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	int j, l;
	for (l=Nv; l<Np-Nv; l++)
	{
		for (j=NvN; j<NDN-NvN; j++)
		{
			M0m(j-NvN,l-Nv)=accu(M0tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M1m(j-NvN,l-Nv)=accu(M1tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M2m(j-NvN,l-Nv)=accu(M2tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			varM0(j-NvN,l-Nv)=accu(pow(M0tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M0m(j-NvN,l-Nv),2))/((2*Nv+1)*(2*NvN+1));
			varM1(j-NvN,l-Nv)=accu(pow(M1tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M1m(j-NvN,l-Nv),2))/((2*Nv+1)*(2*NvN+1));
			varM2(j-NvN,l-Nv)=accu(pow(M2tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M2m(j-NvN,l-Nv),2))/((2*Nv+1)*(2*NvN+1));
		}
	}
	
	uword jvmin, lvmin;
	varM0.min(jvmin,lvmin);
	double M0_NP_tmp=M0m(jvmin,lvmin);
	varM1.min(jvmin,lvmin);
	double M1_NP_tmp=M1m(jvmin,lvmin);
	varM2.min(jvmin,lvmin);
	double M2_NP_tmp=M2m(jvmin,lvmin);
	
//	cout<<"M0_NP_tmp: "<<M0_NP_tmp<<endl;
//	cout<<"M1_NP_tmp: "<<M1_NP_tmp<<endl;
//	cout<<"M2_NP_tmp: "<<M2_NP_tmp<<endl;
	
	double Wtmp;
	if (M1_NP_tmp/M1n>pow(M0_NP_tmp/M1n,2))
		Wtmp=sqrt(M1_NP_tmp/M1n-pow(M0_NP_tmp/M1n,2));
	else
		Wtmp=abs(M0_NP_tmp/M1n);
	
	int Nfitmax=ceil(FNfitTauW*Ntau*tem/(abs(M0_NP_tmp/M1n)+Wtmp));
	if (Nfitmax>Ntau/2) Nfitmax=Ntau/2;
	if (Nfitmax>Nfitmax_max) Nfitmax=Nfitmax_max;
	
	mat X;
	int p;
	
//	cout<<"Nfitmax: "<<Nfitmax<<endl;
	
	Nfit=Nfitmax;
	np=Nfit-1;
	X.zeros(Nfit,np+1);
	for (p=0; p<=np; p++)
		X.col(p)=pow(tau.rows(0,Nfit-1)/tau(Nfit-1),p);
	
	mat U, V;
	vec sK;
	svd(U,sK,V,X,"std");
	
//	cout<<"sK(np)/sK(0): "<<sK(np)/sK(0)<<endl;
	
	p=0;
	while (p<=np && sK(p)/sK(0)>R_sv_min) p++;
	Nfitmax=p;
	
//	cout<<"Nfitmax: "<<Nfitmax<<endl;
	
	double M0_N, M1_N, M2_N, M3_N;
	
	mat CG, invCG, AM;
	vec Gchi2tmp, BM, Mtmp;
	npmin=3;
	int Nfitmin=npmin+1;
	int NNfit=Nfitmax-Nfitmin+1;
	
	if (NNfit<5)
	{
		cout<<"compute_moments_tau_bosons(): unable to compute the moments from G(tau). The imaginary time step can be either too small or too large. You can either change the step, provide the first moment, or increase parameter R_sv_min in file \"OmegaMaxEnt_other_params.dat\".\n";
		return false;
	}
	
	mat M0b=zeros<mat>(NNfit,NNfit);
	mat M1b=zeros<mat>(NNfit,NNfit);
	mat M2b=zeros<mat>(NNfit,NNfit);
	mat M3b=zeros<mat>(NNfit,NNfit);

	//		char UPLO='U';
	//		int NA, NRHS=1, INFO;
	
	for (Nfit=Nfitmin; Nfit<=Nfitmax; Nfit++)
	{
		for (np=npmin; np<Nfit; np++)
		{
	//		cout<<"np= "<<np<<endl;
			X=zeros<mat>(Nfit,np+1);
			for (p=0; p<=np; p++)
				X.col(p)=pow(tau.rows(0,Nfit-1)/tau(Nfit-1),p);
			
			Gchi2tmp=Gtau.rows(0,Nfit-1)+flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)+fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))+flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
			invCG=inv(CG);
//			invCG=inv_sympd(CG);
			AM=(X.t())*invCG*X;
			BM=(X.t())*invCG*Gchi2tmp;
//			Mtmp=solve(AM,BM);
			
//			cout<<"cond(AM): "<<cond(AM)<<endl;
			
			if (!solve(Mtmp,AM,BM))
			{
				cout<<"compute_moments_tau_bosons(): solve() failed\n";

		//		Mtmp=BM;
		//		NA=np+1;
		//		dposv_(&UPLO, &NA, &NRHS, AM.memptr(), &NA, Mtmp.memptr(), &NA, &INFO);
		//		cout<<"INFO: "<<INFO<<endl;
				
				return false;
			}
		
			M1b(Nfit-np-1,np-npmin)=Mtmp(1)/tau(Nfit-1);
			M3b(Nfit-np-1,np-npmin)=6*Mtmp(3)/pow(tau(Nfit-1),3);
			
			Gchi2tmp=Gtau.rows(0,Nfit-1)-flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)- fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))-flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
			invCG=inv(CG);
//			invCG=inv_sympd(CG);
			AM=(X.t())*invCG*X;
			BM=(X.t())*invCG*Gchi2tmp;
		//	Mtmp=solve(AM,BM);
			if (!solve(Mtmp,AM,BM))
			{
				cout<<"compute_moments_tau_bosons(): solve() failed\n";
				
	//			Mtmp=BM;
	//			NA=np+1;
	//			dposv_(&UPLO, &NA, &NRHS, AM.memptr(), &NA, Mtmp.memptr(), &NA, &INFO);
	//			cout<<"INFO: "<<INFO<<endl;
			//	cout<<"cond(invCG): "<<cond(invCG)<<endl;
				return false;
			}
			M0b(Nfit-np-1,np-npmin)=-Mtmp(0);
			M2b(Nfit-np-1,np-npmin)=-2*Mtmp(2)/pow(tau(Nfit-1),2);
		}
	}
	
	Nv=1;
	NvN=1;
	int jmin=NvN;
	int jmax=NNfit-NvN-2*Nv-1;
	int Nj=jmax-jmin+1;
	int lmin=Nv;
	int lmax=NNfit-2*NvN-Nv-1;
	int Nl=lmax-lmin+1;
	M0m.zeros(Nj,Nl);
	M1m.zeros(Nj,Nl);
	M2m.zeros(Nj,Nl);
	mat M3m=zeros<mat>(Nj,Nl);
	varM0.zeros(Nj,Nl);
	varM1.zeros(Nj,Nl);
	varM2.zeros(Nj,Nl);
	mat varM3=zeros<mat>(Nj,Nl);
	for (j=jmin; j<=jmax; j++)
	{
		for (l=lmin; l<=NNfit-1-j-Nv-NvN; l++)
		{
			M0m(j-jmin,l-lmin)=accu(M0b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M1m(j-jmin,l-lmin)=accu(M1b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M2m(j-jmin,l-lmin)=accu(M2b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M3m(j-jmin,l-lmin)=accu(M3b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			varM0(j-jmin,l-lmin)=accu(pow(M0b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M0m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
			varM1(j-jmin,l-lmin)=accu(pow(M1b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M1m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
			varM2(j-jmin,l-lmin)=accu(pow(M2b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M2m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
			varM3(j-jmin,l-lmin)=accu(pow(M3b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M3m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
		}
	}
	
	//	cout<<varM0<<endl;
	
	double varM0max=max(max(varM0));
	double varM1max=max(max(varM1));
	double varM2max=max(max(varM2));
	double varM3max=max(max(varM3));
	
	for (j=1; j<Nj; j++)
	{
		varM0.submat(j,Nl-j,j,Nl-1)=2*varM0max*ones<rowvec>(j);
		varM1.submat(j,Nl-j,j,Nl-1)=2*varM1max*ones<rowvec>(j);
		varM2.submat(j,Nl-j,j,Nl-1)=2*varM2max*ones<rowvec>(j);
		varM3.submat(j,Nl-j,j,Nl-1)=2*varM3max*ones<rowvec>(j);
	}
	
	varM0.min(jvmin,lvmin);
	M0_N=M0m(jvmin,lvmin);
	varM1.min(jvmin,lvmin);
	M1_N=M1m(jvmin,lvmin);
	varM2.min(jvmin,lvmin);
	M2_N=M2m(jvmin,lvmin);
	varM3.min(jvmin,lvmin);
	M3_N=M3m(jvmin,lvmin);
	
	cout<<"moments determined by polynomial fit to G(tau) at boundaries:\n";
	cout<<"norm: "<<M0_N<<endl;
	cout<<"first moment: "<<M1_N<<endl;
	cout<<"second moment: "<<M2_N<<endl;
	cout<<"third moment: "<<M3_N<<endl;
	
	
	if (displ_adv_prep_figs)
	{
		graph_2D g1, g2;
		char lgd_format[]="Nfit=%d", lgd_entry[20];
		
		vec pfit=linspace<vec>(npmin,Nfitmax-1,NNfit);
		vec vtmp=M0b.col(0);
		g1.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
		sprintf(lgd_entry,lgd_format,Nfitmin);
		g1.add_to_legend(lgd_entry);
		for (j=1; j<NNfit; j++)
		{
			vtmp=M0b.col(j);
			g1.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
			sprintf(lgd_entry,lgd_format,Nfitmin+j);
			g1.add_to_legend(lgd_entry);
		}
		g1.add_title("$M_0$");
		g1.curve_plot();
		
		vtmp=M1b.col(0);
		g2.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
		sprintf(lgd_entry,lgd_format,Nfitmin);
		g2.add_to_legend(lgd_entry);
		for (j=1; j<NNfit; j++)
		{
			vtmp=M1b.col(j);
			g2.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
			sprintf(lgd_entry,lgd_format,Nfitmin+j);
			g2.add_to_legend(lgd_entry);
		}
		g2.add_title("$M_1$");
		g2.curve_plot();
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
		
	}

	varM1.min(jvmin,lvmin);
	np=lvmin+Nv+2;
	Nfit=jvmin+NvN+np+1;
	X=zeros<mat>(Nfit,np+1);
	for (p=0; p<=np; p++)
		X.col(p)=pow(tau.rows(0,Nfit-1),p);
	
	CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)+fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))+flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
//	CG(0,0)=CG(1,1);
	invCG=inv(CG);
//	invCG=inv_sympd(CG);
	AM=(X.t())*invCG*X;
	mat invAMp=inv(AM);
	
	CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)-fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))-flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
	invCG=inv(CG);
	AM=(X.t())*invCG*X;
	mat invAMn=inv(AM);
	
	double std_omega_tmp;
	double var_omega=M1_N/M1n-pow(M0_N/M1n,2);
	if (var_omega>0)
		std_omega_tmp=sqrt(var_omega);
	else
	{
		cout<<"Negative variance found during computation of moments.\n";
		return false;
	}
	
	covm_diag=true;
	if (!moments_provided)
	{
		if (col_Gi>0)
		{
			M.zeros(3);
			NM=3;
			M(0)=M0;
			M(1)=M1_N;
			M(2)=M2_N;
			M1=M(1);
			M2=M(2);
			M1_set=true;
			M2_set=true;
			errM.zeros(NM);
			errM(0)=errM0;
			errM(1)=sqrt(invAMp(1,1));
			errM(2)=sqrt(invAMn(2,2));
			errM1=errM(1);
			errM2=errM(2);
			M_ord=linspace<vec>(0,NM-1,NM);
		}
		else
		{
			M.zeros(1);
			NM=1;
			M1=M1_N;
			M(0)=M1;
			errM.zeros(NM);
			errM1=sqrt(invAMp(1,1));
			errM(0)=errM1;
			M_ord.zeros(1);
			M_ord(0)=1;
		}
	}
	else
	{
		if (abs(M1-M1_N)/M1_N>tol_M1)
			cout<<"warning: first moment different from provided one\n";
		
		if (col_Gi>0)
		{
			if (M2_in.size())
			{
				if (abs(M2-M2_N)/pow(std_omega_tmp,2)>tol_M2)
					cout<<"warning: second moment different from provided one\n";
			}
			else
			{
				M2=M2_N;
				M2_set=true;
				errM2=sqrt(invAMn(2,2));
				M.zeros(3);
				NM=3;
				M(0)=M0;
				M(1)=M1;
				M(2)=M2;
				errM.zeros(NM);
				errM(0)=errM0;
				errM(1)=errM1;
				errM(2)=errM2;
				M_ord=linspace<vec>(0,NM-1,NM);
			}
		}
	}
	COVM.zeros(NM,NM);
	COVM.diag()=square(errM);
	
	dG_dtau_computed=compute_dG_dtau();
	
	return true;
}

bool OmegaMaxEnt_data::Kernel_G_bosons()
{
	bool use_HF_exp=true;
	double fg=1.7;
	double fi=fg;
	int pnmax=100;
	
	cout<<"defining kernel matrix...\n";
	
	Kcx.zeros(Nn,Nw);
	KM.zeros(3,Nw);
 
	int Nug=Nw_lims(0);
	int Nud=Nw-Nw_lims(1)-1;
 
	vec ud, ug;
	if (Du_constant)
	{
		double dug=dwl/((wl-dwl-w0l)*(wl-w0l));
		vec ug_int=linspace<vec>(1,Nug,Nug);
		ug=-dug*ug_int;
		
		double dud=dwr/((wr-w0r)*(wr+dwr-w0r));
		vec ud_int=linspace<vec>(Nud,1,Nud);
		ud=dud*ud_int;
	}
	else
	{
		ug=1.0/(w.rows(0,Nug-1)-w0l);
		ud=1.0/(w.rows(Nw_lims(1)+1,Nw-1)-w0r);
	}
 
	mat MM;
	spline_matrix_G_part(w, Nw_lims, ws, MM);
	
	int Ncfs0=MM.n_rows;
	
	mat MC(Ncfs0+Nw,Nw);
	
	MC.submat(0,0,Ncfs0-1,Nw-1)=MM;
	MC.submat(Ncfs0,0,Ncfs0+Nw-1,Nw-1)=eye<mat>(Nw,Nw);
	
	int Nint=Nw+1;
	int Nintg=Nug+1;
	
	mat Pa_g=zeros<mat>(Nintg,3*Nint-2+Nw);
	mat Pb_g_r=zeros<mat>(Nintg,3*Nint-2+Nw);
	mat Pc_g_r=zeros<mat>(Nintg,3*Nint-2+Nw);
	mat Pd_g_r=zeros<mat>(Nintg,3*Nint-2+Nw);
	
	Pa_g(0,0)=1;
	Pb_g_r(0,1)=1;
	
	int j;
	for (j=2; j<=Nintg; j++)
	{
		Pa_g(j-1,3*j-4)=1;
		Pb_g_r(j-1,3*j-3)=1;
		Pc_g_r(j-1,3*j-2)=1;
		Pd_g_r(j-1,j+3*Nint-4)=1;
	}
	
	mat U=zeros<mat>(Nug+1,Nug+1);
	vec ug1=zeros<vec>(Nug+1);
	ug1.rows(1,Nug)=ug;
	
	U.diag()=ug1;
	
	mat Pb_g=Pb_g_r-3*U*Pa_g;
	mat Pc_g=Pc_g_r+3*pow(U,2)*Pa_g-2*U*Pb_g_r;
	mat Pd_g=Pd_g_r-pow(U,3)*Pa_g+pow(U,2)*Pb_g_r-U*Pc_g_r;
	
	int	Nintc=Nwc-1;
	
	mat	Pa_c=zeros<mat>(Nintc,3*Nint-2+Nw);
	mat	Pb_c_r=zeros<mat>(Nintc,3*Nint-2+Nw);
	mat	Pc_c_r=zeros<mat>(Nintc,3*Nint-2+Nw);
	mat	Pd_c_r=zeros<mat>(Nintc,3*Nint-2+Nw);
	
	for (j=1; j<=Nintc; j++)
	{
		Pa_c(j-1,3*j-4+3*Nintg)=1;
		Pb_c_r(j-1,3*j-3+3*Nintg)=1;
		Pc_c_r(j-1,3*j-2+3*Nintg)=1;
		Pd_c_r(j-1,j+3*Nint-3+Nug)=1;
	}
	
	mat W=diagmat(wc.rows(0,Nwc-2));
	
	mat Pb_c=Pb_c_r-3*W*Pa_c;
	mat Pc_c=Pc_c_r+3*pow(W,2)*Pa_c-2*W*Pb_c_r;
	mat Pd_c=Pd_c_r-pow(W,3)*Pa_c+pow(W,2)*Pb_c_r-W*Pc_c_r;
	
	int Nintd=Nud+1;
	
	mat Pa_d=zeros<mat>(Nintd,3*Nint-2+Nw);
	mat Pb_d_r=zeros<mat>(Nintd,3*Nint-2+Nw);
	mat Pc_d_r=zeros<mat>(Nintd,3*Nint-2+Nw);
	mat Pd_d_r=zeros<mat>(Nintd,3*Nint-2+Nw);
	
	for (j=1; j<Nintd; j++)
	{
		Pa_d(j-1,3*j-4+3*Nintg+3*Nintc)=1;
		Pb_d_r(j-1,3*j-3+3*Nintg+3*Nintc)=1;
		Pc_d_r(j-1,3*j-2+3*Nintg+3*Nintc)=1;
		Pd_d_r(j-1,j+3*Nint-3+Nug+Nwc)=1;
	}
	
	j=Nintd;
	Pa_d(j-1,3*j-4+3*Nintg+3*Nintc)=1;
	Pb_d_r(j-1,3*j-3+3*Nintg+3*Nintc)=1;
	
	U.zeros(Nud+1,Nud+1);
	vec ud1=zeros<vec>(Nud+1);
	ud1.rows(0,Nud-1)=ud;
	U.diag()=ud1;
	
	mat Pb_d=Pb_d_r-3*U*Pa_d;
	mat Pc_d=Pc_d_r+3*pow(U,2)*Pa_d-2*U*Pb_d_r;
	mat Pd_d=Pd_d_r-pow(U,3)*Pa_d+pow(U,2)*Pb_d_r-U*Pc_d_r;

	cx_mat Ka_g=zeros<cx_mat>(Nn,Nintg);
 	cx_mat Kb_g=zeros<cx_mat>(Nn,Nintg);
 	cx_mat Kc_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kd_g=zeros<cx_mat>(Nn,Nintg);
 
 	rowvec ug2(Nug+1);
	ug2.cols(0,Nug-1)=ug.t();
	ug2(Nug)=1.0/(wl-w0l);
	
	mat Wng=wn.rows(1,Nn-1)*ones<rowvec>(Nintg-1);
	mat Ug=ones<vec>(Nn-1)*ug2;
	
	dcomplex i(0,1);
 
	rowvec wg=1/ug2+w0l;
	mat Wg=ones<vec>(Nn-1)*wg;
	
	cx_rowvec cx_vec_tmp(Nug);
	cx_vec_tmp.zeros();

	rowvec vec_tmp=-(pow(ug2.cols(1,Nintg-1),2)-pow(ug2.cols(0,Nintg-2),2))/(4*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Ka_g.submat(0,1,0,Nintg-1)=cx_vec_tmp;
	vec_tmp=-(ug2.cols(1,Nintg-1)-ug2.cols(0,Nintg-2))/(2*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kb_g.submat(0,1,0,Nintg-1)=cx_vec_tmp;
	vec_tmp=-log(ug2.cols(1,Nintg-1)/ug2.cols(0,Nintg-2))/(2*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kc_g.submat(0,1,0,Nintg-1)=cx_vec_tmp;
	vec_tmp=trans(w.rows(1,Nintg-1)-w.rows(0,Nintg-2))/(2*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kd_g.submat(0,1,0,Nintg-1)=cx_vec_tmp;
	
// 	Ka_g.submat(0,1,0,Nintg-1)=-(pow(ug2.cols(1,Nintg-1),2)-pow(ug2.cols(0,Nintg-2),2))/(4*PI);
// 	Kb_g.submat(0,1,0,Nintg-1)=-(ug2.cols(1,Nintg-1)-ug2.cols(0,Nintg-2))/(2*PI);
// 	Kc_g.submat(0,1,0,Nintg-1)=-log(ug2.cols(1,Nintg-1)/ug2.cols(0,Nintg-2))/(2*PI);
// 	Kd_g.submat(0,1,0,Nintg-1)=(w.rows(1,Nintg-1)-w.rows(0,Nintg-2))/(2*PI);


 	mat atang=atan((Wng % (Ug.cols(1,Nintg-1)-Ug.cols(0,Nintg-2)))/(1+w0l*(Ug.cols(1,Nintg-1)+Ug.cols(0,Nintg-2))+(pow(w0l,2)+pow(Wng,2)) % Ug.cols(0,Nintg-2) % Ug.cols(1,Nintg-1)));
	mat logg=log((1+2*w0l*Ug.cols(1,Nintg-1)+pow(Ug.cols(1,Nintg-1),2) % (pow(Wng,2)+pow(w0l,2)))/(1+2*w0l*Ug.cols(0,Nintg-2)+pow(Ug.cols(0,Nintg-2),2) % (pow(Wng,2)+pow(w0l,2))));
	
 	Ka_g.submat(1,1,Nn-1,Nintg-1)=(-i*(2*Wng % (Ug.cols(1,Nintg-1)-Ug.cols(0,Nintg-2)))/pow(Wng + i*w0l,2) - i*w0l*(pow(Ug.cols(1,Nintg-1),2)-pow(Ug.cols(0,Nintg-2),2))/(Wng + i*w0l) +  i*(2*Wng % atang)/pow(Wng + i*w0l,3) - (Wng % logg)/pow(Wng + i*w0l,3) )/(4*PI);
 	Kb_g.submat(1,1,Nn-1,Nintg-1)=(-i*2.0*w0l*(Ug.cols(1,Nintg-1)-Ug.cols(0,Nintg-2))/(Wng + i*w0l) -(2*Wng % atang)/pow(Wng + i*w0l,2) -i*(Wng % logg)/pow(Wng + i*w0l,2) )/(4*PI);
 	Kc_g.submat(1,1,Nn-1,Nintg-1)=( -2*log(Ug.cols(1,Nintg-1)/Ug.cols(0,Nintg-2)) - i*(2*Wng % atang)/(Wng + i*w0l) + (Wng % logg)/(Wng + i*w0l) )/(4*PI);
 	Kd_g.submat(1,1,Nn-1,Nintg-1)=( 2*(Wg.cols(1,Nintg-1)-Wg.cols(0,Nintg-2)) -i*2.0*Wng % log(Ug.cols(1,Nintg-1)/Ug.cols(0,Nintg-2)) + 2*Wng % atang + i*Wng % logg )/(4*PI);
	

	cx_mat Ka_c=zeros<cx_mat>(Nn,Nintc);
 	cx_mat Kb_c=zeros<cx_mat>(Nn,Nintc);
 	cx_mat Kc_c=zeros<cx_mat>(Nn,Nintc);
 	cx_mat Kd_c=zeros<cx_mat>(Nn,Nintc);
 
	cx_vec_tmp.zeros(Nintc);
	
	vec_tmp=-trans(pow(wc.rows(1,Nwc-1),4)-pow(wc.rows(0,Nwc-2),4))/(8*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Ka_c.row(0)=cx_vec_tmp;
	vec_tmp=-trans(pow(wc.rows(1,Nwc-1),3)-pow(wc.rows(0,Nwc-2),3))/(6*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kb_c.row(0)=cx_vec_tmp;
	vec_tmp=-trans(pow(wc.rows(1,Nwc-1),2)-pow(wc.rows(0,Nwc-2),2))/(4*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kc_c.row(0)=cx_vec_tmp;
	vec_tmp=-trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2))/(2*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kd_c.row(0)=cx_vec_tmp;
	
//	Ka_c.row(0)=-(pow(wc.rows(1,Nwc-1),4)-pow(wc.rows(0,Nwc-2),4))/(8*PI);
//	Kb_c.row(0)=-(pow(wc.rows(1,Nwc-1),3)-pow(wc.rows(0,Nwc-2),3))/(6*PI);
//	Kc_c.row(0)=-(pow(wc.rows(1,Nwc-1),2)-pow(wc.rows(0,Nwc-2),2))/(4*PI);
//	Kd_c.row(0)=-(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2))/(2*PI);
 
	mat Wnc=wn.rows(1,Nn-1)*ones<rowvec>(Nintc);
	mat Wc=ones<vec>(Nn-1)*wc.t();
 
 	mat logc=log((pow(Wnc,2)+pow(Wc.cols(1,Nintc),2))/(pow(Wnc,2)+pow(Wc.cols(0,Nintc-1),2)));
 	mat atanc=atan((Wnc % (Wc.cols(0,Nintc-1)-Wc.cols(1,Nintc)))/(Wc.cols(1,Nintc) % Wc.cols(0,Nintc-1)+pow(Wnc,2)));
	
	mat dWc=Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1);
	mat dWc2=pow(Wc.cols(1,Nintc),2)-pow(Wc.cols(0,Nintc-1),2);
	mat dWc3=pow(Wc.cols(1,Nintc),3)-pow(Wc.cols(0,Nintc-1),3);
	mat Wnc2=pow(Wnc,2);
	mat Wnc3=pow(Wnc,3);
	mat Wnc4=pow(Wnc,4);
 
 	Ka_c.rows(1,Nn-1)=-i*( -Wnc3 % dWc + i*(Wnc2 % dWc2)/2 + (Wnc % dWc3)/3 -i*(pow(Wc.cols(1,Nintc),4)-pow(Wc.cols(0,Nintc-1),4))/4 - Wnc4 % atanc - i*Wnc4 % logc/2 )/(2*PI);
	Kb_c.rows(1,Nn-1)=-i*( i*Wnc2 % dWc + (Wnc % dWc2)/2 - i*dWc3/3 + i*Wnc3 % atanc - Wnc3 % logc/2 )/(2*PI);
	Kc_c.rows(1,Nn-1)=-i*( Wnc % dWc - i*dWc2/2 + Wnc2 % atanc + i*Wnc2 % logc/2 )/(2*PI);
 	Kd_c.rows(1,Nn-1)=-i*( -i*dWc -i*Wnc % atanc + Wnc % logc/2 )/(2*PI);

 
 	int Pmax=pnmax+4;
	imat MP;
	pascal(Pmax+1,MP);
 
	double wtmp, wj, dw1;
	vec dwp;
	int p,l, jni;
	for (j=0; j<Nwc-1; j++)
	{
		wtmp=abs(wc(j+1));
		if (abs(wc(j))>wtmp) wtmp=abs(wc(j));
 		jni=1;
		while (jni<Nn && wn(jni)<fi*wtmp)	jni++;
		while (jni<Nn && pow(wn(jni),pnmax)==0)		jni++;
		
		if (jni<Nn)
		{
			wj=wc(j);
			dw1=wc(j+1)-wj;
			dwp.zeros(Pmax);
			dwp(0)=dw1;
 			dwp(1)=pow(dw1,2)+2*wj*dw1;
			for (p=3; p<=Pmax; p++)
			{
				dwp(p-1)=0;
				for (l=0; l<p; l++)
				{
					dwp(p-1)=dwp(p-1)+MP(l,p-l)*pow(dw1,p-l)*pow(wj,l);
				}
			}
 		
			Ka_c.submat(jni,j,Nn-1,j)=zeros<cx_vec>(Nn-jni);
			Kb_c.submat(jni,j,Nn-1,j)=zeros<cx_vec>(Nn-jni);
			Kc_c.submat(jni,j,Nn-1,j)=zeros<cx_vec>(Nn-jni);
			Kd_c.submat(jni,j,Nn-1,j)=zeros<cx_vec>(Nn-jni);
			for (p=pnmax; p>=1; p--)
			{
				Ka_c.submat(jni,j,Nn-1,j)=Ka_c.submat(jni,j,Nn-1,j) + pow(i,-p)*dwp(p+3)/(2*PI*(p+4)*pow(wn.rows(jni,Nn-1),p));
				Kb_c.submat(jni,j,Nn-1,j)=Kb_c.submat(jni,j,Nn-1,j) + pow(i,-p)*dwp(p+2)/(2*PI*(p+3)*pow(wn.rows(jni,Nn-1),p));
				Kc_c.submat(jni,j,Nn-1,j)=Kc_c.submat(jni,j,Nn-1,j) + pow(i,-p)*dwp(p+1)/(2*PI*(p+2)*pow(wn.rows(jni,Nn-1),p));
				Kd_c.submat(jni,j,Nn-1,j)=Kd_c.submat(jni,j,Nn-1,j) + pow(i,-p)*dwp(p)/(2*PI*(p+1)*pow(wn.rows(jni,Nn-1),p));
			}
		}
	}


	cx_mat Ka_d=zeros<cx_mat>(Nn,Nintd);
 	cx_mat Kb_d=zeros<cx_mat>(Nn,Nintd);
 	cx_mat Kc_d=zeros<cx_mat>(Nn,Nintd);
 	cx_mat Kd_d=zeros<cx_mat>(Nn,Nintd);
	
	rowvec ud2(Nud+1);
	ud2.cols(1,Nud)=ud.t();
	ud2(0)=1.0/(wr-w0r);
	
	mat Wnd=wn.rows(1,Nn-1)*ones<rowvec>(Nintd-1);
	mat Ud=ones<vec>(Nn-1)*ud2;
	
	rowvec wd=1/ud2+w0r;
	mat Wd=ones<vec>(Nn-1)*wd;
	
	cx_vec_tmp.zeros(Nud);


	vec_tmp=-(pow(ud2.cols(1,Nintd-1),2)-pow(ud2.cols(0,Nintd-2),2))/(4*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Ka_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
	vec_tmp=-(ud2.cols(1,Nintd-1)-ud2.cols(0,Nintd-2))/(2*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kb_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
	vec_tmp=-log(ud2.cols(1,Nintd-1)/ud2.cols(0,Nintd-2))/(2*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kc_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
	vec_tmp=(wd.cols(1,Nintd-1)-wd.cols(0,Nintd-2))/(2*PI);
	cx_vec_tmp.set_real(vec_tmp);
	Kd_d.submat(0,0,0,Nintd-2)=cx_vec_tmp;
	
//	Ka_d.submat(0,0,0,Nintd-2)=-(pow(ud2.cols(1,Nintd-1),2)-pow(ud2.cols(0,Nintd-2),2))/(4*PI);
// 	Kb_d.submat(0,0,0,Nintd-2)=-(ud2.cols(1,Nintd-1)-ud2.cols(0,Nintd-2))/(2*PI);
// 	Kc_d.submat(0,0,0,Nintd-2)=-log(ud2.cols(1,Nintd-1)/ud2.cols(0,Nintd-2))/(2*PI);
//	Kd_d.submat(0,0,0,Nintd-2)=(wd.rows(1,Nintd-1)-wd.rows(0,Nintd-2))/(2*PI);


	mat atand=atan((Wnd % (Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2)))/(1+w0r*(Ud.cols(1,Nintd-1)+Ud.cols(0,Nintd-2))+(pow(w0r,2)+pow(Wnd,2)) % Ud.cols(0,Nintd-2) % Ud.cols(1,Nintd-1)));
 	mat logd=log((1+2*w0r*Ud.cols(1,Nintd-1)+pow(Ud.cols(1,Nintd-1),2) % (pow(Wnd,2)+pow(w0r,2)))/(1+2*w0r*Ud.cols(0,Nintd-2)+pow(Ud.cols(0,Nintd-2),2) % (pow(Wnd,2)+pow(w0r,2))));
	
	Ka_d.submat(1,0,Nn-1,Nintd-2)=(-i*(2*Wnd % (Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2)))/pow(Wnd + i*w0r,2) - i*(w0r*(pow(Ud.cols(1,Nintd-1),2)-pow(Ud.cols(0,Nintd-2),2)))/(Wnd + i*w0r) +  i*(2*Wnd % atand)/pow(Wnd + i*w0r,3)  - (Wnd % logd)/pow(Wnd + i*w0r,3) )/(4*PI);
 	Kb_d.submat(1,0,Nn-1,Nintd-2)=(-i*(2*w0r*(Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2)))/(Wnd + i*w0r) -(2*Wnd % atand)/pow(Wnd + i*w0r,2) -i*(Wnd % logd)/pow(Wnd + i*w0r,2) )/(4*PI);
 	Kc_d.submat(1,0,Nn-1,Nintd-2)=( -2*log(Ud.cols(1,Nintd-1)/Ud.cols(0,Nintd-2)) -i*(2*Wnd % atand)/(Wnd + i*w0r) + (Wnd % logd)/(Wnd + i*w0r) )/(4*PI);
 	Kd_d.submat(1,0,Nn-1,Nintd-2)=( 2*(Wd.cols(1,Nintd-1)-Wd.cols(0,Nintd-2)) -i*2.0*Wnd % log(Ud.cols(1,Nintd-1)/Ud.cols(0,Nintd-2)) + 2*Wnd % atand +i*Wnd % logd )/(4*PI);
	
	
	cx_mat KG=-(Ka_g*Pa_g+Kb_g*Pb_g+Kc_g*Pc_g+Kd_g*Pd_g)*MC;
 	cx_mat KC=(Ka_c*Pa_c+Kb_c*Pb_c+Kc_c*Pc_c+Kd_c*Pd_c)*MC;
 	cx_mat KD=-(Ka_d*Pa_d+Kb_d*Pb_d+Kc_d*Pc_d+Kd_d*Pd_d)*MC;
 
 	Kcx=KG+KC+KD;
	

	rowvec Knorm_a_g=zeros<rowvec>(Nintg);
	rowvec Knorm_b_g=zeros<rowvec>(Nintg);
	rowvec Knorm_c_g=zeros<rowvec>(Nintg);
	rowvec Knorm_d_g=zeros<rowvec>(Nintg);
	
	Knorm_a_g(0)=-pow(ug(0),2)/2;
 	Knorm_b_g(0)=-ug(0);
	
	Knorm_a_g.cols(1,Nintg-1)=-(pow(ug2.cols(1,Nug),2)-pow(ug2.cols(0,Nug-1),2))/2;
 	Knorm_b_g.cols(1,Nintg-1)=-(ug2.cols(1,Nug)-ug2.cols(0,Nug-1));
 	Knorm_c_g.cols(1,Nintg-1)=-log(ug2.cols(1,Nug)/ug2.cols(0,Nug-1));
 	Knorm_d_g.cols(1,Nintg-1)=1.0/ug2.cols(1,Nug)-1.0/ug2.cols(0,Nug-1);
	
	rowvec KM0g=(Knorm_a_g*Pa_g+Knorm_b_g*Pb_g+Knorm_c_g*Pc_g+Knorm_d_g*Pd_g)*MC/(2*PI);
	
	rowvec Knorm_a_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_b_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_c_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_d_c_r=zeros<rowvec>(Nintc);
	
	Knorm_a_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),4)/4);
	Knorm_b_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),3)/3);
	Knorm_c_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),2)/2);
	Knorm_d_c_r=trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2));
	
	rowvec KM0c=(Knorm_a_c_r*Pa_c+Knorm_b_c_r*Pb_c_r+Knorm_c_c_r*Pc_c_r+Knorm_d_c_r*Pd_c_r)*MC/(2*PI);
	
	rowvec Knorm_a_d=zeros<rowvec>(Nintd);
	rowvec Knorm_b_d=zeros<rowvec>(Nintd);
	rowvec Knorm_c_d=zeros<rowvec>(Nintd);
	rowvec Knorm_d_d=zeros<rowvec>(Nintd);
	
	Knorm_a_d.cols(0,Nintd-2)=-(pow(ud2.cols(1,Nud),2)-pow(ud2.cols(0,Nud-1),2))/2;
	Knorm_b_d.cols(0,Nintd-2)=-(ud2.cols(1,Nud)-ud2.cols(0,Nud-1));
	Knorm_c_d.cols(0,Nintd-2)=-log(ud2.cols(1,Nud)/ud2.cols(0,Nud-1));
	Knorm_d_d.cols(0,Nintd-2)=1.0/ud2.cols(1,Nud)-1.0/ud2.cols(0,Nud-1);
	
	Knorm_a_d(Nintd-1)=pow(ud2(Nud-1),2)/2;
	Knorm_b_d(Nintd-1)=ud2(Nud-1);
	
	rowvec KM0d=(Knorm_a_d*Pa_d+Knorm_b_d*Pb_d+Knorm_c_d*Pc_d+Knorm_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM0=KM0g+KM0c+KM0d;
	

	rowvec KM1_a_g=zeros<rowvec>(Nintg);
	rowvec KM1_b_g=zeros<rowvec>(Nintg);
	rowvec KM1_c_g=zeros<rowvec>(Nintg);
	rowvec KM1_d_g=zeros<rowvec>(Nintg);
	
	KM1_a_g.cols(1,Nintg-1)=Knorm_b_g.cols(1,Nintg-1);
	KM1_b_g.cols(1,Nintg-1)=Knorm_c_g.cols(1,Nintg-1);
	KM1_c_g.cols(1,Nintg-1)=Knorm_d_g.cols(1,Nintg-1);
	KM1_d_g.cols(1,Nintg-1)=(1.0/pow(ug2.cols(1,Nug),2)-1.0/pow(ug2.cols(0,Nug-1),2))/2;
	
	rowvec KM1g_tmp=(KM1_a_g*Pa_g+KM1_b_g*Pb_g+KM1_c_g*Pc_g+KM1_d_g*Pd_g)*MC/(2*PI);
	
	rowvec KM1g=w0l*KM0g+KM1g_tmp;
	
	mat Wjc=diagmat(wc.rows(0,Nintc-1));
	
	rowvec KM1_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),5)/5.0);
	
	rowvec KM1_a_c=KM1_a_c_tmp+Knorm_a_c_r*Wjc;
	rowvec KM1_b_c=Knorm_a_c_r+Knorm_b_c_r*Wjc;
	rowvec KM1_c_c=Knorm_b_c_r+Knorm_c_c_r*Wjc;
	rowvec KM1_d_c=Knorm_c_c_r+Knorm_d_c_r*Wjc;
	
	rowvec KM1c=(KM1_a_c*Pa_c+KM1_b_c*Pb_c_r+KM1_c_c*Pc_c_r+KM1_d_c*Pd_c_r)*MC/(2*PI);
	
	rowvec KM1_a_d=zeros<rowvec>(Nintd);
	rowvec KM1_b_d=zeros<rowvec>(Nintd);
	rowvec KM1_c_d=zeros<rowvec>(Nintd);
	rowvec KM1_d_d=zeros<rowvec>(Nintd);
	
	KM1_a_d.cols(0,Nintd-2)=Knorm_b_d.cols(0,Nintd-2);
	KM1_b_d.cols(0,Nintd-2)=Knorm_c_d.cols(0,Nintd-2);
	KM1_c_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM1_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),2)-1.0/pow(ud2.cols(0,Nud-1),2))/2;
	
	rowvec KM1d_tmp=(KM1_a_d*Pa_d+KM1_b_d*Pb_d+KM1_c_d*Pc_d+KM1_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM1d=w0r*KM0d+KM1d_tmp;
	
	rowvec KM1=KM1g+KM1c+KM1d;
	
	
	rowvec KM2_a_g=zeros<rowvec>(Nintg);
	rowvec KM2_b_g=zeros<rowvec>(Nintg);
	rowvec KM2_c_g=zeros<rowvec>(Nintg);
	rowvec KM2_d_g=zeros<rowvec>(Nintg);
	
	KM2_a_g.cols(1,Nintg-1)=Knorm_c_g.cols(1,Nintg-1);
	KM2_b_g.cols(1,Nintg-1)=Knorm_d_g.cols(1,Nintg-1);
	KM2_c_g.cols(1,Nintg-1)=KM1_d_g.cols(1,Nintg-1);
	KM2_d_g.cols(1,Nintg-1)=(1.0/pow(ug2.cols(1,Nug),3)-1.0/pow(ug2.cols(0,Nug-1),3))/3;
	
	rowvec KM2g_tmp=(KM2_a_g*Pa_g+KM2_b_g*Pb_g+KM2_c_g*Pc_g+KM2_d_g*Pd_g)*MC/(2*PI);
	
	rowvec KM2g=pow(w0l,2)*KM0g+2*w0l*KM1g_tmp+KM2g_tmp;
	
	rowvec KM2_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),6)/6);
	
	rowvec KM2_a_c=KM2_a_c_tmp+2*KM1_a_c_tmp*Wjc+Knorm_a_c_r*pow(Wjc,2);
	rowvec KM2_b_c=KM1_a_c_tmp+2*Knorm_a_c_r*Wjc+Knorm_b_c_r*pow(Wjc,2);
	rowvec KM2_c_c=Knorm_a_c_r+2*Knorm_b_c_r*Wjc+Knorm_c_c_r*pow(Wjc,2);
	rowvec KM2_d_c=Knorm_b_c_r+2*Knorm_c_c_r*Wjc+Knorm_d_c_r*pow(Wjc,2);
	
	rowvec KM2c=(KM2_a_c*Pa_c+KM2_b_c*Pb_c_r+KM2_c_c*Pc_c_r+KM2_d_c*Pd_c_r)*MC/(2*PI);
	
	rowvec KM2_a_d=zeros<rowvec>(Nintd);
	rowvec KM2_b_d=zeros<rowvec>(Nintd);
	rowvec KM2_c_d=zeros<rowvec>(Nintd);
	rowvec KM2_d_d=zeros<rowvec>(Nintd);
	
	KM2_a_d.cols(0,Nintd-2)=Knorm_c_d.cols(0,Nintd-2);
	KM2_b_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM2_c_d.cols(0,Nintd-2)=KM1_d_d.cols(0,Nintd-2);
	KM2_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),3)-1.0/pow(ud2.cols(0,Nud-1),3))/3;
	
	rowvec KM2d_tmp=(KM2_a_d*Pa_d+KM2_b_d*Pb_d+KM2_c_d*Pc_d+KM2_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM2d=pow(w0r,2)*KM0d+2*w0r*KM1d_tmp+KM2d_tmp;
	
	rowvec KM2=KM2g+KM2c+KM2d;
	
	rowvec KM3_a_g=zeros<rowvec>(Nintg);
	rowvec KM3_b_g=zeros<rowvec>(Nintg);
	rowvec KM3_c_g=zeros<rowvec>(Nintg);
	rowvec KM3_d_g=zeros<rowvec>(Nintg);
	
	KM3_a_g.cols(1,Nintg-1)=Knorm_d_g.cols(1,Nintg-1);
	KM3_b_g.cols(1,Nintg-1)=KM1_d_g.cols(1,Nintg-1);
	KM3_c_g.cols(1,Nintg-1)=KM2_d_g.cols(1,Nintg-1);
	KM3_d_g.cols(1,Nintg-1)=(1.0/pow(ug2.cols(1,Nug),4)-1.0/pow(ug2.cols(0,Nug-1),4))/4;
	
	rowvec KM3g_tmp=(KM3_a_g*Pa_g+KM3_b_g*Pb_g+KM3_c_g*Pc_g+KM3_d_g*Pd_g)*MC/(2*PI);
	
	rowvec KM3g=pow(w0l,3)*KM0g+3*pow(w0l,2)*KM1g_tmp+3*w0l*KM2g_tmp+KM3g_tmp;
	
	rowvec KM3_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),7)/7);
	
	rowvec KM3_a_c=KM3_a_c_tmp+3*KM2_a_c_tmp*Wjc+3*KM1_a_c_tmp*pow(Wjc,2)+Knorm_a_c_r*pow(Wjc,3);
	rowvec KM3_b_c=KM2_a_c_tmp+3*KM1_a_c_tmp*Wjc+3*Knorm_a_c_r*pow(Wjc,2)+Knorm_b_c_r*pow(Wjc,3);
	rowvec KM3_c_c=KM1_a_c_tmp+3*Knorm_a_c_r*Wjc+3*Knorm_b_c_r*pow(Wjc,2)+Knorm_c_c_r*pow(Wjc,3);
	rowvec KM3_d_c=Knorm_a_c_r+3*Knorm_b_c_r*Wjc+3*Knorm_c_c_r*pow(Wjc,2)+Knorm_d_c_r*pow(Wjc,3);
	
	rowvec KM3c=(KM3_a_c*Pa_c+KM3_b_c*Pb_c_r+KM3_c_c*Pc_c_r+KM3_d_c*Pd_c_r)*MC/(2*PI);
	
	rowvec KM3_a_d=zeros<rowvec>(Nintd);
	rowvec KM3_b_d=zeros<rowvec>(Nintd);
	rowvec KM3_c_d=zeros<rowvec>(Nintd);
	rowvec KM3_d_d=zeros<rowvec>(Nintd);
	
	KM3_a_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM3_b_d.cols(0,Nintd-2)=KM1_d_d.cols(0,Nintd-2);
	KM3_c_d.cols(0,Nintd-2)=KM2_d_d.cols(0,Nintd-2);
	KM3_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),4)-1.0/pow(ud2.cols(0,Nud-1),4))/4;
	
	rowvec KM3d_tmp=(KM3_a_d*Pa_d+KM3_b_d*Pb_d+KM3_c_d*Pc_d+KM3_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM3d=pow(w0r,3)*KM0d+3*pow(w0r,2)*KM1d_tmp+3*w0r*KM2d_tmp+KM3d_tmp;
	
	rowvec KM3=KM3g+KM3c+KM3d;
	
	KM.row(0)=KM1;
	KM.row(1)=KM2;
	KM.row(2)=KM3;
	
	K.zeros(2*Nn,Nw);
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	K.rows(even_ind)=real(Kcx);
	K.rows(even_ind+1)=imag(Kcx);

/*
	//test K
	rowvec x0={-0.2, 0, 0.3};
	rowvec s0={0.02, 0.002, 0.03};
	rowvec wgt={0.5,1,2};
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
	test_A=test_A/0.1428571428578;
	
	cx_vec Gtest=Kcx*test_A;
	vec Gr_test=real(Gtest);
	vec Gi_test=imag(Gtest);
	
	graph_2D g1, g2, g3, g4;
	char attr1[]="'o',color='r',markerfacecolor='none'";
	char attr2[]="'s',color='b',markerfacecolor='none'";
	char attr3[]="'.-',color='r'";
	char attr4[]="'.-',color='b'";
	
	g1.add_data(wn.memptr(), Gr.memptr(), Nn);
	g1.add_attribute(attr1);
	g1.add_data(wn.memptr(), Gr_test.memptr(), Nn);
	g1.add_attribute(attr2);
	g1.curve_plot();
	
	g2.add_data(wn.memptr(), Gi.memptr(), Nn);
	g2.add_attribute(attr1);
	g2.add_data(wn.memptr(), Gi_test.memptr(), Nn);
	g2.add_attribute(attr2);
	g2.curve_plot();
	
	plot(g3,wn,(Gr_test-Gr)/Gr,NULL,NULL,attr3);
	plot(g4,wn,(Gi_test-Gi)/Gi,NULL,NULL,attr4);
	
	graph_2D::show_figures();
*/
	
/*
 // test KM
	vec coeffs0=MM*test_A;
	
	vec coeffs1(4*(Nw+1));
	coeffs1(0)=coeffs0(0);
	coeffs1(1)=coeffs0(1);
	coeffs1(2)=0;
	coeffs1(3)=0;
	coeffs1(4*(Nw+1)-4)=coeffs0(Ncfs0-2);
	coeffs1(4*(Nw+1)-3)=coeffs0(Ncfs0-1);
	coeffs1(4*(Nw+1)-2)=0;
	coeffs1(4*(Nw+1)-1)=0;
	
	uvec ind_tmp=linspace<uvec>(1,Nw-1,Nw-1);
	coeffs1.rows(4*ind_tmp)=coeffs0(3*ind_tmp-1);
	coeffs1.rows(4*ind_tmp+1)=coeffs0(3*ind_tmp);
	coeffs1.rows(4*ind_tmp+2)=coeffs0(3*ind_tmp+1);
	
	ind_tmp=linspace<uvec>(1,Nw_lims(1),Nw_lims(1));
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp-1);
	ind_tmp=linspace<uvec>(Nw_lims(1)+1,Nw-1,Nw-Nw_lims(1)-1);
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp);
	
	//check moments
//	cout<<KM0*test_A<<endl;
	cout<<K.row(0)*test_A<<endl;
	cout<<KM*test_A<<endl;
 
	void *par[5];
	par[0]=&w;
	par[1]=&Nw_lims;
	par[2]=&ws;
	par[3]=&coeffs1;
	j=3;
	par[4]=&j;
	
	fctPtr1 Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::spline_val_G_part_int);
	double int_lims[2];
	double tol=1e-10;
	int nbEval[1];
	
	int_lims[0]=w(0);
	int_lims[1]=w(Nw_lims[0]);
	double M_g=quadInteg1D(Ptr, int_lims, tol, nbEval, par)/(2*PI);
	int_lims[0]=w(Nw_lims[0]);
	int_lims[1]=w(Nw_lims[1]);
	double M_c=quadInteg1D(Ptr, int_lims, tol, nbEval, par)/(2*PI);
	int_lims[0]=w(Nw_lims[1]);
	int_lims[1]=w(Nw-1);
	double M_d=quadInteg1D(Ptr, int_lims, tol, nbEval, par)/(2*PI);
	
//	cout<<"Mg:\n"<<KM1g*test_A<<M_g<<endl;
//	cout<<"Mc:\n"<<KM1c*test_A<<M_c<<endl;
//	cout<<"Md:\n"<<KM1d*test_A<<M_d<<endl;
	cout<<"M: "<<M_g+M_c+M_d<<endl;
*/
	return true;
}

bool OmegaMaxEnt_data::test_low_energy_peak_bosons()
{
//	if (Nwn_test_metal>Nn-2)	Nwn_test_metal=Nn-2;
	
	peak_exists=false;
//	bool test_peak=false;

//	vec D2G=(Gr.rows(0,Nn-3)-2*Gr.rows(1,Nn-2)+Gr.rows(2,Nn-1));
//	if (abs(D2G(0)/Gr(1))>R_d2G_chi_peak && Nn>20) test_peak=true;
	
//	if (test_peak)
//	{
//		cout<<"Re(G) has a jump at low frequency. Looking for a peak in the spectral function at low energy...\n";
		cout<<"Looking for a peak in the spectral function at low energy...\n";
		
		int nmax=n(Nn-1);
		
		int NCnmin=1;
		int NCnmax=3;
		int NNCn=NCnmax-NCnmin+1;
		ivec NCn=linspace<ivec>(NCnmin,NCnmax,NNCn);
		
		int NCpmin=1;
		int NCpmax=15;
		if (NCpmax>(nmax-NCn.max()-2))
		{
			NCpmax=nmax-NCn.max()-2;
		}
		int NNCp=NCpmax-NCpmin+1;
		ivec NCp=linspace<ivec>(NCpmin,NCpmax,NNCp);
		
		int DNwn=2;
		
		int p0_min=2;
		int p0_max=2;
		int Np=p0_max-p0_min+1;
		ivec p0=linspace<ivec>(p0_min,p0_max,Np);
		
		imat NCmin(Np,2);
		
		vec M0_inc_opt(Np,fill::zeros);
		vec M1_pk_opt(Np,fill::zeros);
		vec M2_pk_opt(Np,fill::zeros);
		vec varM2_pk_opt(Np,fill::zeros);
		
		mat X, CG, P, invCG, AM, BM, AMP, BMP, MP, Mtmp, chi2tmp;
		mat M0_inc(NNCp,NNCn), M1_pk(NNCp,NNCn), M2_pk(NNCp,NNCn), chi2_pk(NNCp,NNCn);
		int j, l, m, p, Nfit;
		uword q;
		vec Gtmp, Glftmp, diffG;
		rowvec maxX;
		double vartmp;
		
		//		char UPLO='U';
		//		int NA, NRHS=1, INFO;
		
		//		mat test_M, AM2;
		//		vec BM2;
		
		for (q=0; q<Np; q++)
		{
			M0_inc.zeros();
			M1_pk.zeros();
			M2_pk.zeros();
			chi2_pk.zeros();
			
			for (l=0; l<NNCp; l++)
			{
				for (m=0; m<NNCn; m++)
				{
					Nfit=NCp(l)+NCn(m)+1+DNwn;
					
					X.zeros(2*Nfit,2*(NCp(l)+NCn(m)+1));
					
					for (j=NCp(l); j>=-NCn(m); j--)
					{
						for (p=p0(q); p<=Nfit+p0(q)-1; p++)
						{
							X(2*(p-p0(q)),2*NCp(l)-2*j+1)=pow(wn(p-1),2*j);
							X(2*(p-p0(q))+1,2*NCp(l)-2*j)=pow(wn(p-1),2*j+1);
//							X(2*(p-p0(q)),2*NCp(l)-2*j+1)=pow(-1,j)*pow(wn(p-1),2*j);
//							X(2*(p-p0(q))+1,2*NCp(l)-2*j)=pow(-1,j)*pow(wn(p-1),2*j+1);
						}
					}
					
					Gtmp=Gchi2.rows(2*p0(q)-2,2*(Nfit+p0(q))-3);
					CG=COV.submat(2*p0(q)-2,2*p0(q)-2,2*(Nfit+p0(q))-3,2*(Nfit+p0(q))-3);
					
					invCG=inv(CG);
					//	invCG=inv_sympd(CG);
					AM=(X.t())*invCG*X;
					BM=(X.t())*invCG*Gtmp;
					//					BM2=BM;
					//					AM2=AM;
					
					Mtmp=solve(AM,BM);
					
					//					cout<<l<<" "<<m<<endl;
					
					//					NA=AM.n_rows;
					//					dposv_(&UPLO, &NA, &NRHS, AM.memptr(), &NA, BM.memptr(), &NA, &INFO);
					//					Mtmp=BM;
					
					//					dposv_(&UPLO, &NA, &NRHS, AM2.memptr(), &NA, BM2.memptr(), &NA, &INFO);
					//					test_M.zeros(NA,2);
					//					test_M.col(0)=Mtmp;
					//					test_M.col(1)=BM2;
					//					cout<<test_M<<endl;
					
					//					maxX=max(abs(X),0);
					//					P=diagmat(1.0/maxX);
					//					AMP=(P*(AM*P)+(P*AM)*P)/2;
					//					BMP=P*BM;
					//					dposv_(&UPLO, &NA, &NRHS, AMP.memptr(), &NA, BMP.memptr(), &NA, &INFO);
					//					MP=BMP;
					//					MP=solve(AMP,BMP);
					//					Mtmp=P*MP;
					
					Glftmp=X*Mtmp;
					
					diffG=Gtmp-Glftmp;
					
					chi2tmp=((diffG.t())*invCG*diffG)/(2.0*Nfit);

					chi2_pk(l,m)=chi2tmp(0,0);
					M0_inc(l,m)=-Mtmp(2*NCp(l)+1);
					M1_pk(l,m)=-Mtmp(2*NCp(l)+2);
					M2_pk(l,m)=-Mtmp(2*NCp(l)+3);
				}
			}
			
			uint NvM=2;
			uint NvarM=NNCp-2*NvM;
			mat varM2_pk(NvarM,NNCn,fill::zeros);
			for (j=NvM; j<NNCp-NvM; j++)
			{
				varM2_pk.row(j-NvM)=var(M2_pk.rows(j-NvM,j+NvM));
			}
			
			uword indpM0, indnM0;
			
			double varM2_pk_min=varM2_pk.min(indpM0,indnM0);
			
			l=indpM0+NvM;
			m=indnM0;
			
			NCmin(q,0)=l;
			NCmin(q,1)=m;
			
			M0_inc_opt(q)=M0_inc(l,m);
			M1_pk_opt(q)=M1_pk(l,m);
			M2_pk_opt(q)=M2_pk(l,m);
			varM2_pk_opt(q)=varM2_pk_min;
			
			if (displ_adv_prep_figs)
			{
				graph_2D g1, g2, g3, g4;
				
				vec x(NNCp), y;
				
				for (l=0; l<NNCp; l++)
					x(l)=NCp(l);
				
				char xl[]="NCp";
				char yl[]="norm_peak";
				char yl2[]="Mpk_1";
				char yl3[]="Mpk_2";
				char yl4[]="chi2_LS";
				
				double xlims[2], ylims[2], ymin, ymax;
				
				xlims[0]=x.min();
				xlims[1]=x.max();
				
				char lgd_entry[100];
				
				for (m=0; m<NNCn; m++)
				{
					sprintf(lgd_entry,"NCn=%d",(int)NCn(m));
					
					y=-Gr(0)-M0_inc.col(m);
					ymax=y.max();
					ymin=y.min();
					ylims[0]=ymin-0.1*(ymax-ymin);
					ylims[1]=ymax+0.1*(ymax-ymin);
					g1.add_data(x.memptr(),y.memptr(),x.n_rows);
					g1.add_to_legend(lgd_entry);
					
					y=M1_pk.col(m);
					ymax=y.max();
					ymin=y.min();
					ylims[0]=ymin-0.1*(ymax-ymin);
					ylims[1]=ymax+0.1*(ymax-ymin);
					g2.add_data(x.memptr(),y.memptr(),x.n_rows);
					g2.add_to_legend(lgd_entry);
					
					y=M2_pk.col(m);
					ymax=y.max();
					ymin=y.min();
					ylims[0]=ymin-0.1*(ymax-ymin);
					ylims[1]=ymax+0.1*(ymax-ymin);
					g3.add_data(x.memptr(),y.memptr(),x.n_rows);
					g3.add_to_legend(lgd_entry);
					
					y=chi2_pk.col(m);
					ymax=y.max();
					ymin=y.min();
					ylims[0]=ymin-0.1*(ymax-ymin);
					ylims[1]=ymax+0.1*(ymax-ymin);
					g4.add_data(x.memptr(),y.memptr(),x.n_rows);
					g4.add_to_legend(lgd_entry);
				}
				g1.set_axes_labels(xl,yl);
				g1.set_axes_lims(xlims,ylims);
				g1.curve_plot();
				
				g2.set_axes_labels(xl,yl2);
				g2.set_axes_lims(xlims,ylims);
				g2.curve_plot();
				
				g3.set_axes_labels(xl,yl3);
				g3.set_axes_lims(xlims,ylims);
				g3.curve_plot();
				
				g4.set_axes_labels(xl,yl4);
				g4.set_axes_lims(xlims,ylims);
				g4.curve_plot();
				
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
				
			}
		}
		
		double varM2_min=varM2_pk_opt.min(q);
		
		double peak_weight=-Gr(0)-M0_inc_opt(q);
		double peak_center=M1_pk_opt(q)/peak_weight;
		double var_peak=M2_pk_opt(q)/peak_weight-pow(peak_center,2);
		double peak_width=0;
		if (var_peak>0)		peak_width=sqrt(var_peak);
		double m2_lf_min=M2_pk_opt(q);
	
		if (peak_width>100*EPSILON && M0_inc_opt(q)>100*EPSILON && varM2_min/M2_pk_opt(q)<varM2_peak_max && peak_weight>peak_weight_min*M1n)
		{
			peak_exists=true;
			
			l=NCmin(q,0);
			m=NCmin(q,1);
			
			Nfit=NCp(l)+NCn(m)+1+DNwn;
			
			X.zeros(2*Nfit,2*(NCp(l)+NCn(m)+1));
			
			for (j=NCp(l); j>=-NCn(m); j--)
			{
				for (p=p0(q); p<=Nfit+p0(q)-1; p++)
				{
					X(2*(p-p0(q)),2*NCp(l)-2*j+1)=pow(wn(p-1),2*j);
					X(2*(p-p0(q))+1,2*NCp(l)-2*j)=pow(wn(p-1),2*j+1);
//					X(2*(p-p0(q)),2*NCp(l)-2*j+1)=pow(-1,j)*pow(wn(p-1),2*j);
//					X(2*(p-p0(q))+1,2*NCp(l)-2*j)=pow(-1,j)*pow(wn(p-1),2*j+1);
				}
			}
			
			Gtmp=Gchi2.rows(2*p0(q)-2,2*(Nfit+p0(q))-3);
			CG=COV.submat(2*p0(q)-2,2*p0(q)-2,2*(Nfit+p0(q))-3,2*(Nfit+p0(q))-3);
			
			invCG=inv(CG);
			//	invCG=inv_sympd(CG);
			AM=(X.t())*invCG*X;
			AM=0.5*(AM.t()+AM);
			BM=(X.t())*invCG*Gtmp;
			
			Mtmp=solve(AM,BM);
			
			//			NA=AM.n_rows;
			//			dposv_(&UPLO, &NA, &NRHS, AM.memptr(), &NA, BM.memptr(), &NA, &INFO);
			//			Mtmp=BM;
			
			//			rowvec maxX=max(abs(X),0);
			//			mat P=diagmat(1.0/maxX);
			//			AMP=(P*(AM*P)+(P*AM)*P)/2.0;
			//			BMP=P*BM;
			//			dposv_(&UPLO, &NA, &NRHS, AMP.memptr(), &NA, BMP.memptr(), &NA, &INFO);
			//			MP=BMP;
			//			MP=solve(AMP,BMP);
			//			Mtmp=P*MP;
			
			Glftmp=X*Mtmp;
			
			mat COVpeak=inv(AM);
			
			//			invAMP=inv(AMP);
			//			COVpeak=((P*invAMP)*P+P*(invAMP*P))/2;
			
			double err_norm_peak=sqrt(COVpeak(2*NCp(l)+1,2*NCp(l)+1));
			double err_M1_peak=sqrt(COVpeak(2*NCp(l)+2,2*NCp(l)+2));
			double err_M2_peak=sqrt(COVpeak(2*NCp(l)+3,2*NCp(l)+3));
			double err_peak_position=err_M1_peak/peak_weight-err_norm_peak*peak_center/peak_weight;
			double err_std_peak=(-err_norm_peak*m2_lf_min/peak_weight+2*err_norm_peak*peak_center*peak_center/peak_weight+err_M2_peak/peak_weight-2*err_M1_peak*peak_center/peak_weight)/(2*peak_width);
			
			cout<<"Peak detected\n";
			cout<<"peak width: "<<peak_width<<endl;
			//cout<<"error on width: "<<err_std_peak<<endl;
			cout<<"peak weight: "<<peak_weight<<endl;
			//cout<<"error on weight: "<<err_norm_peak<<endl;
			//cout<<"peak position: "<<peak_center<<endl;
			//cout<<"error on position: "<<err_peak_position<<endl;
			
			if (displ_adv_prep_figs)
			{
				graph_2D g1, g2;
				
				vec x=wn.rows(p0(q)-1,Nfit+p0(q)-2), y;
				
				char xl[]="$\\\\omega_n$";
				char yl[]="Gr";
				char yl2[]="Gi";
				char lgd1[]="data";
				char lgd2[]="fit";
				char attr1[]="'o', markeredgecolor='r', markerfacecolor='none'";
				char attr2[]="'s', markeredgecolor='b', markerfacecolor='none'";
				
				double xlims[2], ylims[2], ymin, ymax;
				
				xlims[0]=x.min();
				xlims[1]=x.max();
				
				y=Gr.rows(p0(q)-1,Nfit+p0(q)-2);
				ymax=y.max();
				ymin=y.min();
				ylims[0]=ymin-0.1*(ymax-ymin);
				ylims[1]=ymax+0.1*(ymax-ymin);
				g1.add_data(x.memptr(),y.memptr(),Nfit);
				g1.add_to_legend(lgd1);
				g1.add_attribute(attr1);
				uvec even_ind=linspace<uvec>(0,2*Nfit-2,Nfit);
				y=Glftmp.rows(even_ind);
				g1.add_data(x.memptr(),y.memptr(),Nfit);
				g1.add_to_legend(lgd2);
				g1.add_attribute(attr2);
				g1.set_axes_labels(xl,yl);
				g1.set_axes_lims(xlims,ylims);
				g1.curve_plot();
				
				y=Gi.rows(p0(q)-1,Nfit+p0(q)-2);
				ymax=y.max();
				ymin=y.min();
				ylims[0]=ymin-0.1*(ymax-ymin);
				ylims[1]=ymax+0.1*(ymax-ymin);
				g2.add_data(x.memptr(),y.memptr(),Nfit);
				g2.add_to_legend(lgd1);
				g2.add_attribute(attr1);
				uvec odd_ind=linspace<uvec>(1,2*Nfit-1,Nfit);
				y=Glftmp.rows(odd_ind);
				g2.add_data(x.memptr(),y.memptr(),Nfit);
				g2.add_to_legend(lgd2);
				g2.add_attribute(attr2);
				g2.set_axes_labels(xl,yl2);
				g2.set_axes_lims(xlims,ylims);
				g2.curve_plot();
				
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			dw_peak=peak_width/2.0;
		}
		else
		{
			cout<<"no peak found\n";
		}
//	}
	
	return peak_exists;
}

bool OmegaMaxEnt_data::set_moments_bosons()
{
	if (!tau_GF) M1n=abs(Gr(0));
	moments_provided=false;
	covm_diag=true;
	std_omega=0;
	NM=0;
	
	if (!tau_GF)
	{
		if (max(abs(Gi))!=0)
		{
			M0=1.0;
			if (M0_in.size())
				M0=stod(M0_in);
			errM0=err_norm*M0;
		}
		else
		{
			M0=0;
			errM0=default_error_M;
			M2=0;
			errM2=default_error_M;
		}
	}
	else
	{
		M0=M0t;
		if (M0_in.size())
			M0=stod(M0_in);
		if (M0)
			errM0=err_norm*M0;
		else
			errM0=default_error_M;
	}
	
	if (!tau_GF)
	{
		if (!SC_set)
		{
			SC=M0/M1n;
			SC_set=true;
		}
	}
	
	if (col_Gi>0)
	{
		M.zeros(1);
		M(0)=M0;
		errM.zeros(1);
		errM(0)=errM0;
		NMinput=1;
		NM=1;
		M_ord=linspace<vec>(0,NM-1,NM);
	}

	if (M1_in.size())
	{
		M1=stod(M1_in);
		M1_set=true;
		if (M1<=0)
		{
			cout<<"error: first moment of spectral function must be greater than 0 for bosons\n";
			return false;
		}
		moments_provided=true;
		cout<<"moments provided\n";
		if (col_Gi>0)
		{
			M.zeros(2);
			M(0)=M0;
			M(1)=M1;
			NM=2;
			M_ord=linspace<vec>(0,NM-1,NM);
		}
		else
		{
			M.zeros(1);
			M(0)=M1;
			NM=1;
			M_ord.zeros(1);
			M_ord(0)=1;
		}
		
		double var_omega=M1/M1n-pow(M0/M1n,2);
		if (var_omega<0)
		{
			cout<<"error: the variance computed from the provided moments is negative\n";
			return false;
		}
		std_omega=sqrt(var_omega);
		
		if (!SW_set)
		{
			SW=f_SW_std_omega*std_omega;
			SW_set=true;
		}
		
		if (errM1_in.size())
			errM1=stod(errM1_in);
		else
			errM1=default_error_M*M1;
		if (col_Gi>0)
		{
			if (M2_in.size())
			{
				M2=stod(M2_in);
				M2_set=true;
				M.zeros(3);
				M(0)=M0;
				M(1)=M1;
				M(2)=M2;
				NM=3;
				M_ord=linspace<vec>(0,NM-1,NM);
				if (errM2_in.size())
					errM2=stod(errM2_in);
				else
					errM2=default_error_M*M1n*pow(std_omega,3);
			}
		}
		errM.zeros(NM);
		if (col_Gi>0)
		{
			errM(0)=errM0;
			errM(1)=errM1;
			if (M2_in.size())	errM(2)=errM2;
		}
		else
			errM(0)=errM1;
	}
	COVM=diagmat(square(errM));
	
	if (col_Gi>0)
	{
		if ( NM<2 && !SW_in.size() && (!use_grid_params || omega_grid_params.n_cols<3) && !grid_omega_file.size() && !init_spectr_func_file.size() )
		{
			if (!tau_GF)
				cout<<"note: not enough information provided to define the real frequency grid. The program will try to extract the moments from the high frequencies of the Green function\n";
			else
				cout<<"not enough information provided to define the real frequency grid. The program will try to extract moments from G(tau) around tau=0 and tau=beta.\n";
			eval_moments=true;
		}
		
	}
	else
	{
		if ( NM<1 && !SW_in.size() && (!use_grid_params || omega_grid_params.n_cols<3) && !grid_omega_file.size() && !init_spectr_func_file.size() )
		{
			if (!tau_GF)
				cout<<"note: not enough information provided to define the real frequency grid. The program will try to extract the moments from the high frequencies of the Green function\n";
			else
				cout<<"not enough information provided to define the real frequency grid. The program will try to extract moments from G(tau) around tau=0 and tau=beta.\n";
			eval_moments=true;
		}
	}
	
	return true;
}

bool OmegaMaxEnt_data::set_G_omega_n_bosons()
{
	int j;
	
	signG=1;
	
	wn=green_data.col(0);
	Nn=green_data.n_rows;
	Gr=green_data.col(col_Gr-1);
	if (col_Gi>0)
		Gi=green_data.col(col_Gi-1);
	else
		Gi.zeros(Nn);
	
	if (G_omega_inf_in.size())
	{
		Gr=Gr-G_omega_inf;
	}
	
	indG_0=0, indG_f=Nn-1;
	wn_sign_change=false;
	wn_inverted=false;
	if (wn(0)*wn(Nn-1)<0)
	{
		wn_sign_change=true;
		j=1;
		while (wn(j)*wn(Nn-1)<0) j++;
		if ((Nn-j)>=j+1)
		{
			indG_0=j;
			indG_f=Nn-1;
		}
		else
		{
			indG_0=0;
			indG_f=j-1;
		}
		wn=wn.rows(indG_0,indG_f);
		Nn=wn.n_rows;
		Gr=Gr.rows(indG_0,indG_f);
		Gi=Gi.rows(indG_0,indG_f);
	}
	if (wn(1)<0 )
	{
		wn=abs(wn);
		Gi=-Gi;
	}
	if (wn(0)>wn(Nn-1))
	{
		wn_inverted=true;
		wn=flipud(wn);
		Gr=flipud(Gr);
		Gi=flipud(Gi);
	}
	if (Gr.max()*Gr.min()<0)
	{
		cout<<"error: Re[G] must not change sign.\n";
		return false;
	}
	if (Gr.max()>0) signG=-1;
	Gr=signG*Gr;
	Gi=signG*Gi;
	
	G.zeros(Nn);
	G.set_real(Gr);
	G.set_imag(Gi);
	
	for (j=0; j<Nn-1; j++)
	{
		if ((wn(j+1)-wn(j))<0)
		{
			cout<<"error: Matsubara frequency is not strictly increasing\n";
			return false;
		}
	}
	
	if (col_Gi>0)
	{
		uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
		uvec odd_ind=even_ind+1;
		
		Gchi2.zeros(2*Nn);
		Gchi2.rows(even_ind)=Gr;
		Gchi2.rows(odd_ind)=Gi;
	}
	else
	{
		Gchi2=Gr;
	}
	
	double tem_wn=wn(1)/(2*PI);
	
	n.zeros(Nn);
	vec ntmp;
	if (tem_in.size()==0)
	{
		if (  (wn(1)-floor(wn(1)))==0  && (wn(2)-floor(wn(2)))==0 )
		{
			cout<<"If the Matsubara frequency are given by index, temperature must be provided.\n";
			return false;
		}
		tem=tem_wn;
		ntmp=round(wn/(2*PI*tem));
		for (j=0; j<Nn; j++)	n(j)=ntmp(j);
	}
	else if ( (abs(tem-tem_wn)/tem)>tol_tem )
	{
		if ( wn(1)-floor(wn(1)) )
		{
			ntmp=round(wn/(2*PI*tem_wn));
			for (j=0; j<Nn; j++)	n(j)=ntmp(j);
			cout<<"warning: temperature in file "<<data_file_name<<" is different from temperature given in parameter file "<<input_params_file_name<<endl;
			cout<<"provided temperature: "<<tem<<endl;
			cout<<"temperature extracted from first finite Matsubara frequency: "<<tem_wn<<endl;
		}
		else if ( (wn(2)-floor(wn(2)))==0 )
		{
			ntmp=round(wn);
			for (j=0; j<Nn; j++) n(j)=ntmp(j);
		}
		else
		{
			cout<<"oops! First finite Matsubara frequency is an integer but second is not!\n";
			n=linspace<uvec>(0,Nn-1,Nn);
		}
	}
	else
	{
		ntmp=round(wn/(2*PI*tem_wn));
		for (j=0; j<Nn; j++) n(j)=ntmp(j);
	}
	
	cout<<"temperature: "<<tem<<endl;
	
	wn=2*PI*tem*conv_to<vec>::from(n);
	
	if (omega_n_trunc_in.size())
	{
		j=Nn-1;
		while (j>Nn_min && wn(j)>omega_n_trunc) j--;
		if (j<Nn-1 && j>Nn_min)
		{
			n=n.rows(0,j);
			wn=wn.rows(0,j);
			G=G.rows(0,j);
			Gr=Gr.rows(0,j);
			Gi=Gi.rows(0,j);
			if (col_Gi>0)
				Gchi2=Gchi2.rows(0,2*j+1);
			else
				Gchi2=Gchi2.rows(0,j);
			Nn=j+1;
		}
		else if (j==Nn_min)
		{
			cout<<"warning: truncation frequency ignored because too small.\n";
		}
	}
	
	cout<<"Number of Matsubara frequencies in the Green function: "<<Nn<<endl;
	
	if (displ_prep_figs)
	{
		if (col_Gi>0)
		{
			graph_2D g1, g2;
			
			//		graph_2D::show_commands(true);
			
			char ttlRe[]="Real part of data";
			char ttlIm[]="Imaginary part of data";
			char xl[]="$\\\\omega_n$";
			char yl1[]="$Re[G]$";
			char yl2[]="$Im[G]$";
			char attr1[]="'o-', color='b', markeredgecolor='b', markerfacecolor='b'";
			char attr2[]="'o-', color='r', markeredgecolor='r', markerfacecolor='r'";
			
			g1.add_title(ttlRe);
			plot(g1,wn,Gr,xl,yl1,attr1);
			g2.add_title(ttlIm);
			plot(g2,wn,Gi,xl,yl2,attr2);
			
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		else
		{
			graph_2D g1;
			
			//		graph_2D::show_commands(true);
			
			char ttlRe[]="Real part of data";
			char xl[]="$\\\\omega_n$";
			char yl1[]="$Re[G]$";
			char attr1[]="'o-', color='b', markeredgecolor='b', markerfacecolor='b'";
			
			g1.add_title(ttlRe);
			plot(g1,wn,Gr,xl,yl1,attr1);
			
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
	}
	
	return true;
}

void OmegaMaxEnt_data::set_Ntau_to_pow_2()
{
	cout<<"setting the number of imaginary time slices to a power of 2\n";
	
	vec cfs_LC(4);
	vec LC(2);
	if (!boson)
	{
		LC(0)=M1;
		LC(1)=-M2;
		cfs_LC=ones<vec>(4);
	}
	else
	{
		LC(0)=M1;
		LC(1)=-M2;
		cfs_LC(0)=1;
		cfs_LC(1)=-1;
		cfs_LC(2)=1;
		cfs_LC(3)=-1;
	}
	
	vec cfs_Gtau;
	spline_coeffs_LC(tau,Gtau,LC,cfs_LC,cfs_Gtau);
	
	vec tau_in=tau;
	vec Gtau_in=Gtau;
	int Ntau_in=Ntau;
	
	Ntau=(int)floor(log2(Ntau));
	tau=linspace<vec>(0,1/tem,Ntau+1);
	Gtau.zeros(Ntau+1);
	
	cout<<"number of imaginary time slices: "<<Ntau<<endl;

	spline_val(tau, tau_in, cfs_Gtau, Gtau);

/*
	graph_2D g1;
	const char attr1[]="ro-";
	const char attr2[]="bo-";
	
	g1.add_data(tau_in.memptr(),Gtau_in.memptr(),Ntau_in+1);
	g1.add_attribute(attr1);
	g1.add_data(tau.memptr(),Gtau.memptr(),Ntau+1);
	g1.add_attribute(attr2);
	g1.curve_plot();
	graph_2D::show_figures();
 */
}

bool OmegaMaxEnt_data::Fourier_transform_G_tau()
{
	int j, m, q;
	dcomplex I(0,1);
	double M1_FT, M2_FT;
	int Ntau_max_LC=1024;
	
//	bool use_fftw=true;
	
	cout<<"computing Fourier transform of G(tau)...\n";
	
	Nn=Ntau/2;
	n=linspace<uvec>(0,Nn-1,Nn);
	if (!boson)
		wn=PI*tem*conv_to<vec>::from(2*n+1);
	else
		wn=2*PI*tem*conv_to<vec>::from(n);
	
	uvec l=linspace<uvec>(0,Ntau-1,Ntau);
	vec p=linspace<vec>(0,Ntau-1,Ntau);
	vec cfs_Gtau(4*Ntau);


	if (Ntau<=Ntau_max_LC || !dG_dtau_computed)
	{
		vec cfs_LC(4);
		vec LC(2);
		if (!boson)
		{
			LC(0)=M1;
			LC(1)=-M2;
			cfs_LC=ones<vec>(4);
		}
		else
		{
			LC(0)=M1;
			LC(1)=-M2;
			cfs_LC(0)=1;
			cfs_LC(1)=-1;
			cfs_LC(2)=1;
			cfs_LC(3)=-1;
		}
		
		spline_coeffs_LC(tau,Gtau,LC,cfs_LC,cfs_Gtau);
		
		M1_FT=M1;
		M2_FT=M2;
	}
	else
	{
		cfs_Gtau(0)=dG_tau(0);
		cfs_Gtau(1)=dG_tau(1);
		spline_coeffs(tau.memptr(), Gtau.memptr(), Ntau+1, cfs_Gtau.memptr());
		
		double dtau=tau(1)-tau(0);
		M1_FT=dG_tau(0)+dG_tau(1);
		M2_FT=-(2*cfs_Gtau(1)+6*cfs_Gtau(4*Ntau-4)*dtau+2*cfs_Gtau(4*Ntau-3));
	//	cout<<"M1_FT: "<<M1_FT<<endl;
	//	cout<<"M2_FT: "<<M2_FT<<endl;
	}
	
	
/*
	int Ntau2=4*Ntau+1;
	vec tau2=linspace<vec>(0,1.0/tem,Ntau2);
	vec Gtau2;
	spline_val(tau2, tau, cfs_Gtau, Gtau2);
	graph_2D g1;
	char xl[]="tau", yl[]="G";
	char attr1[]="'o',markerfacecolor='none'";
	char attr2[]="'s-',markerfacecolor='none'";
	g1.set_axes_labels(xl,yl);
	g1.add_data(tau.memptr(),Gtau.memptr(),Ntau+1);
	g1.add_attribute(attr1);
	g1.add_data(tau2.memptr(),Gtau2.memptr(),Ntau2);
	g1.add_attribute(attr2);
	g1.curve_plot();
	graph_2D::show_figures();
*/

	G.zeros(Nn);
	Gr.zeros(Nn);
	Gi.zeros(Nn);
	if (boson)
	{
		double a, b, c, d, dtau;
		for (j=0; j<Ntau; j++)
		{
			a=cfs_Gtau(4*j);
			b=cfs_Gtau(4*j+1);
			c=cfs_Gtau(4*j+2);
			d=Gtau(j);
			dtau=tau(j+1)-tau(j);
			Gr(0)=Gr(0)+a*pow(dtau,4)/4+b*pow(dtau,3)/3+c*pow(dtau,2)/2+d*dtau;
		}
	}
	
	cx_vec TF_d3Gtau;
	cx_vec d3Gtau=zeros<cx_vec>(Ntau);
	
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	uvec odd_ind=even_ind+1;
	
//	clock_t tc=clock();
	
/*
	if (use_fftw)
	{
	//	unsigned fftflag = FFTW_MEASURE;
		unsigned fftflag = FFTW_ESTIMATE;
		
		fftw_plan fftplan_G=fftw_plan_dft_1d(Ntau, reinterpret_cast<fftw_complex *>(d3Gtau.memptr()), reinterpret_cast<fftw_complex *>(d3Gtau.memptr()), FFTW_BACKWARD, fftflag);
		
		d3Gtau.set_real(6*cfs_Gtau.rows(4*l));
		d3Gtau.set_imag(zeros<vec>(Ntau));
		if (!boson)
		{
			d3Gtau=exp(I*PI*p/Ntau) % d3Gtau;
			fftw_execute(fftplan_G);
			TF_d3Gtau=(1.0-exp(I*(2*p+1)*PI/Ntau)) % d3Gtau;
			G=-I*M0/wn-M1/pow(wn,2)+I*M2/pow(wn,3)+TF_d3Gtau.rows(0,Nn-1)/pow(wn,4);
		}
		else
		{
			fftw_execute(fftplan_G);
			TF_d3Gtau=(1.0-exp(I*(2*p)*PI/Ntau)) % d3Gtau;
			G.rows(1,Nn-1)=-I*M0/wn.rows(1,Nn-1)-M1/pow(wn.rows(1,Nn-1),2)+I*M2/pow(wn.rows(1,Nn-1),3)+TF_d3Gtau.rows(1,Nn-1)/pow(wn.rows(1,Nn-1),4);
			G(0)=cx_double(Gr(0),0);
		}
		Gr=real(G);
		Gi=imag(G);
		
		if (col_Gi>0)
		{
			Gchi2.zeros(2*Nn);
			Gchi2.rows(even_ind)=Gr;
			Gchi2.rows(odd_ind)=Gi;
		}
		else
		{
			Gchi2=Gr;
		}
	}
 
	*/
	
	/*
		cx_mat Ctau_n(Ntau,Ntau), Ctau_n_R(Ntau,Ntau), Ctau_n_I(Ntau,Ntau), Cmn_R(Ntau,Ntau), Cmn_I(Ntau,Ntau), Ctau2;
		
		int fftsize[1];
		fftsize[0]=Ntau;
		
		cx_vec Cvec(Ntau);
		
		fftw_plan fftplan_Cv=fftw_plan_dft_1d(Ntau, reinterpret_cast<fftw_complex *>(Cvec.memptr()), reinterpret_cast<fftw_complex *>(Cvec.memptr()), FFTW_BACKWARD, fftflag);
		
		fftw_plan fftplan_C_1=fftw_plan_many_dft(1,fftsize,Ntau,reinterpret_cast<fftw_complex *>(Ctau_n.memptr()),NULL,1,Ntau,reinterpret_cast<fftw_complex *>(Ctau_n.memptr()),NULL,1,Ntau,FFTW_BACKWARD, fftflag);
		
		fftw_plan fftplan_C_2=fftw_plan_many_dft(1,fftsize,Nn,reinterpret_cast<fftw_complex *>(Ctau_n.memptr()),NULL,1,Ntau,reinterpret_cast<fftw_complex *>(Ctau_n.memptr()),NULL,1,Ntau,FFTW_BACKWARD, fftflag);
		
		cout<<setiosflags(ios::left);
		if (!boson)
		{
			cx_mat Mph=diagmat(exp(I*PI*p/Ntau));
			Ctau2=Ctau*Mph;
			
			for (j=0; j<Ntau; j++)
			{
				Cvec=Ctau2.row(j).t();
				fftw_execute(fftplan_Cv);
				Ctau_n.row(j)=Cvec.t()/(Ntau*tem);
			//	Ctau_n.row(j)=ifft(Ctau2.row(j))/tem;
			}
			cx_mat Ctau_n_R=Mph*real(Ctau_n);
			cx_mat Ctau_n_I=Mph*imag(Ctau_n);
			cx_mat Cmn_R(Ntau,Ntau), Cmn_I(Ntau,Ntau);
			for (j=0; j<Nn; j++)
			{
				Cvec=Ctau_n_R.col(j);
				fftw_execute(fftplan_Cv);
				Cmn_R.col(j)=Cvec/(Ntau*tem);
			//	Cmn_R.col(j)=ifft(Ctau_n_R.col(j))/tem;
				Cvec=Ctau_n_I.col(j);
				fftw_execute(fftplan_Cv);
				Cmn_I.col(j)=Cvec/(Ntau*tem);
			//  Cmn_I.col(j)=ifft(Ctau_n_I.col(j))/tem;
			}
			CRR=real(Cmn_R);
			CRI=real(Cmn_I);
			CII=imag(Cmn_I);
	 
	//		fftw_execute_dft(fftplan_C_1, reinterpret_cast<fftw_complex *>(Ctau2.memptr()), reinterpret_cast<fftw_complex *>(Ctau_n.memptr()));
	//		Ctau_n_R=trans(Mph*real(Ctau_n))/(Ntau*tem);
	//		Ctau_n_I=trans(Mph*imag(Ctau_n))/(Ntau*tem);
	//		fftw_execute_dft(fftplan_C_2, reinterpret_cast<fftw_complex *>(Ctau_n_R.memptr()), reinterpret_cast<fftw_complex *>(Cmn_R.memptr()));
	//		fftw_execute_dft(fftplan_C_2, reinterpret_cast<fftw_complex *>(Ctau_n_I.memptr()), reinterpret_cast<fftw_complex *>(Cmn_I.memptr()));
	//		CRR=real(Cmn_R.t())/(Ntau*tem);
	//		CRI=real(Cmn_I.t())/(Ntau*tem);
	//		CII=imag(Cmn_I.t())/(Ntau*tem);
		
	
		}
		else
		{
			Ctau2.set_real(Ctau);
			
			fftw_execute_dft(fftplan_C_1, reinterpret_cast<fftw_complex *>(Ctau2.memptr()), reinterpret_cast<fftw_complex *>(Ctau_n.memptr()));
			Ctau_n_R.zeros();
			Ctau_n_I.zeros();
			Ctau_n_R.set_real(real(Ctau_n.t())/(Ntau*tem));
			Ctau_n_I.set_real(imag(Ctau_n.t())/(Ntau*tem));
			fftw_execute_dft(fftplan_C_2, reinterpret_cast<fftw_complex *>(Ctau_n_R.memptr()), reinterpret_cast<fftw_complex *>(Cmn_R.memptr()));
			fftw_execute_dft(fftplan_C_2, reinterpret_cast<fftw_complex *>(Ctau_n_I.memptr()), reinterpret_cast<fftw_complex *>(Cmn_I.memptr()));
			CRR=real(Cmn_R.t())/(Ntau*tem);
			CRI=real(Cmn_I.t())/(Ntau*tem);
			CII=imag(Cmn_I.t())/(Ntau*tem);
			CII(0,0)=CII(1,1);
			
		}
		fftw_destroy_plan(fftplan_G);
		fftw_destroy_plan(fftplan_C_1);
		fftw_destroy_plan(fftplan_C_2);
		fftw_destroy_plan(fftplan_Cv);
	 */
	 
	
	d3Gtau.set_real(6*cfs_Gtau.rows(4*l));
	if (!boson)
	{
		d3Gtau=exp(I*PI*p/Ntau) % d3Gtau;
		TF_d3Gtau=Ntau*(1.0-exp(I*(2*p+1)*PI/Ntau)) % ifft(d3Gtau);
		G=-I*M0/wn-M1_FT/pow(wn,2)+I*M2_FT/pow(wn,3)+TF_d3Gtau.rows(0,Nn-1)/pow(wn,4);
	}
	else
	{
		TF_d3Gtau=Ntau*(1.0-exp(I*(2*p)*PI/Ntau)) % ifft(d3Gtau);
		G.rows(1,Nn-1)=-I*M0/wn.rows(1,Nn-1)-M1/pow(wn.rows(1,Nn-1),2)+I*M2/pow(wn.rows(1,Nn-1),3)+TF_d3Gtau.rows(1,Nn-1)/pow(wn.rows(1,Nn-1),4);
		G(0)=cx_double(Gr(0),0);
	}
	Gr=real(G);
	Gi=imag(G);
	
	if (col_Gi>0)
	{
		Gchi2.zeros(2*Nn);
		Gchi2.rows(even_ind)=Gr;
		Gchi2.rows(odd_ind)=Gi;
	}
	else
	{
		Gchi2=Gr;
	}
	
	cout<<"Fourier transform of G computed.\n";
	
	struct stat file_stat;
	
	string output_dir_TF=input_dir;
	output_dir_TF+="Fourier_transformed_data/";
	
	if (stat(output_dir_TF.c_str(),&file_stat)) mkdir(output_dir_TF.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
	
	mat save_mat_G=zeros<mat>(Nn,3);
	save_mat_G.col(0)=wn;
	save_mat_G.col(1)=Gr;
	save_mat_G.col(2)=Gi;
	string G_file_name("Fourier_transform_G_ascii.dat");
	string complete_file_name_G(output_dir_TF);
	complete_file_name_G+=G_file_name;
	save_mat_G.save(complete_file_name_G.c_str(),raw_ascii);
	
	G_file_name.assign("Fourier_transform_G.dat");
	complete_file_name_G.assign(output_dir_TF);
	complete_file_name_G+=G_file_name;
	save_mat_G.save(complete_file_name_G.c_str(),arma_binary);
	
	if (displ_prep_figs)
	{
		graph_2D g1, g2;
		
		//		graph_2D::show_commands(true);
		
		char xl[]="$\\\\omega_n$";
		char yl1[]="$Re[G]$";
		char yl2[]="$Im[G]$";
		char attr1[]="'o-', color='b', markeredgecolor='b', markerfacecolor='b'";
		char attr2[]="'o-', color='r', markeredgecolor='r', markerfacecolor='r'";
		
		plot(g1,wn,Gr,xl,yl1,attr1);
		plot(g2,wn,Gi,xl,yl2,attr2);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	
	if (error_file.size() || covar_tau_file.size())
	{
		cout<<"Fourier transforming covariance matrix\n";
	//	cout<<"Nn: "<<Nn<<endl;
		cout<<setiosflags(ios::left);
		if (!boson)
		{
			cx_mat Mph=diagmat(exp(I*PI*p/Ntau));
			cx_mat Ctau2=Ctau*Mph;
			cx_mat Ctau_n(Ntau,Ntau);
			/*
			 if (error_file.size())
			 {
			 rowvec m=linspace<rowvec>(0,Ntau-1,Ntau);
			 cx_rowvec vexp;
			 for (j=0; j<Ntau; j++)
			 {
				vexp=exp(I*2.0*PI*(j*m)/Ntau);
				Ctau_n.row(j)=Ctau2(j,j)*vexp/(tem*Ntau);
			 }
			 }
			 else
			 {
			 for (j=0; j<Ntau; j++) Ctau_n.row(j)=ifft(Ctau2.row(j))/tem;
			 }
			 */
			for (j=0; j<Ntau; j++) Ctau_n.row(j)=ifft(Ctau2.row(j))/tem;
			cx_mat Ctau_n_R=Mph*real(Ctau_n);
			cx_mat Ctau_n_I=Mph*imag(Ctau_n);
			cx_mat Cmn_R(Ntau,Ntau), Cmn_I(Ntau,Ntau);
			for (j=0; j<Nn; j++)
			{
				Cmn_R.col(j)=ifft(Ctau_n_R.col(j))/tem;
				Cmn_I.col(j)=ifft(Ctau_n_I.col(j))/tem;
			}
			CRR=real(Cmn_R);
			CRI=real(Cmn_I);
			CII=imag(Cmn_I);
		}
		else
		{
			cx_mat Ctau_n(Ntau,Ntau), Ctau_n_R(Ntau,Ntau), Ctau_n_I(Ntau,Ntau), Cmn_R(Ntau,Ntau), Cmn_I(Ntau,Ntau);
			/*
			 if (error_file.size())
			 {
			 rowvec m=linspace<rowvec>(0,Ntau-1,Ntau);
			 cx_rowvec vexp;
			 for (j=0; j<Ntau; j++)
			 {
				vexp=exp(-I*2.0*PI*(j*m)/Ntau);
				Ctau_n.row(j)=Ctau(j,j)*vexp/(tem*Ntau);
			 }
			 }
			 else
			 {
			 for (j=0; j<Ntau; j++) Ctau_n.row(j)=fft(Ctau.row(j))/(Ntau*tem);
			 }
			 */
			for (j=0; j<Ntau; j++) Ctau_n.row(j)=fft(Ctau.row(j))/(Ntau*tem);
			Ctau_n_R.zeros();
			Ctau_n_I.zeros();
			Ctau_n_R.set_real(real(Ctau_n));
			Ctau_n_I.set_real(imag(Ctau_n));
			for (j=0; j<Nn; j++)
			{
				Cmn_R.col(j)=ifft(Ctau_n_R.col(j))/tem;
				Cmn_I.col(j)=-ifft(Ctau_n_I.col(j))/tem;
			}
			CRR=real(Cmn_R);
			CRI=real(Cmn_I);
			CII=imag(Cmn_I);
			CII(0,0)=CII(1,1);
		}
		
		string cov_file_RR("covar_ReRe.dat");
		string complete_file_name_CRR(output_dir_TF);
		complete_file_name_CRR+=cov_file_RR;
		CRR.save(complete_file_name_CRR.c_str(),arma_binary);
		
		string cov_file_RI("covar_ReIm.dat");
		string complete_file_name_CRI(output_dir_TF);
		complete_file_name_CRI+=cov_file_RI;
		CRI.save(complete_file_name_CRI.c_str(),arma_binary);
		
		string cov_file_II("covar_ImIm.dat");
		string complete_file_name_CII(output_dir_TF);
		complete_file_name_CII+=cov_file_II;
		CII.save(complete_file_name_CII.c_str(),arma_binary);
//	}
	
//	tc=clock()-tc;
//	cout<<"time for Fourier transforming G: "<<tc/CLOCKS_PER_SEC<<endl;
	
		if (col_Gi>0)
		{
			COV.zeros(2*Nn,2*Nn);
			COV(even_ind,even_ind)=CRR.submat(0,0,Nn-1,Nn-1);
			COV(odd_ind,odd_ind)=CII.submat(0,0,Nn-1,Nn-1);
			COV(even_ind,odd_ind)=CRI.submat(0,0,Nn-1,Nn-1);
			COV(odd_ind,even_ind)=trans(CRI.submat(0,0,Nn-1,Nn-1));
		}
		else
		{
			COV=CRR.submat(0,0,Nn-1,Nn-1);
		}
		COV=0.5*(COV+COV.t());
	}
	else if (col_Gi>0)
	{
		if (!set_covar_G_omega_n())
		{
			cout<<"covariance definition failed\n";
			return false;
		}
	}
	else
	{
		set_covar_chi_omega_n();
	}
	
	cout<<"Fourier transform of covariance matrix computed.\n";

	jfit=0;
	if (!cutoff_wn_in.size())
	{
		int NC=3;
		
		if (col_Gi>0)
		{
			vec C1b(Nn-2), C2b(Nn-2), C3b(Nn-2), C4b(Nn-2);
			double wn1, wn2, reG1, reG2, imG1, imG2, denom2;
			for (j=1; j<Nn-1; j++)
			{
				wn1=wn(j);
				wn2=wn(j+1);
				reG1=Gr(j);
				reG2=Gr(j+1);
				imG1=Gi(j);
				imG2=Gi(j+1);
				denom2=wn1*wn1 - wn2*wn2;
				C1b(j-1)=-(imG1*pow(wn1,3) - imG2*pow(wn2,3))/denom2;
				C2b(j-1)=-(reG1*pow(wn1,4) - reG2*pow(wn2,4))/denom2;
				C3b(j-1)=((wn1*wn1)*(wn2*wn2)*(-imG1*wn1 + imG2*wn2))/denom2;
				C4b(j-1)=(-reG1*pow(wn1,4)*pow(wn2,2) + reG2*pow(wn1,2)*pow(wn2,4))/denom2;
			}
			
			if (displ_adv_prep_figs)
			{
				graph_2D g1, g2;
				
				vec x=wn.rows(1,Nn-2);
				char xl[]="fit frequency $\\\\omega_n$";
				char yl[]="$M_0$";
				char yl2[]="$M_1$";
				
				plot(g1, x, C1b, xl, yl);
				plot(g2, x, C2b, xl, yl2);
				
				//cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			int p, Nfitmin, jfitmin, jfitmax, NNfit, Nfit;
			jfitmin=2;
			Nfitmin=2*NC+4;
			jfitmax=Nn-Nfitmin;
			NNfit=jfitmax-jfitmin+1;
			
			vec M0v(NNfit), M1v(NNfit), M2v(NNfit), M3v(NNfit);
			mat invCG, A, X, CG;
			vec Mtmp;
			
			for (jfit=jfitmin; jfit<=jfitmax; jfit++)
			{
				Nfit=Nn_fit_max;
				if ((Nn-jfit+1)<Nn_fit_max)
					Nfit=Nn-jfit+1;
				
				X.zeros(2*Nfit,2*NC);
				
				for (j=1; j<=NC; j++)
				{
					for (p=jfit; p<=jfit+Nfit-1; p++)
					{
						X(2*(p-jfit),2*j-1)=pow(-1,j)/pow(wn(p-1),2*j);
						X(2*(p-jfit)+1,2*j-2)=pow(-1,j)/pow(wn(p-1),2*j-1);
					}
				}
				
				CG=COV.submat(2*jfit-2,2*jfit-2,2*(jfit+Nfit-1)-1,2*(jfit+Nfit-1)-1);
				
				invCG=inv(CG);
				//	invCG=inv_sympd(CG);
				A=trans(X)*invCG*X;
				A=0.5*(A+A.t());
				Mtmp=trans(X)*invCG*Gchi2.rows(2*jfit-2,2*(jfit+Nfit-1)-1);
				
				Mtmp=solve(A,Mtmp);
				
				M0v(jfit-jfitmin)=Mtmp(0);
				M1v(jfit-jfitmin)=Mtmp(1);
				M2v(jfit-jfitmin)=Mtmp(2);
				M3v(jfit-jfitmin)=Mtmp(3);
				
			}
			
			int Nv=Nn/16;
			if (Nv<2) Nv=2;
			
			vec varM0(NNfit-2*Nv), varM1(NNfit-2*Nv), varM2(NNfit-2*Nv), varM3(NNfit-2*Nv);
			
			for (j=Nv; j<NNfit-Nv; j++)
			{
				varM0(j-Nv)=var(M0v.rows(j-Nv,j+Nv));
				varM1(j-Nv)=var(M1v.rows(j-Nv,j+Nv));
				varM2(j-Nv)=var(M2v.rows(j-Nv,j+Nv));
				varM3(j-Nv)=var(M3v.rows(j-Nv,j+Nv));
			}
			
			if (displ_adv_prep_figs)
			{
				graph_2D g1, g2, g3, g4;
				
				vec x=wn.rows(Nv,NNfit-Nv-1);
				
				char xl[]="fit starting frequency $\\\\omega_n$";
				char yl[]="varM0";
				char yl2[]="varM1";
				char yl3[]="varM2";
				char yl4[]="varM3";
				
				plot(g1, x, varM0, xl, yl);
				plot(g2, x, varM1, xl, yl2);
				plot(g3, x, varM2, xl, yl3);
				plot(g4, x, varM3, xl, yl4);
				
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			uword j0, j1, j2, j3;
			varM0.min(j0);
			varM1.min(j1);
			varM2.min(j2);
			varM3.min(j3);
			
			j0=j0+Nv;
			j1=j1+Nv;
			j2=j2+Nv;
			j3=j3+Nv;
			
			Mfit.zeros(4);
			Mfit(0)=mean(M0v.rows(j0-Nv,j0+Nv));
			Mfit(1)=mean(M1v.rows(j1-Nv,j1+Nv));
			Mfit(2)=mean(M2v.rows(j2-Nv,j2+Nv));
			Mfit(3)=mean(M3v.rows(j3-Nv,j3+Nv));
			
			j=Nv;
			if (!boson)
			{
				while ( j<j0 && (abs(mean(M0v.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(M0v.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) )
				{
					j=j+1;
				}
			}
			else
			{
				while ( j<j0 && (abs(mean(M1v.rows(j-Nv,j+Nv))-Mfit(1))/Mfit(1)>tol_mean_C1 || stddev(M1v.rows(j-Nv,j+Nv))/Mfit(1)>tol_std_C1) )
				{
					j=j+1;
				}
			}
			jfit=j+jfitmin-1;
			
		/*
			j=Nv;
			if (!boson)
			{
				while ((abs(mean(C1b.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(C1b.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) && j<Nn-Nv-3)
				{
					j=j+1;
				}
			}
			else
			{
				while ((abs(mean(C2b.rows(j-Nv,j+Nv))-Mfit(1))/Mfit(1)>tol_mean_C1 || stddev(C2b.rows(j-Nv,j+Nv))/Mfit(1)>tol_std_C1) && j<Nn-Nv-3)
				{
					j=j+1;
				}
			}
			jfit=j;
		*/
			cout<<"frequency range of asymptotic behavior: "<<wn(jfit-1)<<" to "<<wn(Nn-1)<<" (indices "<<jfit-1<<" to "<<Nn-1<<")"<<endl;
			
			if (displ_prep_figs)
			{
				graph_2D g1, g2, g3, g4;
				
				vec x=wn.rows(jfitmin-1,jfitmax-1);
				
				char xl[]="fit starting frequency $\\\\omega_n$";
				char yl[]="$M_0$";
				char yl2[]="$M_1$";
				char yl3[]="$M_2$";
				char yl4[]="$M_3$";
				
				plot(g1, x, M0v, xl, yl);
				plot(g2, x, M1v, xl, yl2);
				plot(g3, x, M2v, xl, yl3);
				plot(g4, x, M3v, xl, yl4);
				
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
		}
		else
		{
			vec C2b(Nn-2), C4b(Nn-2);
			double wn1, wn2, reG1, reG2, denom2;
			for (j=1; j<Nn-1; j++)
			{
				wn1=wn(j);
				wn2=wn(j+1);
				reG1=Gr(j);
				reG2=Gr(j+1);
				denom2=wn1*wn1 - wn2*wn2;
				C2b(j-1)=-(reG1*pow(wn1,4) - reG2*pow(wn2,4))/denom2;
				C4b(j-1)=(-reG1*pow(wn1,4)*pow(wn2,2) + reG2*pow(wn1,2)*pow(wn2,4))/denom2;
			}
			
		/*
			if (displ_prep_figs)
			{
				graph_2D g1;
				
				vec x=wn.rows(1,Nn-2);
				char xl[]="fit frequency $\\\\omega_n$";
				char yl[]="$M_1$";
				
				plot(g1, x, C2b, xl, yl);
				
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
		 */
			
			int p, Nfitmin, jfitmin, jfitmax, NNfit, Nfit;
			jfitmin=2;
			Nfitmin=2*NC+4;
			jfitmax=Nn-Nfitmin;
			NNfit=jfitmax-jfitmin+1;
			
			vec M1v(NNfit), M3v(NNfit);
			mat invCG, A, X, CG;
			vec Mtmp;
			
			for (jfit=jfitmin; jfit<=jfitmax; jfit++)
			{
				Nfit=Nn_fit_max;
				if ((Nn-jfit+1)<Nn_fit_max)
					Nfit=Nn-jfit+1;
				
				X.zeros(Nfit,NC);
				
				for (j=1; j<=NC; j++)
				{
					for (p=jfit; p<=jfit+Nfit-1; p++)
					{
						X((p-jfit),j-1)=pow(-1,j)/pow(wn(p-1),2*j);
					}
				}
				
				CG=COV.submat(jfit-1,jfit-1,jfit+Nfit-2,jfit+Nfit-2);
				
				invCG=inv(CG);
				//		invCG=inv_sympd(CG);
				A=trans(X)*invCG*X;
				A=0.5*(A+A.t());
				Mtmp=trans(X)*invCG*Gchi2.rows(jfit-1,jfit+Nfit-2);
				
				Mtmp=solve(A,Mtmp);
				
				M1v(jfit-jfitmin)=Mtmp(0);
				M3v(jfit-jfitmin)=Mtmp(1);
				
			}
			
			int Nv=Nn/16;
			if (Nv<2) Nv=2;
			
			vec varM1(NNfit-2*Nv), varM3(NNfit-2*Nv);
			
			for (j=Nv; j<NNfit-Nv; j++)
			{
				varM1(j-Nv)=var(M1v.rows(j-Nv,j+Nv));
				varM3(j-Nv)=var(M3v.rows(j-Nv,j+Nv));
			}
			
			if (displ_adv_prep_figs)
			{
				graph_2D g1, g2;
				
				vec x=wn.rows(Nv,NNfit-Nv-1);
				
				char xl[]="fit starting frequency $\\\\omega_n$";
				char yl1[]="varM1";
				char yl2[]="varM3";
				
				plot(g1, x, varM1, xl, yl1);
				plot(g2, x, varM3, xl, yl2);
				
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			uword j1, j3;
			varM1.min(j1);
			varM3.min(j3);
			
			j1=j1+Nv;
			j3=j3+Nv;
			
			Mfit.zeros(2);
			Mfit(0)=mean(M1v.rows(j1-Nv,j1+Nv));
			Mfit(1)=mean(M3v.rows(j3-Nv,j3+Nv));
		
			j=Nv;
			while (j<j1 && (abs(mean(M1v.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(M1v.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) && j<Nn-Nv-3) j++;
			jfit=j+jfitmin-1;
			
		/*
			j=Nv;
			while ((abs(mean(C2b.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(C2b.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) && j<Nn-Nv-3) j++;
			jfit=j;
		*/
			
			cout<<"frequency range of asymptotic behavior: "<<wn(jfit-1)<<" to "<<wn(Nn-1)<<" (indices "<<jfit-1<<" to "<<Nn-1<<")"<<endl;
			
			if (displ_prep_figs)
			{
				graph_2D g1, g2;
				
				vec x=wn.rows(jfitmin-1,jfitmax-1);
				
				char xl[]="fit starting frequency $\\\\omega_n$";
				char yl1[]="$M_1$";
				char yl2[]="$M_3$";
				
				plot(g1, x, M1v, xl, yl1);
				plot(g2, x, M3v, xl, yl2);
				
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			
		}
	}
	
	if (Nn-jfit<Nn_as_min)
	{
		jfit=0;
		if (!moments_provided && maxM>0) maxM=0;
	}

	return true;
}

//compute derivatives of G(tau) at tau=0 and tau=beta
bool OmegaMaxEnt_data::compute_dG_dtau()
{
	cout<<"computing derivatives of G(tau) at tau=0 and tau=beta\n";
	
	int sgn=1;
	if (boson) sgn=-1;
	
	int Nfitmax_max=50;
	
	int Nv=1;
	int NvN=1;
	int npmin=2;
	
	int np, Nfit;
	int j, l;
	uword jvmin, lvmin;
	
	double Wtmp=std_omega;
	
	int Nfitmax=ceil(FNfitTauW*Ntau*tem/(abs(M1/M0)+Wtmp));
	if (Nfitmax>Ntau) Nfitmax=Ntau/2;
	if (Nfitmax>Nfitmax_max) Nfitmax=Nfitmax_max;
	
	mat X;
	int p;
	
	Nfit=Nfitmax;
	np=Nfit-1;
	X.zeros(Nfit,np+1);
	for (p=0; p<=np; p++)
		X.col(p)=pow(tau.rows(0,Nfit-1)/tau(Nfit-1),p);
	
	mat U, V;
	vec sK;
	svd(U,sK,V,X,"std");
	
	p=0;
	while (p<=np && sK(p)/sK(0)>R_sv_min) p++;
	Nfitmax=p;
	
	mat CG, invCG, AM;
	vec Gchi2tmp, BM, Mtmp;
	npmin=3;
	int Nfitmin=npmin+1;
	int NNfit=Nfitmax-Nfitmin+1;
	
	if (NNfit<5)
	{
		cout<<"compute_dG_dtau(): unable to determine dG/dtau. The imaginary time step can be either too small or too large. You can either change the step or increase parameter R_sv_min in file \"OmegaMaxEnt_other_params.dat\".\n";
		return false;
	}
	
	mat G0=zeros<mat>(NNfit,NNfit);
	mat dG0=zeros<mat>(NNfit,NNfit);
	mat d2G0=zeros<mat>(NNfit,NNfit);
	mat d3G0=zeros<mat>(NNfit,NNfit);
	mat Gb=zeros<mat>(NNfit,NNfit);
	mat dGb=zeros<mat>(NNfit,NNfit);
	mat d2Gb=zeros<mat>(NNfit,NNfit);
	mat d3Gb=zeros<mat>(NNfit,NNfit);
	
	for (Nfit=Nfitmin; Nfit<=Nfitmax; Nfit++)
	{
		for (np=npmin; np<Nfit; np++)
		{
			X=zeros<mat>(Nfit,np+1);
			for (p=0; p<=np; p++)
				X.col(p)=pow(tau.rows(0,Nfit-1)/tau(Nfit-1),p);
			
			Gchi2tmp=Gtau.rows(0,Nfit-1);
			CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1);
			invCG=inv(CG);
			//	invCG=inv_sympd(CG);
			AM=(X.t())*invCG*X;
			BM=(X.t())*invCG*Gchi2tmp;
			Mtmp=solve(AM,BM);
			G0(Nfit-np-1,np-npmin)=Mtmp(0);
			dG0(Nfit-np-1,np-npmin)=Mtmp(1)/tau(Nfit-1);
			d2G0(Nfit-np-1,np-npmin)=2*Mtmp(2)/pow(tau(Nfit-1),2);
			d3G0(Nfit-np-1,np-npmin)=6*Mtmp(3)/pow(tau(Nfit-1),3);
			
			Gchi2tmp=flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			CG=flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
			invCG=inv(CG);
			//	invCG=inv_sympd(CG);
			AM=(X.t())*invCG*X;
			BM=(X.t())*invCG*Gchi2tmp;
			Mtmp=solve(AM,BM);
			Gb(Nfit-np-1,np-npmin)=Mtmp(0);
			dGb(Nfit-np-1,np-npmin)=-Mtmp(1)/tau(Nfit-1);
			d2Gb(Nfit-np-1,np-npmin)=2*Mtmp(2)/pow(tau(Nfit-1),2);
			d3Gb(Nfit-np-1,np-npmin)=-6*Mtmp(3)/pow(tau(Nfit-1),3);
		}
	}
	
	int Nacc=(2*Nv+1)*(2*NvN+1);
	Nv=1;
	NvN=1;
	int jmin=NvN;
	int jmax=NNfit-NvN-2*Nv-1;
	int Nj=jmax-jmin+1;
	int lmin=Nv;
	int lmax=NNfit-2*NvN-Nv-1;
	int Nl=lmax-lmin+1;
	mat G0m=zeros<mat>(Nj,Nl);
	mat dG0m=zeros<mat>(Nj,Nl);
	mat d2G0m=zeros<mat>(Nj,Nl);
	mat d3G0m=zeros<mat>(Nj,Nl);
	mat Gbm=zeros<mat>(Nj,Nl);
	mat dGbm=zeros<mat>(Nj,Nl);
	mat d2Gbm=zeros<mat>(Nj,Nl);
	mat d3Gbm=zeros<mat>(Nj,Nl);
	mat var_G0=zeros<mat>(Nj,Nl);
	mat var_dG0=zeros<mat>(Nj,Nl);
	mat var_d2G0=zeros<mat>(Nj,Nl);
	mat var_d3G0=zeros<mat>(Nj,Nl);
	mat var_Gb=zeros<mat>(Nj,Nl);
	mat var_dGb=zeros<mat>(Nj,Nl);
	mat var_d2Gb=zeros<mat>(Nj,Nl);
	mat var_d3Gb=zeros<mat>(Nj,Nl);
	for (j=jmin; j<=jmax; j++)
	{
		for (l=lmin; l<=NNfit-1-j-Nv-NvN; l++)
		{
			G0m(j-jmin,l-lmin)=accu(G0.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			dG0m(j-jmin,l-lmin)=accu(dG0.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			d2G0m(j-jmin,l-lmin)=accu(d2G0.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			d3G0m(j-jmin,l-lmin)=accu(d3G0.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			var_G0(j-jmin,l-lmin)=accu(pow(G0.submat(j-NvN,l-Nv,j+NvN,l+Nv)-G0m(j-jmin,l-lmin),2))/Nacc;
			var_dG0(j-jmin,l-lmin)=accu(pow(dG0.submat(j-NvN,l-Nv,j+NvN,l+Nv)-dG0m(j-jmin,l-lmin),2))/Nacc;
			var_d2G0(j-jmin,l-lmin)=accu(pow(d2G0.submat(j-NvN,l-Nv,j+NvN,l+Nv)-d2G0m(j-jmin,l-lmin),2))/Nacc;
			var_d3G0(j-jmin,l-lmin)=accu(pow(d3G0.submat(j-NvN,l-Nv,j+NvN,l+Nv)-d3G0m(j-jmin,l-lmin),2))/Nacc;
			Gbm(j-jmin,l-lmin)=accu(Gb.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			dGbm(j-jmin,l-lmin)=accu(dGb.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			d2Gbm(j-jmin,l-lmin)=accu(d2Gb.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			d3Gbm(j-jmin,l-lmin)=accu(d3Gb.submat(j-NvN,l-Nv,j+NvN,l+Nv))/Nacc;
			var_Gb(j-jmin,l-lmin)=accu(pow(Gb.submat(j-NvN,l-Nv,j+NvN,l+Nv)-Gbm(j-jmin,l-lmin),2))/Nacc;
			var_dGb(j-jmin,l-lmin)=accu(pow(dGb.submat(j-NvN,l-Nv,j+NvN,l+Nv)-dGbm(j-jmin,l-lmin),2))/Nacc;
			var_d2Gb(j-jmin,l-lmin)=accu(pow(d2Gb.submat(j-NvN,l-Nv,j+NvN,l+Nv)-d2Gbm(j-jmin,l-lmin),2))/Nacc;
			var_d3Gb(j-jmin,l-lmin)=accu(pow(d3Gb.submat(j-NvN,l-Nv,j+NvN,l+Nv)-d3Gbm(j-jmin,l-lmin),2))/Nacc;
		}
	}
	
	double var_G0max=max(max(var_G0));
	double var_dG0max=max(max(var_dG0));
	double var_d2G0max=max(max(var_d2G0));
	double var_d3G0max=max(max(var_d3G0));
	double var_Gbmax=max(max(var_Gb));
	double var_dGbmax=max(max(var_dGb));
	double var_d2Gbmax=max(max(var_d2Gb));
	double var_d3Gbmax=max(max(var_d3Gb));
	
	for (j=1; j<Nj; j++)
	{
		var_G0.submat(j,Nl-j,j,Nl-1)=2*var_G0max*ones<rowvec>(j);
		var_dG0.submat(j,Nl-j,j,Nl-1)=2*var_dG0max*ones<rowvec>(j);
		var_d2G0.submat(j,Nl-j,j,Nl-1)=2*var_d2G0max*ones<rowvec>(j);
		var_d3G0.submat(j,Nl-j,j,Nl-1)=2*var_d3G0max*ones<rowvec>(j);
		var_Gb.submat(j,Nl-j,j,Nl-1)=2*var_Gbmax*ones<rowvec>(j);
		var_dGb.submat(j,Nl-j,j,Nl-1)=2*var_dGbmax*ones<rowvec>(j);
		var_d2Gb.submat(j,Nl-j,j,Nl-1)=2*var_d2Gbmax*ones<rowvec>(j);
		var_d3Gb.submat(j,Nl-j,j,Nl-1)=2*var_d3Gbmax*ones<rowvec>(j);
	}
	
	//	cout<<varM0<<endl;
	
	cout<<"moments computed from the derivatives:\n";
	
	vec G0b(2);
	var_G0.min(jvmin,lvmin);
	G0b(0)=G0m(jvmin,lvmin);
	var_Gb.min(jvmin,lvmin);
	G0b(1)=Gbm(jvmin,lvmin);
	cout<<"norm: "<<-(G0b(0)+sgn*G0b(1))<<endl;
	
	dG_tau.zeros(2);
	var_dG0.min(jvmin,lvmin);
	dG_tau(0)=dG0m(jvmin,lvmin);
	var_dGb.min(jvmin,lvmin);
	dG_tau(1)=dGbm(jvmin,lvmin);
	cout<<"first moment: "<<dG_tau(0)+sgn*dG_tau(1)<<endl;
	
	d2G_tau.zeros(2);
	var_d2G0.min(jvmin,lvmin);
	d2G_tau(0)=d2G0m(jvmin,lvmin);
	var_d2Gb.min(jvmin,lvmin);
	d2G_tau(1)=d2Gbm(jvmin,lvmin);
	cout<<"second moment: "<<-(d2G_tau(0)+sgn*d2G_tau(1))<<endl;
	
	d3G_tau.zeros(2);
	var_d3G0.min(jvmin,lvmin);
	d3G_tau(0)=d3G0m(jvmin,lvmin);
	var_d3Gb.min(jvmin,lvmin);
	d3G_tau(1)=d3Gbm(jvmin,lvmin);
	cout<<"third moment: "<<(d3G_tau(0)+sgn*d3G_tau(1))<<endl;
	
	/*
	if (displ_adv_prep_figs)
	{
		graph_2D g1, g2;
		char lgd_format[]="Nfit=%d", lgd_entry[20];
		
		vec pfit=linspace<vec>(npmin,Nfitmax-1,NNfit);
		vec vtmp=M0b.col(0);
		g1.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
		sprintf(lgd_entry,lgd_format,Nfitmin);
		g1.add_to_legend(lgd_entry);
		for (j=1; j<NNfit; j++)
		{
			vtmp=M0b.col(j);
			g1.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
			sprintf(lgd_entry,lgd_format,Nfitmin+j);
			g1.add_to_legend(lgd_entry);
		}
		g1.add_title("$M_0$");
		g1.curve_plot();
		
		vtmp=M1b.col(0);
		g2.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
		sprintf(lgd_entry,lgd_format,Nfitmin);
		g2.add_to_legend(lgd_entry);
		for (j=1; j<NNfit; j++)
		{
			vtmp=M1b.col(j);
			g2.add_data(pfit.memptr(),vtmp.memptr(),NNfit);
			sprintf(lgd_entry,lgd_format,Nfitmin+j);
			g2.add_to_legend(lgd_entry);
		}
		g2.add_title("$M_1$");
		g2.curve_plot();
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
		
	}
	*/
	
	return true;
}

bool OmegaMaxEnt_data::compute_moments_tau_fermions()
{
	//cout<<"COMPUTING MOMENTS with compute_moments_tau_fermions()\n";
	cout<<"COMPUTING MOMENTS\n";
	
	int Nfitmax_max=50;
	
	int Nv=1;
	int NvN=1;
	int npmin=2;
	int npmax=5;
	int Np=npmax-npmin+1;
	int DNfitmin=0;
	int DNfitmax=Ntau-npmax-1;
	if (16-npmax-1<DNfitmax) DNfitmax=16-npmax-1;
	int NDN=DNfitmax-DNfitmin+1;
	
	mat M0tmp=zeros<mat>(NDN,Np);
	mat M1tmp=zeros<mat>(NDN,Np);
	mat M2tmp=zeros<mat>(NDN,Np);
	
	vec Gtmp, pp;
	int DNfit, np, Nfit;
	for (DNfit=DNfitmin; DNfit<=DNfitmax; DNfit++)
	{
		for (np=npmin; np<=npmax; np++)
		{
			Nfit=np+1+DNfit;
			Gtmp=Gtau.rows(0,Nfit-1)+flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			polyfit(tau.rows(0,Nfit-1),Gtmp,np,0,pp);
			M0tmp(DNfit-DNfitmin,np-npmin)=-pp(np);
			M2tmp(DNfit-DNfitmin,np-npmin)=-2*pp(np-2);
			
			Gtmp=Gtau.rows(0,Nfit-1)-flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			polyfit(tau.rows(0,Nfit-1),Gtmp,np,0,pp);
			M1tmp(DNfit-DNfitmin,np-npmin)=pp(np-1);
		}
	}
	
	mat M0m=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat M1m=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat M2m=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat varM0=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat varM1=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	mat varM2=zeros<mat>(NDN-2*NvN,Np-2*Nv);
	int j, l;
	for (l=Nv; l<Np-Nv; l++)
	{
		for (j=NvN; j<NDN-NvN; j++)
		{
			M0m(j-NvN,l-Nv)=accu(M0tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M1m(j-NvN,l-Nv)=accu(M1tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M2m(j-NvN,l-Nv)=accu(M2tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			varM0(j-NvN,l-Nv)=accu(pow(M0tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M0m(j-NvN,l-Nv),2))/((2*Nv+1)*(2*NvN+1));
			varM1(j-NvN,l-Nv)=accu(pow(M1tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M1m(j-NvN,l-Nv),2))/((2*Nv+1)*(2*NvN+1));
			varM2(j-NvN,l-Nv)=accu(pow(M2tmp.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M2m(j-NvN,l-Nv),2))/((2*Nv+1)*(2*NvN+1));
		}
	}
	
	uword jvmin, lvmin;
	varM0.min(jvmin,lvmin);
	double M0_NP_tmp=M0m(jvmin,lvmin);
	varM1.min(jvmin,lvmin);
	double M1_NP_tmp=M1m(jvmin,lvmin);
	varM2.min(jvmin,lvmin);
	double M2_NP_tmp=M2m(jvmin,lvmin);
	
	double Wtmp;
	if (M2_NP_tmp/M0_NP_tmp > pow(M1_NP_tmp/M0_NP_tmp,2))
		Wtmp=sqrt(M2_NP_tmp/M0_NP_tmp-pow(M1_NP_tmp/M0_NP_tmp,2));
	else
		Wtmp=abs(M1_NP_tmp/M0_NP_tmp);
	
	int Nfitmax=ceil(FNfitTauW*Ntau*tem/(abs(M1_NP_tmp/M0_NP_tmp)+Wtmp));
	if (Nfitmax>Ntau) Nfitmax=Ntau/2;
	if (Nfitmax>Nfitmax_max) Nfitmax=Nfitmax_max;
	
	mat X;
	int p, pmax;
	
	Nfit=Nfitmax;
	np=Nfit-1;
	X.zeros(Nfit,np+1);
	for (p=0; p<=np; p++)
		X.col(p)=pow(tau.rows(0,Nfit-1)/tau(Nfit-1),p);
	
	mat U, V;
	vec sK;
	svd(U,sK,V,X,"std");
	
//	cout<<"sK(np)/sK(0): "<<sK(np)/sK(0)<<endl;
	
	p=0;
	while (p<=np && sK(p)/sK(0)>R_sv_min) p++;
	Nfitmax=p;
	
//	cout<<"Nfitmax: "<<Nfitmax<<endl;
	
	mat CG, invCG, AM;
	vec Gchi2tmp, BM, Mtmp;
	npmin=3;
	int Nfitmin=npmin+1;
	int NNfit=Nfitmax-Nfitmin+1;
	
	if (NNfit<5)
	{
		cout<<"compute_moments_tau_fermions(): unable to compute the moments from G(tau). The imaginary time step can be either too small or too large. You can either change the step, provide the first moment, or increase parameter R_sv_min in file \"OmegaMaxEnt_other_params.dat\".\n";
		return false;
	}
	
	mat M0b=zeros<mat>(NNfit,NNfit);
	mat M1b=zeros<mat>(NNfit,NNfit);
	mat M2b=zeros<mat>(NNfit,NNfit);
	mat M3b=zeros<mat>(NNfit,NNfit);
	
	for (Nfit=Nfitmin; Nfit<=Nfitmax; Nfit++)
	{
		pmax=Nfit-1;
		for (np=npmin; np<=pmax; np++)
		{
			X.zeros(Nfit,np+1);
			for (p=0; p<=np; p++)
				X.col(p)=pow(tau.rows(0,Nfit-1)/tau(Nfit-1),p);
			
			Gchi2tmp=Gtau.rows(0,Nfit-1)+flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
			CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)+fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))+flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
			invCG=inv(CG);
			//	invCG=inv_sympd(CG);
			AM=(X.t())*invCG*X;
			BM=(X.t())*invCG*Gchi2tmp;
			Mtmp=solve(AM,BM);
		//	if (!solve(Mtmp,AM,BM))
			M0b(Nfit-np-1,np-npmin)=-Mtmp(0);
			M2b(Nfit-np-1,np-npmin)=-2*Mtmp(2)/pow(tau(Nfit-1),2);
		//	M2b(Nfit-np-1,np-npmin)=-2*Mtmp(2);
		 
		 	Gchi2tmp=Gtau.rows(0,Nfit-1)-flipud(Gtau.rows(Ntau-Nfit+1,Ntau));
		 	CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)- fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))-flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
		 	invCG=inv(CG);
			//	invCG=inv_sympd(CG);
		 	AM=(X.t())*invCG*X;
			BM=(X.t())*invCG*Gchi2tmp;
			Mtmp=solve(AM,BM);
		//	if (!solve(Mtmp,AM,BM))
		//	M1b(Nfit-np-1,np-npmin)=Mtmp(1);
		//	M3b(Nfit-np-1,np-npmin)=6*Mtmp(3);
			M1b(Nfit-np-1,np-npmin)=Mtmp(1)/tau(Nfit-1);
			M3b(Nfit-np-1,np-npmin)=6*Mtmp(3)/pow(tau(Nfit-1),3);
		}
	}
	
	Nv=1;
	NvN=1;
	int jmin=NvN;
	int jmax=NNfit-NvN-2*Nv-1;
	int Nj=jmax-jmin+1;
	int lmin=Nv;
	int lmax=NNfit-2*NvN-Nv-1;
	int Nl=lmax-lmin+1;
	M0m.zeros(Nj,Nl);
	M1m.zeros(Nj,Nl);
	M2m.zeros(Nj,Nl);
	mat M3m=zeros<mat>(Nj,Nl);
	varM0.zeros(Nj,Nl);
	varM1.zeros(Nj,Nl);
	varM2.zeros(Nj,Nl);
	mat varM3=zeros<mat>(Nj,Nl);
	for (j=jmin; j<=jmax; j++)
	{
		for (l=lmin; l<=NNfit-1-j-Nv-NvN; l++)
		{
			M0m(j-jmin,l-lmin)=accu(M0b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M1m(j-jmin,l-lmin)=accu(M1b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M2m(j-jmin,l-lmin)=accu(M2b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			M3m(j-jmin,l-lmin)=accu(M3b.submat(j-NvN,l-Nv,j+NvN,l+Nv))/((2*Nv+1)*(2*NvN+1));
			varM0(j-jmin,l-lmin)=accu(pow(M0b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M0m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
			varM1(j-jmin,l-lmin)=accu(pow(M1b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M1m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
			varM2(j-jmin,l-lmin)=accu(pow(M2b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M2m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
			varM3(j-jmin,l-lmin)=accu(pow(M3b.submat(j-NvN,l-Nv,j+NvN,l+Nv)-M3m(j-jmin,l-lmin),2))/((2*Nv+1)*(2*NvN+1));
		}
	}
	
//	cout<<varM0<<endl;

	double varM0max=max(max(varM0));
	double varM1max=max(max(varM1));
	double varM2max=max(max(varM2));
	double varM3max=max(max(varM3));

	for (j=1; j<Nj; j++)
	{
		varM0.submat(j,Nl-j,j,Nl-1)=2*varM0max*ones<rowvec>(j);
		varM1.submat(j,Nl-j,j,Nl-1)=2*varM1max*ones<rowvec>(j);
		varM2.submat(j,Nl-j,j,Nl-1)=2*varM2max*ones<rowvec>(j);
		varM3.submat(j,Nl-j,j,Nl-1)=2*varM3max*ones<rowvec>(j);
	}
	
//	cout<<"M0b:\n"<<M0b<<endl;
//	cout<<"varM0:\n"<<varM0<<endl;
//	cout<<"M1b:\n"<<M1b<<endl;
//	cout<<"varM1:\n"<<varM1<<endl;
//	cout<<"M2b:\n"<<M2b<<endl;
//	cout<<"varM2:\n"<<varM2<<endl;
//	cout<<"M3b:\n"<<M3b<<endl;
//	cout<<"varM3:\n"<<varM3<<endl;

	varM0.min(jvmin,lvmin);
	double M0_N=M0m(jvmin,lvmin);
	varM1.min(jvmin,lvmin);
	double M1_N=M1m(jvmin,lvmin);
	varM2.min(jvmin,lvmin);
	double M2_N=M2m(jvmin,lvmin);
	varM3.min(jvmin,lvmin);
	double M3_N=M3m(jvmin,lvmin);
	
	cout<<"moments determined by polynomial fit to G(tau) at boundaries:\n";
	cout<<"norm: "<<M0_N<<endl;
	cout<<"first moment: "<<M1_N<<endl;
	cout<<"second moment: "<<M2_N<<endl;
	cout<<"third moment: "<<M3_N<<endl;
	

	if (displ_adv_prep_figs)
	{
		graph_2D g1, g2;
		char lgd_format[]="pfit=%d", lgd_entry[20];
		char xl[]="Nfit-p";
		
//		vec pfit=linspace<vec>(npmin,Nfitmax-1,NNfit);
		vec dNp=linspace<vec>(1,NNfit,NNfit);
		vec vtmp=M0b.col(0);
		g1.add_data(dNp.memptr(),vtmp.memptr(),NNfit);
		sprintf(lgd_entry,lgd_format,npmin);
		g1.add_to_legend(lgd_entry);
		for (j=1; j<NNfit; j++)
		{
			vtmp=M0b.col(j);
			g1.add_data(dNp.memptr(),vtmp.memptr(),NNfit);
			sprintf(lgd_entry,lgd_format,npmin+j);
			g1.add_to_legend(lgd_entry);
		}
		g1.add_title("$M_0$");
		g1.set_axes_labels(xl,NULL);
		g1.curve_plot();
		
		vtmp=M1b.col(0);
		g2.add_data(dNp.memptr(),vtmp.memptr(),NNfit);
		sprintf(lgd_entry,lgd_format,npmin);
		g2.add_to_legend(lgd_entry);
		for (j=1; j<NNfit; j++)
		{
			vtmp=M1b.col(j);
			g2.add_data(dNp.memptr(),vtmp.memptr(),NNfit);
			sprintf(lgd_entry,lgd_format,npmin+j);
			g2.add_to_legend(lgd_entry);
		}
		g2.add_title("$M_1$");
		g2.set_axes_labels(xl,NULL);
		g2.curve_plot();
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
		
	}

	varM1.min(jvmin,lvmin);
	np=lvmin+Nv+2;
	Nfit=jvmin+NvN+np+1;
	X=zeros<mat>(Nfit,np+1);
	for (p=0; p<=np; p++)
		X.col(p)=pow(tau.rows(0,Nfit-1),p);
	
	CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)+fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))+flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
//	CG(0,0)=CG(1,1);
	invCG=inv(CG);
	//	invCG=inv_sympd(CG);
	AM=(X.t())*invCG*X;
	mat invAMp=inv(AM);
	
	CG=Ctau_all.submat(0,0,Nfit-1,Nfit-1)-fliplr(Ctau_all.submat(0,Ntau-Nfit+1,Nfit-1,Ntau))-flipud(Ctau_all.submat(Ntau-Nfit+1,0,Ntau,Nfit-1))+flipud(fliplr(Ctau_all.submat(Ntau-Nfit+1,Ntau-Nfit+1,Ntau,Ntau)));
	invCG=inv(CG);
	//	invCG=inv_sympd(CG);
	AM=(X.t())*invCG*X;
	mat invAMn=inv(AM);
	
	double std_omega_tmp;
	double var_omega=M2_N/M0_N-pow(M1_N/M0_N,2);
	if (var_omega>0)
		std_omega_tmp=sqrt(var_omega);
	else
	{
		cout<<"Negative variance found during computation of moments.\n";
		return false;
	}
	
	covm_diag=true;
	if (!moments_provided)
	{
		M.zeros(4);
		M(0)=M0;
		M(1)=M1_N;
		M(2)=M2_N;
		M(3)=M3_N;
		M1=M(1);
		M2=M(2);
		M3=M(3);
		NM=4;
		M1_set=true;
		M2_set=true;
		errM.zeros(NM);
		errM(0)=errM0;
		errM(1)=sqrt(invAMn(1,1));
		errM(2)=sqrt(invAMp(2,2));
		errM(3)=sqrt(invAMn(3,3));
		errM1=errM(1);
		errM2=errM(2);
		errM3=errM(3);
		M_ord=linspace<vec>(0,NM-1,NM);
	}
	else
	{
		if (M0)
		{
			if (abs(M0_N-M0)/M0_N>tol_norm)
			{
				if (M0_in.size())
					cout<<"warning: norm of spectral function is different from provided one.\n";
				else
				{
					cout<<"warning: spectral function is not normalized.\n";
					cout<<"Use parameter \"norm of spectral function:\" in subsection DATA PARAMETERS to provide a norm different from 1.\n";
				}
			}
		}
		
		if (abs(M1-M1_N)/std_omega_tmp>tol_M1)
			cout<<"warning: first moment different from provided one\n";
		
		if (M2_in.size())
		{
			if (abs(M2-M2_N)/M2_N>tol_M2)
				cout<<"warning: second moment different from provided one\n";
			
			if (M3_in.size())
			{
				if (abs(M3-M3_N)/pow(std_omega_tmp,3)>tol_M3)
				{
					cout<<"warning: third moment different from provided one\n";
				}
			}
			else
			{
				M3=M3_N;
				M.zeros(4);
				M(0)=M0;
				M(1)=M1;
				M(2)=M2;
				M(3)=M3;
				errM3=sqrt(invAMn(3,3));
				NM=4;
				errM.zeros(NM);
				errM(0)=errM0;
				errM(1)=errM1;
				errM(2)=errM2;
				errM(3)=errM3;
				M_ord=linspace<vec>(0,NM-1,NM);
			}
		}
		else if (M3_in.size())
		{
			M2=M2_N;
			M2_set=true;
			M.zeros(4);
			M(0)=M0;
			M(1)=M1;
			M(2)=M2;
			M(3)=M3;
			errM2=sqrt(invAMp(2,2));
			NM=4;
			errM.zeros(NM);
			errM(0)=errM0;
			errM(1)=errM1;
			errM(2)=errM2;
			errM(3)=errM3;
			M_ord=linspace<vec>(0,NM-1,NM);
		}
		else
		{
			M2=M2_N;
			M2_set=true;
			M3=M3_N;
			errM2=sqrt(invAMp(2,2));
			errM3=sqrt(invAMn(3,3));
			M.zeros(4);
			M(0)=M0;
			M(1)=M1;
			M(2)=M2;
			M(3)=M3;
			NM=4;
			errM.zeros(NM);
			errM(0)=errM0;
			errM(1)=errM1;
			errM(2)=errM2;
			errM(3)=errM3;
			M_ord=linspace<vec>(0,NM-1,NM);
		}
	}
	COVM.zeros(NM,NM);
	COVM.diag()=square(errM);
	
	if (!std_omega)
	{
		var_omega=M2/M0-pow(M1/M0,2);
		std_omega=sqrt(var_omega);
	}
	
	if (!SC_set)
	{
		SC=M1/M0;
		SC_set=true;
	}
	if (!SW_set)
	{
		SW=f_SW_std_omega*std_omega;
		SW_set=true;
	}
	if (M(2)<0)
	{
		M=M.rows(0,1);
		COVM=COVM.submat(0,0,1,1);
		NM=2;
	}
	
	dG_dtau_computed=compute_dG_dtau();
	
	return true;
}

bool OmegaMaxEnt_data::set_covar_Gtau()
{
	cov_diag=false;
	error_provided=false;
	double sgn=-1;
	if (boson) sgn=1;
	
	if (error_file.size())
	{
		cout<<"error file provided\n";
		if (error_data.n_rows<Ntau)
		{
			cout<<"set_covar_Gtau() error: number of lines in error file is smaller than number of imaginary time steps\n";
			return false;
		}
		errGtau=error_data.col(col_errGtau-1);
	//	errGtau.resize(Ntau);
		Ctau.zeros(Ntau,Ntau);
		Ctau.diag()=square(errGtau.rows(0,Ntau-1));
		Ctau_all.zeros(Ntau+1,Ntau+1);
		Ctau_all.diag()=square(errGtau);
	//	Ctau_all.submat(0,0,Ntau-1,Ntau-1)=Ctau;
	//	Ctau_all(Ntau,Ntau)=pow(errGtau(0),2);
		
		error_provided=true;
	}
	else if (covar_tau_file.size())
	{
		if (Ctau.n_rows<Ntau)
		{
			cout<<"set_covar_Gtau() error: number of lines in covariance file is smaller than number of imaginary time steps\n";
			return false;
		}
		cout<<"imaginary time covariance matrix provided\n";
		Ctau=Ctau.submat(0,0,Ntau-1,Ntau-1);
		Ctau_all.zeros(Ntau+1,Ntau+1);
		Ctau_all.submat(0,0,Ntau-1,Ntau-1)=Ctau;
		errGtau.reset();
		error_provided=true;
		
		Ctau_all.submat(Ntau,0,Ntau,Ntau-1)=sgn*Ctau.row(0);
		Ctau_all.submat(0,Ntau,Ntau-1,Ntau)=sgn*Ctau.col(0);
		Ctau_all(Ntau,Ntau)=Ctau(0,0);
	}
	else if (N_params_noise)
	{
		cout<<"added noise relative error: "<<noise_params(ind_noise)<<endl;
		errGtau=noise_params(ind_noise)*abs(Gtau);
		
		int j,k;
		
		if (Nsmooth_errG)
		{
			double b=-log(wgt_min_sm)/Nsmooth_errG;
			
			vec wgt_sm(2*Nsmooth_errG+1);
			for (j=0; j<Nsmooth_errG; j++) wgt_sm(j)=exp(-b*(Nsmooth_errG-j));
			wgt_sm(Nsmooth_errG)=1;
			for (j=Nsmooth_errG+1; j<=2*Nsmooth_errG; j++) wgt_sm(j)=wgt_sm(2*Nsmooth_errG-j);
			wgt_sm=wgt_sm/sum(wgt_sm);
			
			
			//	cout<<"wgt_sm:\n";
			//	for (j=0; j<=2*Nsmooth_errG; j++) cout<<setw(20)<<wgt_sm(j);
			//	cout<<endl;
			//	cout<<"sum(wgt_sm): "<<sum(wgt_sm)<<endl;
			
			vec errG_tmp=errGtau;
			
		//	vec errG_tmp(Nn+2*Nsmooth_errG);
		//	errG_tmp.rows(Nsmooth_errG,Nn+Nsmooth_errG-1)=errGtau;
		//	errG_tmp.rows(0,Nsmooth_errG-1)=flipud(errGtau.rows(1,Nsmooth_errG));
		//	errG_tmp.rows(Nn+Nsmooth_errG,Nn+2*Nsmooth_errG-1)=flipud(errGtau.rows(Nn-Nsmooth_errG-1,Nn-2));
			
			double wgt_tmp;
			for (j=0; j<Nsmooth_errG; j++)
			{
				errGtau(j)=0;
				wgt_tmp=0;
				for (k=Nsmooth_errG-j; k<=2*Nsmooth_errG; k++)
				{
					errGtau(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGtau(j)=errGtau(j)/wgt_tmp;
			}
			for (j=Nsmooth_errG; j<Ntau-Nsmooth_errG; j++)
			{
				errGtau(j)=0;
				for (k=0; k<=2*Nsmooth_errG; k++)
				{
					errGtau(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
				}
			}
			for (j=Ntau-Nsmooth_errG; j<Ntau; j++)
			{
				errGtau(j)=0;
				wgt_tmp=0;
				for (k=0; k<=Nsmooth_errG+Ntau-1-j; k++)
				{
					errGtau(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGtau(j)=errGtau(j)/wgt_tmp;
			}
		}
		
		Ctau.zeros(Ntau,Ntau);
		Ctau.diag()=square(errGtau.rows(0,Ntau-1));
		Ctau_all.zeros(Ntau+1,Ntau+1);
		Ctau_all.diag()=square(errGtau);
//		Ctau_all.submat(0,0,Ntau-1,Ntau-1)=Ctau;
//		Ctau_all(Ntau,Ntau)=pow(errGtau(0),2);
	}
	else
	{
		cout<<"no errors provided\nusing a constant error\n";
		double maxGtau=max(abs(Gtau));
	 	errGtau=default_error_G*maxGtau*ones<vec>(Ntau+1);
		Ctau.zeros(Ntau,Ntau);
		Ctau.diag()=square(errGtau.rows(0,Ntau-1));
		Ctau_all.zeros(Ntau+1,Ntau+1);
		Ctau_all.diag()=square(errGtau);
//		Ctau_all.submat(0,0,Ntau-1,Ntau-1)=Ctau;
//		Ctau_all(Ntau,Ntau)=pow(errGtau(0),2);
	}
	
//	cout<<"cond(Ctau_all): "<<cond(Ctau_all)<<endl;
//	mat inv_tmp=inv(Ctau_all);
	
	COV=Ctau;
	
//	Ctau_all.submat(Ntau,0,Ntau,Ntau-1)=-Ctau.row(0);
//	Ctau_all.submat(0,Ntau,Ntau-1,Ntau)=-Ctau.col(0);
//	Ctau_all(Ntau,Ntau)=Ctau(0,0);
	
	COV=(COV+COV.t())/2.0;
	
	vec errGtau_tmp=sqrt(Ctau_all.diag());
	
	if (displ_prep_figs)
	{
		graph_2D g1;
		
		char ttl[]="error on $G(\\\\tau)$";
		char xl[]="$\\\\tau$";
		char attr1[]="'o-', color='b', markeredgecolor='b', markerfacecolor='b'";
		
		g1.add_title(ttl);
		plot(g1, tau, errGtau_tmp, xl, NULL, attr1);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	return true;
}

bool OmegaMaxEnt_data::fit_circle_arc(vec x, vec y, vec &arc_params)
{
	bool displ_fig=false;
	
	arc_params.set_size(3);
	arc_params(0)=INF;
	arc_params(1)=NAN;
	arc_params(2)=NAN;
	
	double tol_r=1e-10;
	int Niter_max=20;
	
	int N=x.n_rows;
	
	if (N<3)
	{
		cout<<"fit_circle_arc(): number of points must be larger of equal to 3\n";
		return false;
	}
	
	mat X(N,3);
	X.zeros();
	
	X.col(0)=pow(x-x(0),2);
	X.col(1)=x-x(0);
	X.col(2)=ones<vec>(N);
	
	mat A=X.t()*X;
	vec B=X.t()*y;
	
	vec cfs;
	
//	vec cfs=solve(A,B);
	
	if (!solve(cfs,A,B))
	{
		return true;
	}
	
	int jmid=floor(N/2);
	
	double Dx;
	double x0=x(jmid-1);
	Dx=x0-x(0);
	double y0=cfs(0)*pow(Dx,2)+cfs(1)*Dx+cfs(2);
	double x1=x(jmid);
	Dx=x1-x(0);
	double y1=cfs(0)*pow(Dx,2)+cfs(1)*Dx+cfs(2);
	double x2=x(jmid+1);
	Dx=x2-x(0);
	double y2=cfs(0)*pow(Dx,2)+cfs(1)*Dx+cfs(2);
	
	double denom=2*(x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1));
	
	if (denom==0)
	{
		arc_params(0)=INF;
		arc_params(1)=NAN;
		arc_params(2)=NAN;
		return true;
	}
	
	double xc_init=(pow(x2,2)*(y1-y0) + pow(x1,2)*(y0-y2) + pow(x0,2)*(y2-y1) + (y1-y0)*(y0-y2)*(y1-y2))/denom;
	double yc_init=-(pow(y2,2)*(x1-x0) + pow(y1,2)*(x0-x2) + pow(y0,2)*(x2-x1) + (x1-x0)*(x2-x0)*(x2-x1))/denom;
	
//	double r2_init=pow(x1-xc_init,2)+pow(y1-yc_init,2);
//	double r_init=sqrt(r2_init);
	
	double xc=xc_init;
	double yc=yc_init;
	
	vec rj=sqrt(pow(x-xc,2)+pow(y-yc,2));
	double r=sum(rj)/N;
	vec drj=r-rj;
	
	vec dr2=pow(drj,2);
	double chi2=sum(dr2);
	
	double xc_m=2*x1-xc;
	double yc_m=2*y1-yc;
	
	vec rjm=sqrt(pow(x-xc_m,2)+pow(y-yc_m,2));
	double rm=sum(rjm)/N;
	vec drjm=rm-rjm;
	dr2=pow(drjm,2);
	double chi2m=sum(dr2);
	
	if (chi2m<chi2)
	{
		xc=xc_m;
		yc=yc_m;
		rj=rjm;
		r=rm;
		drj=drjm;
		chi2=chi2m;
	}
	
	vec Drjxc=(xc-x)/rj;
	vec Drjyc=(yc-y)/rj;
	double Drxc=sum(Drjxc)/N;
	double Dryc=sum(Drjyc)/N;
	
	vec D2rjxc=1.0/rj-pow(x-xc,2)/pow(rj,3);
	vec D2rjyc=1.0/rj-pow(y-yc,2)/pow(rj,3);
	vec D2rjxcyc=-((x-xc) % (y-yc))/pow(rj,3);
	double D2rxc=sum(D2rjxc)/N;
	double D2ryc=sum(D2rjyc)/N;
	double D2rxcyc=sum(D2rjxcyc)/N;
	
	double Dchi2xc=sum(drj % (Drxc-Drjxc));
	double Dchi2yc=sum(drj % (Dryc-Drjyc));
	
	vec gradchi2(2);
	gradchi2(0)=-Dchi2xc;
	gradchi2(1)=-Dchi2yc;
	
	double D2chi2xc=sum( pow(Drxc-Drjxc,2) + drj % (D2rxc-D2rjxc) );
	double D2chi2xcyc=sum( (Drxc-Drjxc) % (Dryc-Drjyc) + drj % (D2rxcyc-D2rjxcyc) );
	double D2chi2yc=sum( pow(Dryc-Drjyc,2) + drj % (D2ryc-D2rjyc) );
	
	vec chi2_vec_arc(Niter_max), r_vec(Niter_max), xc_vec(Niter_max), yc_vec(Niter_max);
	chi2_vec_arc(0)=chi2;
	r_vec(0)=r;
	xc_vec(0)=xc;
	yc_vec(0)=yc;
	
	mat H(2,2);
	H(0,0)=D2chi2xc;
	H(0,1)=D2chi2xcyc;
	H(1,0)=D2chi2xcyc;
	H(1,1)=D2chi2yc;
	
	vec p;
	if (!solve(p,H,gradchi2))
	{
		return true;
//		cout<<"fit_circle_arc(): solve failed, using solve_LU.\n";
//		p=solve_LU(H,gradchi2);
	}
//	vec p=solve(H,gradchi2);
	
	double xc_prec=xc;
	double yc_prec=yc;
	double r_prec=r;
	
	xc=xc+p(0);
	yc=yc+p(1);
	
	rj=sqrt(pow(x-xc,2)+pow(y-yc,2));
	r=sum(rj)/N;
	drj=r-rj;
	
	double chi2_prec=chi2;
	dr2=pow(drj,2);
	chi2=sum(dr2);
	
	chi2_vec_arc(1)=chi2;
	r_vec(1)=r;
	xc_vec(1)=xc;
	yc_vec(1)=yc;
	
	int j=1;
	while (abs(r-r_prec)/r>tol_r && j<Niter_max-1 && chi2<chi2_prec)
	{
		Drjxc=(xc-x)/rj;
		Drjyc=(yc-y)/rj;
		Drxc=sum(Drjxc)/N;
		Dryc=sum(Drjyc)/N;
		
		D2rjxc=1.0/rj-pow(x-xc,2)/pow(rj,3);
		D2rjyc=1.0/rj-pow(y-yc,2)/pow(rj,3);
		D2rjxcyc=-((x-xc) % (y-yc))/pow(rj,3);
		D2rxc=sum(D2rjxc)/N;
		D2ryc=sum(D2rjyc)/N;
		D2rxcyc=sum(D2rjxcyc)/N;
		
		Dchi2xc=sum(drj % (Drxc-Drjxc));
		Dchi2yc=sum(drj % (Dryc-Drjyc));
		
		gradchi2(0)=-Dchi2xc;
		gradchi2(1)=-Dchi2yc;
		
		D2chi2xc=sum( pow(Drxc-Drjxc,2) + drj % (D2rxc-D2rjxc) );
		D2chi2xcyc=sum( (Drxc-Drjxc) % (Dryc-Drjyc) + drj % (D2rxcyc-D2rjxcyc) );
		D2chi2yc=sum( pow(Dryc-Drjyc,2) + drj % (D2ryc-D2rjyc) );
		
		H(0,0)=D2chi2xc;
		H(0,1)=D2chi2xcyc;
		H(1,0)=D2chi2xcyc;
		H(1,1)=D2chi2yc;
		
	//	p=solve(H,gradchi2);
		if (!solve(p,H,gradchi2))
		{
			j--;
			break;
		}
		
		xc_prec=xc;
		yc_prec=yc;
	 	r_prec=r;
	 
	 	xc=xc+p(0);
	 	yc=yc+p(1);
	 
		rj=sqrt(pow(x-xc,2)+pow(y-yc,2));
		r=sum(rj)/N;
		drj=r-rj;
		
		chi2_prec=chi2;
		dr2=pow(drj,2);
		chi2=sum(dr2);
	 
	 	xc_m=2*x1-xc;
	 	yc_m=2*y1-yc;
	 
	 	rjm=sqrt(pow(x-xc_m,2)+pow(y-yc_m,2));
	 	rm=sum(rjm)/N;
	 	drjm=rm-rjm;
	 
		dr2=pow(drjm,2);
	 	chi2m=sum(dr2);
	 
		if (chi2m<chi2)
		{
			xc=xc_m;
			yc=yc_m;
			rj=rjm;
			r=rm;
			drj=drjm;
			chi2=chi2m;
		}
	 
		j++;
		
		chi2_vec_arc(j)=chi2;
		r_vec(j)=r;
		xc_vec(j)=xc;
		yc_vec(j)=yc;
	}
	int Nvec=j+1;
	
	if (chi2>chi2_prec)
	{
		xc=xc_prec;
		yc=yc_prec;
		r=r_prec;
	}
	
	double r2=pow(r,2);
	vec ypl=zeros<vec>(N);
	vec dr2xc2=r2-pow(x-xc,2);
	vec xpl=x;
	double sgn=0;
	for (j=0; j<N; j++)
	{
		if (dr2xc2(j)<0)
		{
		 	if (x(j)>xc)
				xpl(j)=xc+r;
		 	else
				xpl(j)=xc-r;
		}
		if (y(j)<yc)
		{
			ypl(j)=yc-sqrt(r2-pow(xpl(j)-xc,2));
			sgn=sgn+1;
		}
		else
		{
			ypl(j)=yc+sqrt(r2-pow(xpl(j)-xc,2));
			if (y(j)>yc)
				sgn=sgn-1;
		}
	}
	
	if (sgn) r=sgn*r/abs(sgn);
	
	arc_params(0)=r;
	arc_params(1)=xc;
	arc_params(2)=yc;
	
	if (displ_fig)
	{
		graph_2D g1, g2, g3, g4, g5;
		
		char at1[]="'o'", at2[]="'s'";
		g1.add_data(x.memptr(),y.memptr(),N);
		g1.add_attribute(at1);
		g1.add_data(xpl.memptr(),ypl.memptr(),N);
		g1.add_attribute(at2);
		g1.curve_plot();
		
		char xl[]="iteration";
		char yl[]="chi2_arc";
		char yl1[]="r";
		char yl2[]="x_c";
		char yl3[]="y_c";
		char attr[]="'o', color='b', markerfacecolor='none'";
		char attr1[]="'o', color='r', markerfacecolor='none'";
		char attr2[]="'o', color='m', markerfacecolor='none'";
		char attr3[]="'o', color='k', markerfacecolor='none'";
		vec iter=linspace<vec>(0,Nvec-1,Nvec);
		
		plot(g2, iter, chi2_vec_arc, xl, yl, attr);
		plot(g3, iter, r_vec, xl, yl1, attr1);
		plot(g4, iter, xc_vec, xl, yl2, attr2);
		plot(g5, iter, yc_vec, xl, yl3, attr3);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	return true;
}

void OmegaMaxEnt_data::compute_Bryan_spectrum(vec &Abr)
{
	int i,j;
	
	mat PA(ind_alpha_vec,Nw);
	
	char file_name[200];
	mat data;
	vec A_tmp;

	for (i=ind_alpha_vec-1; i>=0; i--)
	{
		sprintf(file_name, output_name_format.c_str(),tem,alpha_vec(i));
		if (data.load(file_name))
		{
			A_tmp=data.col(1);
			PA.row(ind_alpha_vec-i-1)=P_alpha_G(i)*A_tmp.t();
		}
		else
			cout<<"spectral function was not saved at alpha= "<<alpha_vec(i)<<endl;
	}
	
	int imin, imax=Nw-2;
	if (col_Gi>0)
		imin=1;
	else
		imin=0;
	
	vec lalpha=flipud(log(alpha_vec.rows(0,ind_alpha_vec-1)));
	vec coeffs(4*(ind_alpha_vec-1));
	vec PA_alpha;
	
	Abr.zeros(Nw);
	for (i=imin; i<=imax; i++)
	{
		coeffs(0)=0;
		coeffs(1)=0;
		PA_alpha=PA.col(i);
		spline_coeffs(lalpha.memptr(), PA_alpha.memptr(), ind_alpha_vec, coeffs.memptr());
		Abr(i)=integrate_spline(lalpha, coeffs);
	}
	
}

void OmegaMaxEnt_data::normalize_P_alpha_G()
{
	double log_P_alpha_G_max=log_P_alpha_G.rows(0,ind_alpha_vec-1).max();
	
//	uword ind_P_alpha_G_max;
//	P_alpha_G.max(ind_P_alpha_G_max);
	
	vec lalpha=flipud(log(alpha_vec.rows(0,ind_alpha_vec-1)));
	P_alpha_G=flipud(exp(log_P_alpha_G.rows(0,ind_alpha_vec-1)-log_P_alpha_G_max));
	vec coeffs(4*(ind_alpha_vec-1));
	coeffs(0)=0;
	coeffs(1)=0;
	
	spline_coeffs(lalpha.memptr(), P_alpha_G.memptr(), ind_alpha_vec, coeffs.memptr());

	double sum=integrate_spline(lalpha, coeffs);
	
//	double sum2=oned_spline_integral(coeffs.memptr(),lalpha.memptr(),P_alpha_G.memptr(), ind_alpha_vec);
	
//	cout<<"integral P_alpha_G: "<<sum<<endl;
//	cout<<"avec oned_spline_integral: "<<sum2<<endl;
//	cout<<"difference relative: "<<(sum-sum2)/sum<<endl;
	
	P_alpha_G=flipud(P_alpha_G)/sum;
}

double OmegaMaxEnt_data::integrate_spline(vec x, vec coeffs)
{
	int N=x.n_rows;
	double dx;
	double sum=0;
	for (int i=0; i<N-1; i++)
	{
		dx=x(i+1)-x(i);
		sum+=coeffs(4*i)*dx*dx*dx*dx/4+coeffs(4*i+1)*dx*dx*dx/3+coeffs(4*i+2)*dx*dx/2+coeffs(4*i+3)*dx;
	}
	
	return sum;
}

double OmegaMaxEnt_data::integrate_P_A_i_alpha(double alphaD_p)
{
	int i, j;
	
	fctPtr1 Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::P_A_alpha_val);
	fctPtr PtrS=static_cast<fctPtr> (&OmegaMaxEnt_data::S_i);
	
	double *alpha_D=new double[1];
	void *par[1];
	par[0]=alpha_D;
	
	double lims[2];
	
	double umax, umax_max, umax_min;
	double umin=1e-100;
	
	double argmax=30;
	double argmax2=40;
	double arglogmax=2*exp(1);
	double arg;
	
	double tol0=1e-10;
	double tol1=1e-4;
	double tol;
	
	double sq2=sqrt(2);
	
	vec lims_val={2,4,8,16};
	int Nvals=lims_val.n_rows;
	vec u_lims(2*Nvals+1);
	double uv[1];
	double parS[1];
	double uv_init[3];
	
	double IPA=0;
	double IPA_gaussian;
	
	int nbEval[1];
	
	alpha_D[0]=alphaD_p;
	
	int j0, jfin;
	
	if (alpha_D[0]<argmax/arglogmax) umax=sqrt(argmax/alpha_D[0]);
	else umax=sqrt(arglogmax);
	
	parS[0]=0;
	while (alpha_D[0]*S_i(umax, parS)<argmax)
	{
		umax=2*umax;
	}
	umax_min=sq2;
	while (alpha_D[0]*S_i(umax, parS)>argmax2)
	{
		umax_max=umax;
		umax=umax_min+(umax-umax_min)/2;
		while (alpha_D[0]*S_i(umax, parS)<argmax)
		{
			umax_min=umax;
			umax=umax+(umax_max-umax)/2;
		}
	}
	
	lims[0]=umin;
	lims[1]=sq2;
	uv_init[0]=0.5*(lims[0]+lims[1])/2;
	uv_init[1]=lims[0];
	uv_init[2]=lims[1];
	for (j=Nvals-1; j>=0; j--)
	{
		parS[0]=lims_val(j)/alpha_D[0];
		tol=tol1*parS[0];
		if (parS[0]<1 && find_zero(PtrS, uv_init, parS, uv, lims, tol))
		{
			u_lims(Nvals-j-1)=uv[0];
			lims[0]=uv[0];
			//				cout<<setw(20)<<uv[0]<<S_i(uv[0], parS)<<endl;
		}
		else
		{
			u_lims(Nvals-j-1)=-1;
		}
		uv_init[0]=0.5*(lims[0]+lims[1])/2;
		uv_init[1]=lims[0];
		uv_init[2]=lims[1];
	}
	u_lims(Nvals)=sq2;
	lims[0]=sq2;
	lims[1]=umax;
	uv_init[0]=0.5*(lims[0]+lims[1])/2;
	uv_init[1]=lims[0];
	uv_init[2]=lims[1];
	for (j=0; j<Nvals; j++)
	{
		parS[0]=lims_val(j)/alpha_D[0];
		if (find_zero(PtrS, uv_init, parS, uv, lims, tol))
		{
			u_lims(Nvals+j+1)=uv[0];
			lims[0]=uv[0];
			//				cout<<setw(20)<<uv[0]<<S_i(uv[0], parS)<<endl;
		}
		else
		{
			u_lims(Nvals+j+1)=-1;
		}
		uv_init[0]=0.5*(lims[0]+lims[1])/2;
		uv_init[1]=lims[0];
		uv_init[2]=lims[1];
	}
	
	if (u_lims(Nvals+1)>0)
	{
		tol=tol0;
		lims[0]=u_lims(Nvals);
		lims[1]=u_lims(Nvals+1);
		nbEval[0]=0;
		IPA=quadInteg1D(Ptr, lims, tol, nbEval, par);
		tol=tol0*IPA;
		j0=0;
		while (j0<Nvals && u_lims(j0)<0) j0++;
		lims[0]=umin;
		lims[1]=u_lims(j0);
		nbEval[0]=0;
		IPA=IPA+quadInteg1D(Ptr, lims, tol, nbEval, par);
		for (j=j0; j<Nvals; j++)
		{
			lims[0]=u_lims(j);
			lims[1]=u_lims(j+1);
			nbEval[0]=0;
			IPA=IPA+quadInteg1D(Ptr, lims, tol, nbEval, par);
		}
		jfin=2*Nvals;
		while (jfin>Nvals+1 && u_lims(jfin)<0) jfin--;
		lims[0]=u_lims(jfin);
		lims[1]=umax;
		nbEval[0]=0;
		IPA=IPA+quadInteg1D(Ptr, lims, tol, nbEval, par);
		for (j=jfin; j>Nvals+1; j--)
		{
			lims[0]=u_lims(j-1);
			lims[1]=u_lims(j);
			nbEval[0]=0;
			IPA=IPA+quadInteg1D(Ptr, lims, tol, nbEval, par);
		}
	}
	else
	{
		tol=tol0;
		lims[0]=umin;
		lims[1]=u_lims(Nvals);
		nbEval[0]=0;
		IPA=IPA+quadInteg1D(Ptr, lims, tol, nbEval, par);
		lims[0]=u_lims(Nvals);
		lims[1]=umax;
		nbEval[0]=0;
		IPA=IPA+quadInteg1D(Ptr, lims, tol, nbEval, par);
	}
	IPA_gaussian=sqrt(PI/alpha_D[0]);
//	cout<<setw(15)<<alpha_D[0]<<setw(15)<<IPA<<IPA_gaussian<<endl;
	
	delete [] alpha_D;
	
	return IPA;
}

void OmegaMaxEnt_data::integrate_P_A_alpha()
{
	int i, j;
	bool trace_integ=false;
	
	fctPtr1 Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::P_A_alpha_val);
	fctPtr PtrS=static_cast<fctPtr> (&OmegaMaxEnt_data::S_i);
	
	double pow_alphaD_max=5;
	double pow_alphaD_min=-20;
	double delta_pow_alphaD=0.1;
	double pow_alphaD=pow_alphaD_max;
	
	int NalphaD=(pow_alphaD_max-pow_alphaD_min)/delta_pow_alphaD;
	
	pow_alphaD_vec.zeros(NalphaD);
	integ_P_A_alpha.zeros(NalphaD);
	vec integ_P_A_alpha_gaussian(NalphaD);
	
	double *alpha_D=new double[1];
	void *par[1];
	par[0]=alpha_D;
	
	double lims[2];
	
	double umax, umax_max, umax_min;
	double umin=1e-100;
	
	double argmax=30;
	double argmax2=40;
	double arglogmax=2*exp(1);
	double arg;
	
	double tol0=1e-10;
	double tol1=1e-4;
	double tol;
	
	double sq2=sqrt(2);
	
	vec lims_val={2,4,8,16};
	int Nvals=lims_val.n_rows;
	vec u_lims(2*Nvals+1);
	double uv[1];
	double parS[1];
	double uv_init[3];
	
	int nbEval[1];
	
	alpha_D[0]=pow(10,pow_alphaD);
	pow_alphaD_vec(0)=pow_alphaD;
	
	int j0, jfin;
	
	for (i=0; i<NalphaD; i++)
	{
		if (alpha_D[0]<argmax/arglogmax) umax=sqrt(argmax/alpha_D[0]);
		else umax=sqrt(arglogmax);
		
		parS[0]=0;
		while (alpha_D[0]*S_i(umax, parS)<argmax)
		{
			umax=2*umax;
		}
		umax_min=sq2;
		while (alpha_D[0]*S_i(umax, parS)>argmax2)
		{
			umax_max=umax;
			umax=umax_min+(umax-umax_min)/2;
			while (alpha_D[0]*S_i(umax, parS)<argmax)
			{
				umax_min=umax;
				umax=umax+(umax_max-umax)/2;
			}
		}
		

//		cout<<"umax: "<<umax<<endl;
//		parS[0]=0;
//		cout<<"alpha_D[0]*S_i(umax): "<<alpha_D[0]*S_i(umax, parS)<<endl;
//		cout<<"exp(-alpha_D*S_i(umax)): "<<exp(-alpha_D[0]*S_i(umax, parS))<<endl;

		lims[0]=umin;
		lims[1]=sq2;
		uv_init[0]=0.5*(lims[0]+lims[1])/2;
		uv_init[1]=lims[0];
		uv_init[2]=lims[1];
		for (j=Nvals-1; j>=0; j--)
		{
			parS[0]=lims_val(j)/alpha_D[0];
			tol=tol1*parS[0];
			if (parS[0]<1 && find_zero(PtrS, uv_init, parS, uv, lims, tol))
			{
				u_lims(Nvals-j-1)=uv[0];
				lims[0]=uv[0];
//				cout<<setw(20)<<uv[0]<<S_i(uv[0], parS)<<endl;
			}
			else
			{
				u_lims(Nvals-j-1)=-1;
			}
			uv_init[0]=0.5*(lims[0]+lims[1])/2;
			uv_init[1]=lims[0];
			uv_init[2]=lims[1];
		}
		u_lims(Nvals)=sq2;
		lims[0]=sq2;
		lims[1]=umax;
		uv_init[0]=0.5*(lims[0]+lims[1])/2;
		uv_init[1]=lims[0];
		uv_init[2]=lims[1];
		for (j=0; j<Nvals; j++)
		{
			parS[0]=lims_val(j)/alpha_D[0];
			if (find_zero(PtrS, uv_init, parS, uv, lims, tol))
			{
				u_lims(Nvals+j+1)=uv[0];
				lims[0]=uv[0];
//				cout<<setw(20)<<uv[0]<<S_i(uv[0], parS)<<endl;
			}
			else
			{
				u_lims(Nvals+j+1)=-1;
			}
			uv_init[0]=0.5*(lims[0]+lims[1])/2;
			uv_init[1]=lims[0];
			uv_init[2]=lims[1];
		}
/*
		parS[0]=0;
		for (j=0; j<=2*Nvals; j++)
		{
			if (u_lims(j)>0) cout<<setw(20)<<alpha_D[0]*S_i(u_lims(j), parS);
			else
				cout<<setw(20)<<-1;
		}
		cout<<endl;
*/
		if (u_lims(Nvals+1)>0)
		{
			tol=tol0;
			lims[0]=u_lims(Nvals);
			lims[1]=u_lims(Nvals+1);
			nbEval[0]=0;
			integ_P_A_alpha[i]=quadInteg1D(Ptr, lims, tol, nbEval, par);
			tol=tol0*integ_P_A_alpha[i];
			j0=0;
			while (j0<Nvals && u_lims(j0)<0) j0++;
			lims[0]=umin;
			lims[1]=u_lims(j0);
			nbEval[0]=0;
			integ_P_A_alpha[i]=integ_P_A_alpha[i]+quadInteg1D(Ptr, lims, tol, nbEval, par);
			for (j=j0; j<Nvals; j++)
			{
				lims[0]=u_lims(j);
				lims[1]=u_lims(j+1);
				nbEval[0]=0;
				integ_P_A_alpha[i]=integ_P_A_alpha[i]+quadInteg1D(Ptr, lims, tol, nbEval, par);
			}
			jfin=2*Nvals;
			while (jfin>Nvals+1 && u_lims(jfin)<0) jfin--;
			lims[0]=u_lims(jfin);
			lims[1]=umax;
			nbEval[0]=0;
			integ_P_A_alpha[i]=integ_P_A_alpha[i]+quadInteg1D(Ptr, lims, tol, nbEval, par);
			for (j=jfin; j>Nvals+1; j--)
			{
				lims[0]=u_lims(j-1);
				lims[1]=u_lims(j);
				nbEval[0]=0;
				integ_P_A_alpha[i]=integ_P_A_alpha[i]+quadInteg1D(Ptr, lims, tol, nbEval, par);
			}
		}
		else
		{
			tol=tol0;
			lims[0]=umin;
			lims[1]=u_lims(Nvals);
			nbEval[0]=0;
			integ_P_A_alpha[i]=integ_P_A_alpha[i]+quadInteg1D(Ptr, lims, tol, nbEval, par);
			lims[0]=u_lims(Nvals);
			lims[1]=umax;
			nbEval[0]=0;
			integ_P_A_alpha[i]=integ_P_A_alpha[i]+quadInteg1D(Ptr, lims, tol, nbEval, par);
		}
		integ_P_A_alpha_gaussian(i)=sqrt(PI/alpha_D[0]);
//		cout<<setw(15)<<alpha_D[0]<<setw(15)<<integ_P_A_alpha[i]<<setw(15)<<integrate_P_A_i_alpha(alpha_D[0])<<integ_P_A_alpha_gaussian(i)<<endl;

		pow_alphaD=pow_alphaD-delta_pow_alphaD;
		alpha_D[0]=pow(10,pow_alphaD);
		pow_alphaD_vec(i)=pow_alphaD;
	}
	
	if (trace_integ)
	{
		vec l_I_P_A_alpha=log10(integ_P_A_alpha);
		vec l_I_P_A_alpha_gaussian=log10(integ_P_A_alpha_gaussian);
		
		graph_2D g1;
		char xl[]="$\\\\log_{10}(\\\\alpha D_i)$";
		char yl[]="$\\\\log_{10}(I)$";
		char lgd_ex[]="exact";
		char lgd_gauss[]="gaussian";
		double xlims[]={pow_alphaD_min-0.01*(pow_alphaD_max-pow_alphaD_min),pow_alphaD_max+0.01*(pow_alphaD_max-pow_alphaD_min)};
		//	g1.add_data(pow_alphaD_vec.memptr(),integ_P_A_alpha.memptr(),NalphaD);
		g1.add_data(pow_alphaD_vec.memptr(),l_I_P_A_alpha.memptr(),NalphaD);
		g1.add_to_legend(lgd_ex);
		//	g1.add_data(pow_alphaD_vec.memptr(),integ_P_A_alpha_gaussian.memptr(),NalphaD);
		g1.add_data(pow_alphaD_vec.memptr(),l_I_P_A_alpha_gaussian.memptr(),NalphaD);
		g1.add_to_legend(lgd_gauss);
		g1.set_axes_labels(xl,yl);
		g1.set_axes_lims(xlims,NULL);
		g1.curve_plot();
		graph_2D::show_figures();
	}
}

double OmegaMaxEnt_data::P_A_alpha_val(double u, void*par[])
{
	double *alpha_D=reinterpret_cast<double*>(par[0]);
	
	if (u)
		return exp(-alpha_D[0]*((u*u/2)*log(u*u/2)-u*u/2+1));
	else
		return exp(-alpha_D[0]);
}

double OmegaMaxEnt_data::S_i(double u, double c[])
{
	return (u*u/2)*log(u*u/2)-u*u/2+1-c[0];
}

void OmegaMaxEnt_data::minimize()
{
	double tol_int_dA2=1e-2;
	char alpha_output[100], file_name[200];
	char alpha_output_format[]="%d \t alpha: % 1.4e,  Q: % 1.4e,  S: % 1.4e,  chi2: % 1.4e\n";
	double mean_int_dA_prec, mean_int_dA_prec2, A1min, chi2prec, Q, S;
	mat chi2, P, KGMj, U, V, mean_int_dA, M_save;
	vec A1, A1_prec, AS, c1, c2, grS, B, B2, Pd, sK, sK2(NwA), D1, dA1, dA1_prec, VPdA, dA_rel, DG;
	vec G_out, G_V_out, errIm, errRe, M_out, M_V_out, eigv_ind;
	uvec ind_c2_sat, ind_An, ind_Anul;
	int i, ind_alpha, iter_dA;
	
	mat Pw, zAz, SD;
	vec c1w, c2w, grSw, Awt, Pdw, Lw, ASw, IPA(NwA), Dw;
	double alphaD_max_IPA=1e5;
	
	int Nalpha_min=200;
	
	bool alpha_too_small=false;
	
	double pow_alpha0=log10(alpha0);
	double pow_alpha=log10(alpha);
	double alpha_prec=pow(10,pow_alpha+pow_alpha_step);
	
	double realmin=1e-100;
	
	double rADmin=realmin;
	vec Amin=rADmin*default_model;
	
	double rADchange=realmin;
	vec Achange=rADchange*default_model;
	vec Achange_w=rADchange*default_model%dwS/(2*PI);
	
	/*
	if (!svd(U,sK,V,KG_V,"std"))
	{
		cout<<"minimize(): svd error\n";
		return;
	}
	*/
	
	if (!svd(U,sK,V,KG_V))
	{
		if (!svd(U,sK,V,KG_V,"std"))
		{
			cout<<"minimize(): svd error\n";
			return;
		}
	}
	sK2.zeros();
	sK2.rows(0,sK.n_rows-1)=pow(sK,2);
	double alpha_c2_max=sK2.max()*rc2H;
		
	cout<<"Computing spectrum as a function of alpha...\n";
	
	vec lalpha, lchi2, arc_params;
	int j, jmin, jmax;
	double fs=f_scale_lalpha_lchi2;
	double smooth_length=chi2_alpha_smooth_range;
	double la, lc, ls;
	double xc, yc, x1, y1;
	
	c1=(log(rADchange)+1)*dwS;
	vec c2_alpha=2*PI*alpha_c2_max*ones<vec>(NwA);
	
	c1w=(log(rADchange)+1)*ones<vec>(NwA);
	
	uword ind_max_dchi2_alpha;
	
	ind_alpha=1;
	DG=GM-KGM*A;
	chi2=DG.t()*DG;
	
	SD=(exp(-1)*default_model.t()*dwS)/(2*PI);
	
	Dw=(exp(-1)*default_model%dwS)/(2*PI);
	
//	int iter_dA2;
//	vec DGM, Atmp, mod2_B, mod2_B2;
	
	while (ind_alpha<=Nalpha && alpha>=alpha_min)
	{
		A1=A;
		
		c2=c2_alpha/alpha;
		
		grS=dwS % log(A1/default_model) + dwS;
		ind_An=find(A1/default_model<rADchange);
		if (ind_An.n_rows)
		{
			grS.rows(ind_An)=2*c2.rows(ind_An) % (A1.rows(ind_An)-Achange.rows(ind_An))+c1.rows(ind_An);
		}
		
//		DGM=GM-KGM*A1;
//		S=(A1%dwS)%log(A1/default_model);
//		Q=DGM.t()*DGM-(alpha/(4*PI))*S;
		
		B=KGM.t()*(GM-KGM*A1)-(alpha/(4*PI))*grS;
		
		Pd=sqrt(4*PI*A1/(alpha*dwS));
		if (ind_An.n_rows)
			Pd.rows(ind_An)=sqrt(2*PI/(alpha*c2(ind_An)));
		P=diagmat(Pd);
		
		KGMj=KGM*P;
		
		if (!svd(U,sK,V,KGMj))
		{
			if (!svd(U,sK,V,KGMj,"std"))
			{
				cout<<"minimize(): svd error\n";
				return;
			}
		}
	/*
		if (!svd(U,sK,V,KGMj,"std"))
		{
			cout<<"minimize(): svd error\n";
			return;
		}
	 */
		
		B2=V.t()*(P*B);
		sK2.zeros();
		sK2.rows(0,sK.n_rows-1)=pow(sK,2);
		D1=sK2+1;
		VPdA=B2/D1;
		dA1=P*(V*VPdA);
		
		mean_int_dA=abs(dA1.t())*dwS;
		mean_int_dA_prec=2*mean_int_dA(0);
		mean_int_dA_prec2=mean_int_dA_prec;
		
		iter_dA=1;
		
		A1_prec=A1;
		dA1_prec=dA1;
		
		A1=A1+dA1;
		
		ind_Anul=find(A1==0);
		if (ind_Anul.n_rows)
		{
			A1.rows(ind_Anul)=Amin.rows(ind_Anul);
		}
		
		A1min=min(A1-Amin);

		while ( iter_dA<Niter_dA_max && (mean_int_dA(0)>tol_int_dA || A1min<0) && mean_int_dA(0)<mean_int_dA_prec) // (mean_int_dA(0)<mean_int_dA_prec || mean_int_dA(0)<mean_int_dA_prec2))
		{
			grS=dwS % log(A1/default_model) + dwS;
			ind_An=find(A1/default_model<rADchange);
			if (ind_An.n_rows)
			{
				grS.rows(ind_An)=2*c2.rows(ind_An) % (A1.rows(ind_An)-Achange.rows(ind_An))+c1.rows(ind_An);
			}
			
			B=KGM.t()*(GM-KGM*A1)-(alpha/(4*PI))*grS;
			
			Pd=sqrt(4*PI*A1/(alpha*dwS));
			if (ind_An.n_rows)
				Pd.rows(ind_An)=sqrt(2*PI/(alpha*c2(ind_An)));
			P=diagmat(Pd);
			
			KGMj=KGM*P;
			
			/*
			if (!svd(U,sK,V,KGMj,"std"))
			{
				cout<<"minimize(): svd error\n";
				return;
			}
			*/
			
			if (!svd(U,sK,V,KGMj))
			{
				if (!svd(U,sK,V,KGMj,"std"))
				{
					cout<<"minimize(): svd error\n";
					return;
				}
			}
			
			B2=V.t()*(P*B);
			sK2.zeros();
			sK2.rows(0,sK.n_rows-1)=pow(sK,2);
			D1=sK2+1;
			VPdA=B2/D1;
			dA1=P*(V*VPdA);
	 
		/*
			iter_dA2=0;
			while (iter_dA2<Niter_dA_max && mean_int_dA(0)>mean_int_dA_prec)
			{
				dA1=dA1/2;
				mean_int_dA=abs(dA1.t())*dwS;
				iter_dA2++;
			}
		
			Atmp=A1+dA1;
			grS=dwS % log(Atmp/default_model) + dwS;
			ind_An=find(Atmp/default_model<rADchange);
			if (ind_An.n_rows)
			{
				grS.rows(ind_An)=2*c2.rows(ind_An) % (Atmp.rows(ind_An)-Achange.rows(ind_An))+c1.rows(ind_An);
			}
			B2=KGM.t()*(GM-KGM*Atmp)-(alpha/(4*PI))*grS;
			mod2_B2=B2.t()*B2;
			if (mod2_B2<mod2_B)
			{
				A1=Atmp;
				A1min=min(A1-Amin);
				dA1_prec=dA1;
			}
		*/
		
			mean_int_dA_prec2=mean_int_dA_prec;
			mean_int_dA_prec=mean_int_dA(0);
			mean_int_dA=abs(dA1.t())*dwS;
			
			if (mean_int_dA(0)<mean_int_dA_prec)
			{
				A1_prec=A1;
				dA1_prec=dA1;
				A1=A1+dA1;
				A1min=min(A1-Amin);
			}
			else
			{
				mean_int_dA(0)=mean_int_dA_prec;
			}
		/*
			else if (mean_int_dA(0)<mean_int_dA_prec2)
			{
				A1=A1_prec+dA1_prec/2;
				dA1_prec=dA1_prec/2;
				A1min=min(A1-Amin);
				mean_int_dA=abs(dA1_prec.t())*dwS;
			}
		*/
			
			ind_Anul=find(A1==0);
			if (ind_Anul.n_rows)
			{
				A1.rows(ind_Anul)=Amin.rows(ind_Anul);
			//	A1.rows(ind_Anul)=arma::max(Amin.rows(ind_Anul),DBL_MIN);
			}
			
			iter_dA++;
		}

		if (mean_int_dA(0)>tol_int_dA2)
		{
			cout<<"Integrated absolute variation in A is too large. Stopping minimization.\n";
			ind_alpha=Nalpha+1;
		}
		
/*
		if (mean_int_dA(0)>mean_int_dA_prec || iter_dA==Niter_dA_max)
		{
			cout<<"alpha= "<<alpha<<"  iteration: "<<iter_dA<<"  integrated absolute variation in A: "<<setw(20)<<mean_int_dA_prec2<<setw(20)<<mean_int_dA_prec<<mean_int_dA(0)<<endl;
		}
*/
		chi2prec=chi2(0);
		DG=GM-KGM*A1;
		chi2=DG.t()*DG;
		AS=A1;
		ind_An=find(AS/default_model<rADchange);
		if (ind_An.n_rows)
			AS.rows(ind_An)=Amin.rows(ind_An);
		S=-sum(AS % dwS % log(AS/default_model))/(2*PI);
		
		if (chi2(0)/chi2prec<R_chi2_min && alpha<alpha_prec && pow_alpha_step>=2*pow_alpha_step_min)
			pow_alpha_step=pow_alpha_step/2;
		
		if ((chi2(0)<chi2prec && alpha<=alpha_prec) || (chi2(0)>chi2prec && alpha>alpha_prec) || pow_alpha==pow_alpha0)
		{
			Aprec.cols(0,NAprec-2)=Aprec.cols(1,NAprec-1);
			Aprec.col(NAprec-1)=A;
			A=A1;
			alpha_vec(ind_alpha_vec)=alpha;
			chi2_vec(ind_alpha_vec)=chi2(0);
			S_vec(ind_alpha_vec)=S;
			vec logA=log(A/default_model);
			ind_An=find(A/default_model<rADchange);
			if (ind_An.n_rows)
			{
				logA.rows(ind_An)=log(rADmin)*ones<vec>(ind_An.n_rows);
			}
			S_vec(ind_alpha_vec)=-sum(A % dwS % logA)/(2*PI);
			Aw_samp.row(ind_alpha_vec)=trans(A(w_sample_ind));
			
			if (print_alpha)
			{
				Q=chi2(0)-alpha*S_vec(ind_alpha_vec);
				sprintf(alpha_output,alpha_output_format,ind_alpha,alpha,Q,S_vec(ind_alpha_vec),chi2(0));
				cout<<alpha_output;
			}
			
			if (compute_P_alpha_G)
			{
				// compute P_alpha_G in classic and Bryan's method
				Awt=(A%dwS)/(2*PI);
				ASw=(AS%dwS)/(2*PI);
				
				c2w=pow(2*PI,2)*c2_alpha/(alpha*(dwS%dwS));
				grSw=log(A/default_model) + ones<vec>(NwA);
				if (ind_An.n_rows)
					grSw.rows(ind_An)=2*c2w.rows(ind_An)%(Awt.rows(ind_An)-Achange_w.rows(ind_An))+c1w.rows(ind_An);
				
				zAz=grSw.t()*diagmat(ASw)*grSw;
				
				Pd=sqrt(4*PI*A/(alpha*dwS));
				if (ind_An.n_rows)
					Pd.rows(ind_An)=sqrt(2*PI/(alpha*c2(ind_An)));
				P=diagmat(Pd);
				
				KGMj=KGM*P;
 
//				Pdw=sqrt(2*Awt/alpha);
//				if (ind_An.n_rows)
//					Pdw.rows(ind_An)=sqrt(1/(alpha*c2w.rows(ind_An)));
//				Pw=diagmat(Pdw);
//				KGMj=KGMw*Pw;
				
				if (!svd(U,sK,V,KGMj,"std"))
				{
					cout<<"minimize(): svd error\n";
					return;
				/*
					if (!svd(U,sK,V,KGMj,"std"))
					{
						cout<<"minimize(): svd error\n";
						return;
					}
				 */
				}
				
				sK2.zeros();
				sK2.rows(0,sK.n_rows-1)=pow(sK,2);
				Lw=sK2+1;
 
//				for (i=0; i<NwA; i++)
//				{
//					if (alpha*Dw(i)/2>alphaD_max_IPA) IPA(i)=sqrt(2*PI/alpha);
//					else IPA(i)=sqrt(Dw(i))*integrate_P_A_i_alpha(alpha*Dw(i)/2);
//				}
				
				log_P_alpha_G(ind_alpha_vec)=-chi2(0)/2+(alpha/2)*(S-SD(0))-sum(log(Lw))/2;
//				log_P_alpha_G(ind_alpha_vec)=-chi2(0)/2-sum(log(Lw))/2;
//				log_P_alpha_G(ind_alpha_vec)=-chi2(0)/2+(alpha/2)*(S-SD(0))-sum(log(Lw))/2 + (NwA/2)*(log(PI)-log(alpha/2))-sum(log(IPA));
				
			//	log_P_alpha_G(ind_alpha_vec)=-chi2(0)/2-(alpha/4)*zAz(0)-sum(log(Lw))/2;
			//	log_P_alpha_G(ind_alpha_vec)=-chi2(0)/2+ (alpha/2)*(S-SD(0)) +sum(log(exp(1)*AS/default_model))/2-sum(log(Lw))/2;
				
				//end of computation of P_alpha_G
			}
			
			G_out=K*A;
			G_V_out=KG_V*A;
			
			if (!boson || col_Gi>0)
			{
				if (cov_diag)
				{
					uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
					errRe=G_V.rows(even_ind)-G_V_out.rows(even_ind);
					errIm=G_V.rows(even_ind+1)-G_V_out.rows(even_ind+1);
				}
				else
				{
					errRe=G_V-G_V_out;
					//errRe=G_V.rows(0,Nn-1)-G_V_out.rows(0,Nn-1);
					//errIm=G_V.rows(Nn,2*Nn-1)-G_V_out.rows(Nn,2*Nn-1);
				}
			}
			else
			{
				errRe=G_V-G_V_out;
			}
			
			if (cov_diag)
				eigv_ind=wn;
			else
				eigv_ind=linspace<vec>(0,errRe.n_rows-1,errRe.n_rows);
			
			if (NM>0)
			{
				M_out=KM*A;
				M_V_out=KM_V*A;
			}
			
			if (save_spec_func)
			{
				M_save.zeros(Nw,2);
				M_save.col(0)=w;
				if (!boson || col_Gi>0)
					M_save.submat(1,1,Nw-2,1)=A;
				else
					M_save.submat(0,1,Nw-2,1)=A;
				sprintf(file_name,output_name_format.c_str(),tem,alpha);
				M_save.save(file_name,raw_ascii);
				
				if (!boson || col_Gi>0)
				{
					M_save.zeros(Nn,3);
					uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
					M_save.col(0)=wn;
					M_save.col(1)=G_out.rows(even_ind);
					M_save.col(2)=G_out.rows(even_ind+1);
				}
				else
				{
					M_save.zeros(Nn,2);
					M_save.col(0)=wn;
					M_save.col(1)=G_out;
				}
				sprintf(file_name,output_G_format.c_str(),tem,alpha);
				M_save.save(file_name,raw_ascii);
				
				if (!boson || col_Gi>0)
				{
					if (cov_diag)
					{
						M_save.zeros(Nn,3);
						M_save.col(0)=wn;
						M_save.col(1)=errRe;
						M_save.col(2)=errIm;
					}
					else
					{
						M_save.zeros(2*Nn,2);
						M_save.col(0)=eigv_ind;
						M_save.col(1)=errRe;
					}
				}
				else
				{
					M_save.zeros(Nn,2);
					M_save.col(0)=eigv_ind;
					M_save.col(1)=errRe;
				}
				sprintf(file_name,output_error_format.c_str(),tem,alpha);
				M_save.save(file_name,raw_ascii);
				
				if (NM>0)
				{
					sprintf(file_name,output_moments_format.c_str(),tem,alpha);
					M_save.zeros(NM,2);
					M_save.col(0)=M;
					M_save.col(1)=M_out;
					M_save.save(file_name,raw_ascii);
				}
			}
			
			pow_alpha=pow_alpha-pow_alpha_step;
			alpha_prec=alpha;
			alpha=pow(10,pow_alpha);
		}
		else
		{
			cout<<"minimize(): change in chi^2 is in the wrong direction\n";
			break;
		}
		
		if (ind_alpha_vec>2)
		{
			if (ind_curv0==0)
			{
				lalpha=log10(alpha_vec.rows(0,ind_alpha_vec));
				lchi2=log10(chi2_vec.rows(0,ind_alpha_vec));
				
				j=1;
				la=lalpha(j-1)-lalpha(j);
				lc=lchi2(j-1)-lchi2(j);
				ls=sqrt(pow(fs*la,2)+pow(lc,2));
				while (ls<fs*smooth_length && j<ind_alpha_vec)
				{
					j++;
					la=lalpha(j-1)-lalpha(j);
					lc=lchi2(j-1)-lchi2(j);
					ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
				}
				if (ls>=fs*smooth_length)	ind_curv0=j;
			}
 
			if (ind_curv0 && ind_curv0<ind_alpha_vec)
			{
				if (ind_curv>=ind_curv0)
				{
					lalpha=log10(alpha_vec.rows(0,ind_alpha_vec));
					lchi2=log10(chi2_vec.rows(0,ind_alpha_vec));
					
					ind_curv++;
					jmin=ind_curv-1;
					la=lalpha(jmin)-lalpha(jmin+1);
					lc=lchi2(jmin)-lchi2(jmin+1);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && jmin>0)
					{
						jmin--;
						la=lalpha(jmin)-lalpha(jmin+1);
						lc=lchi2(jmin)-lchi2(jmin+1);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					jmax=ind_curv+1;
					la=lalpha(jmax-1)-lalpha(jmax);
					lc=lchi2(jmax-1)-lchi2(jmax);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && jmax<ind_alpha_vec)
					{
						jmax++;
						la=lalpha(jmax-1)-lalpha(jmax);
						lc=lchi2(jmax-1)-lchi2(jmax);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					fit_circle_arc(fs*lalpha.rows(jmin,jmax), lchi2.rows(jmin,jmax), arc_params);
					curv_lchi2_lalpha_1(ind_curv)=1.0/arc_params(0);
					
					xc=arc_params(1);
					yc=arc_params(2);
					x1=fs*lalpha(ind_curv);
					y1=lchi2(ind_curv);
					
					dlchi2_lalpha_1(ind_curv)=(xc-x1)/(y1-yc);
				}
				else
				{
					lalpha=log10(alpha_vec.rows(0,ind_alpha_vec));
					lchi2=log10(chi2_vec.rows(0,ind_alpha_vec));
					
					j=ind_alpha_vec-1;
					la=lalpha(j)-lalpha(j+1);
					lc=lchi2(j)-lchi2(j+1);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && j>ind_curv0)
					{
						j--;
						la=lalpha(j)-lalpha(j+1);
						lc=lchi2(j)-lchi2(j+1);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					if (ls>=fs*smooth_length)	ind_curv=j;
				}
			}
			
			if (ind_curv0 && ind_curv>=ind_curv0)
			{
				dlchi2_lalpha_max=dlchi2_lalpha_1.max(ind_max_dchi2_alpha);
				dlchi2_lalpha_min=dlchi2_lalpha_1(ind_curv);
				
				if (alpha<alpha_min)
				{
					if (dlchi2_lalpha_min/dlchi2_lalpha_max>RMAX_dlchi2_lalpha || ind_alpha_vec<Nalpha_min)
					{
						if (!alpha_min_in.size())
						{
							//cout<<"Reducing alpha_min by a factor "<<f_alpha_min<<".\n";
							alpha_min=alpha_min/f_alpha_min;
							alpha_min_too_high=true;
						}
						else
						{
							cout<<"warning: minimum value of alpha reached, but the calculation does not seem over.\n";
						}
					}
				}
				else if (dlchi2_lalpha_min/dlchi2_lalpha_max<RMAX_dlchi2_lalpha/10 && ind_alpha_vec>Nalpha_min)
				{
					if (!alpha_min_in.size() && !alpha_min_too_high)
					{
						cout<<"dlog(chi2)/dlog(alpha) is below the minimum value. Stopping calculation.\n";
						alpha_min=alpha_prec;
					}
					else if (!alpha_too_small)
					{
						alpha_too_small=true;
						if (alpha_min_in.size())
						{
							cout<<"warning: minimum value of alpha seems too small.\n";
						}
					}
				}
			}
		}
		
		ind_alpha++;
		ind_alpha_vec++;
	}
	
	M_save.zeros(Nw,2);
	M_save.col(0)=w;
	if (!boson || col_Gi>0)
		M_save.submat(1,1,Nw-2,1)=A;
	else
		M_save.submat(0,1,Nw-2,1)=A;
	string file_name_str=output_dir_fin;
	file_name_str+=A_alpha_min_name;
	remove(file_name_str.c_str());
	M_save.save(file_name_str,raw_ascii);
	
	if (alpha<alpha_min) alpha_min_too_high=false;
	
	if (ind_alpha>Nalpha)
		cout<<"maximum number of values of alpha reached\n";
	else if (alpha<alpha_min)
	{
		cout<<"minimum value of alpha reached\n";
	}
}


void OmegaMaxEnt_data::minimize_increase_alpha()
{
	double diff_chi2_min=1.0e-3;
	double pow_alpha_step_max=pow_alpha_step_init;
	
	char alpha_output[100], file_name[200];
	char alpha_output_format[]="%d \t alpha: % 1.4e,  Q: % 1.4e,  S: % 1.4e,  chi2: % 1.4e\n";
	double mean_int_dA_prec, mean_int_dA_prec2, A1min, chi2prec, Q, S;
	mat chi2, P, KGMj, U, V, mean_int_dA, M_save;
	vec A1, AS, c1, c2, grS, B, B2, Pd, sK, sK2(NwA), D1, dA1, VPdA, dA_rel, DG;
	vec G_out, G_V_out, errIm, errRe, M_out, M_V_out, eigv_ind;
	uvec ind_c2_sat, ind_An, ind_Anul;
	int ind_alpha, iter_dA;
	uword ind_max_dchi2_alpha;
	int Nalpha_min=200;
	
	bool alpha_too_small=false;
	
	double pow_alpha0=log10(alpha0);
	double pow_alpha=log10(alpha);
	double alpha_prec=pow(10,pow_alpha+pow_alpha_step);
	
	double realmin=1e-100;
	
	double rADmin=realmin;
	vec Amin=rADmin*default_model;
	
	double rADchange=realmin;
	vec Achange=rADchange*default_model;
	
	svd(U,sK,V,KG_V);
	sK2.zeros();
	sK2.rows(0,sK.n_rows-1)=pow(sK,2);
	double alpha_c2_max=sK2.max()*rc2H;
	
	//	mat absMK=abs(KGM.t()*KGM);
	//	double alpha_c2_max=absMK.max()*rc2H;
	
	//	cout<<"sK2.max(): "<<sK2.max()<<endl;
	//	cout<<"alpha_c2_max: "<<alpha_c2_max<<endl;
	
	cout<<"Computing spectrum as a function of alpha with alpha increasing...\n";
	
	vec lalpha, lchi2, arc_params;
	int j, jmin, jmax;
	double fs=f_scale_lalpha_lchi2;
	double smooth_length=chi2_alpha_smooth_range;
	double la, lc, ls;
	double xc, yc, x1, y1;
	
	c1=(log(rADchange)+1)*dwS;
	vec c2_alpha=2*PI*alpha_c2_max*ones<vec>(NwA);
	
	ind_alpha=1;
	DG=GM-KGM*A;
	chi2=DG.t()*DG;
	while (ind_alpha<=Nalpha && alpha<=alpha0_default)
	{
		A1=A;
		
		c2=c2_alpha/alpha;
		//c2=c2_0;
		//ind_c2_sat=find((alpha*c2)/(2*PI)>alpha_c2_max);
		//if (ind_c2_sat.n_rows)
		//	c2.rows(ind_c2_sat)=(2*PI*alpha_c2_max/alpha)*ones<vec>(ind_c2_sat.n_rows);
		
		grS=dwS % log(A1/default_model) + dwS;
		ind_An=find(A1/default_model<rADchange);
		if (ind_An.n_rows)
		{
			grS.rows(ind_An)=2*c2.rows(ind_An) % (A1.rows(ind_An)-Achange.rows(ind_An))+c1.rows(ind_An);
		}
		
		B=KGM.t()*(GM-KGM*A1)-(alpha/(4*PI))*grS;
		
		Pd=sqrt(4*PI*A1/(alpha*dwS));
		if (ind_An.n_rows)
			Pd.rows(ind_An)=sqrt(2*PI/(alpha*c2(ind_An)));
		P=diagmat(Pd);
		
		KGMj=KGM*P;
		svd(U,sK,V,KGMj);
		
		B2=V.t()*(P*B);
		sK2.zeros();
		sK2.rows(0,sK.n_rows-1)=pow(sK,2);
		D1=sK2+1;
		VPdA=B2/D1;
		dA1=P*(V*VPdA);
		
		mean_int_dA=abs(dA1.t())*dwS;
		mean_int_dA_prec=2*mean_int_dA(0);
		mean_int_dA_prec2=mean_int_dA_prec;
		
		iter_dA=1;
		
		A1=A1+dA1;
		
		ind_Anul=find(A1==0);
		if (ind_Anul.n_rows)
		{
			A1.rows(ind_Anul)=Amin.rows(ind_Anul);
			//	A1.rows(ind_Anul)=arma::max(Amin.rows(ind_Anul),DBL_MIN);
		}
		
		A1min=min(A1-Amin);
		
		while ( (mean_int_dA(0)>tol_int_dA || A1min<0) && iter_dA<Niter_dA_max && (mean_int_dA(0)<mean_int_dA_prec || mean_int_dA(0)<mean_int_dA_prec2))
		{
			grS=dwS % log(A1/default_model) + dwS;
			ind_An=find(A1/default_model<rADchange);
			if (ind_An.n_rows)
			{
				grS.rows(ind_An)=2*c2.rows(ind_An) % (A1.rows(ind_An)-Achange.rows(ind_An))+c1.rows(ind_An);
			}
			
			B=KGM.t()*(GM-KGM*A1)-(alpha/(4*PI))*grS;
			
			Pd=sqrt(4*PI*A1/(alpha*dwS));
			if (ind_An.n_rows)
				Pd.rows(ind_An)=sqrt(2*PI/(alpha*c2(ind_An)));
			P=diagmat(Pd);
			
			KGMj=KGM*P;
			svd(U,sK,V,KGMj);
			
			B2=V.t()*(P*B);
			sK2.zeros();
			sK2.rows(0,sK.n_rows-1)=pow(sK,2);
			D1=sK2+1;
			VPdA=B2/D1;
			dA1=P*(V*VPdA);
			
			mean_int_dA_prec2=mean_int_dA_prec;
			mean_int_dA_prec=mean_int_dA(0);
			mean_int_dA=abs(dA1.t())*dwS;
			
			if (mean_int_dA(0)<mean_int_dA_prec)
			{
				A1=A1+dA1;
				A1min=min(A1-Amin);
			}
			
			ind_Anul=find(A1==0);
			if (ind_Anul.n_rows)
			{
				A1.rows(ind_Anul)=Amin.rows(ind_Anul);
				//	A1.rows(ind_Anul)=arma::max(Amin.rows(ind_Anul),DBL_MIN);
			}
			
			iter_dA++;
		}
		
		/*
		 if (mean_int_dA(0)>mean_int_dA_prec || iter_dA==Niter_dA_max)
		 {
			cout<<"alpha= "<<alpha<<"  iteration: "<<iter_dA<<"  integrated absolute variation in A: "<<mean_int_dA_prec<<endl;
		 }
		 */
		chi2prec=chi2(0);
		DG=GM-KGM*A1;
		chi2=DG.t()*DG;
		AS=A1;
		ind_An=find(AS/default_model<rADchange);
		if (ind_An.n_rows)
			AS.rows(ind_An)=Amin.rows(ind_An);
		S=-sum(AS % dwS % log(AS/default_model))/(2*PI);
		
		if (chi2(0)/chi2prec<R_chi2_min && alpha<alpha_prec && pow_alpha_step>2*pow_alpha_step_min)
			pow_alpha_step=pow_alpha_step/2;
		
		if (alpha>alpha_prec && abs((chi2(0)-chi2prec)/chi2prec)<diff_chi2_min && pow_alpha_step<2*pow_alpha_step_max)
			pow_alpha_step=2*pow_alpha_step;
		
		if ((chi2(0)<chi2prec && alpha<=alpha_prec) || (chi2(0)>chi2prec && alpha>alpha_prec) || pow_alpha==pow_alpha0)
		{
			Aprec.cols(0,NAprec-2)=Aprec.cols(1,NAprec-1);
			Aprec.col(NAprec-1)=A;
			A=A1;
			alpha_vec(ind_alpha_vec)=alpha;
			chi2_vec(ind_alpha_vec)=chi2(0);
			S_vec(ind_alpha_vec)=S;
			vec logA=log(A/default_model);
			ind_An=find(A/default_model<rADchange);
			if (ind_An.n_rows)
			{
				logA.rows(ind_An)=log(rADmin)*ones<vec>(ind_An.n_rows);
			}
			S_vec(ind_alpha_vec)=-sum(A % dwS % logA)/(2*PI);
			Aw_samp.row(ind_alpha_vec)=trans(A(w_sample_ind));
			
			if (print_alpha)
			{
				Q=chi2(0)-alpha*S_vec(ind_alpha_vec);
				sprintf(alpha_output,alpha_output_format,ind_alpha,alpha,Q,S_vec(ind_alpha_vec),chi2(0));
				cout<<alpha_output;
			}
			
			G_out=K*A;
			G_V_out=KG_V*A;
			
			if (!boson || col_Gi>0)
			{
				if (cov_diag)
				{
					uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
					errRe=G_V.rows(even_ind)-G_V_out.rows(even_ind);
					errIm=G_V.rows(even_ind+1)-G_V_out.rows(even_ind+1);
				}
				else
				{
					errRe=G_V-G_V_out;
					//errRe=G_V.rows(0,Nn-1)-G_V_out.rows(0,Nn-1);
					//errIm=G_V.rows(Nn,2*Nn-1)-G_V_out.rows(Nn,2*Nn-1);
				}
			}
			else
			{
				errRe=G_V-G_V_out;
			}
			
			if (cov_diag)
				eigv_ind=wn;
			else
				eigv_ind=linspace<vec>(0,errRe.n_rows-1,errRe.n_rows);
			
			if (NM>0)
			{
				M_out=KM*A;
				M_V_out=KM_V*A;
			}
			
			if (save_spec_func)
			{
				
				M_save.zeros(Nw,2);
				M_save.col(0)=w;
				if (!boson || col_Gi>0)
					M_save.submat(1,1,Nw-2,1)=A;
				else
					M_save.submat(0,1,Nw-2,1)=A;
				sprintf(file_name,output_name_format.c_str(),tem,alpha);
				M_save.save(file_name,raw_ascii);
				
				
				if (!boson || col_Gi>0)
				{
					M_save.zeros(Nn,3);
					uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
					M_save.col(0)=wn;
					M_save.col(1)=G_out.rows(even_ind);
					M_save.col(2)=G_out.rows(even_ind+1);
				}
				else
				{
					M_save.zeros(Nn,2);
					M_save.col(0)=wn;
					M_save.col(1)=G_out;
				}
				sprintf(file_name,output_G_format.c_str(),tem,alpha);
				M_save.save(file_name,raw_ascii);
				
				if (!boson || col_Gi>0)
				{
					if (cov_diag)
					{
						M_save.zeros(Nn,3);
						M_save.col(0)=wn;
						M_save.col(1)=errRe;
						M_save.col(2)=errIm;
					}
					else
					{
						M_save.zeros(2*Nn,2);
						M_save.col(0)=eigv_ind;
						M_save.col(1)=errRe;
					}
				}
				else
				{
					M_save.zeros(Nn,2);
					M_save.col(0)=eigv_ind;
					M_save.col(1)=errRe;
				}
				sprintf(file_name,output_error_format.c_str(),tem,alpha);
				M_save.save(file_name,raw_ascii);
				
				if (NM>0)
				{
					sprintf(file_name,output_moments_format.c_str(),tem,alpha);
					M_save.zeros(NM,2);
					M_save.col(0)=M;
					M_save.col(1)=M_out;
					M_save.save(file_name,raw_ascii);
				}
			}
			
			pow_alpha=pow_alpha+pow_alpha_step;
			alpha_prec=alpha;
			alpha=pow(10,pow_alpha);
		}
		else
		{
			cout<<"minimize(): change in chi^2 is in the wrong direction\n";
			break;
		}
		
	
		if (ind_alpha_vec>2)
		{
			if (ind_curv0==0)
			{
				lalpha=log10(alpha_vec.rows(0,ind_alpha_vec));
				lchi2=log10(chi2_vec.rows(0,ind_alpha_vec));
				
				j=1;
				la=lalpha(j-1)-lalpha(j);
				lc=lchi2(j-1)-lchi2(j);
				ls=sqrt(pow(fs*la,2)+pow(lc,2));
				while (ls<fs*smooth_length && j<ind_alpha_vec)
				{
					j++;
					la=lalpha(j-1)-lalpha(j);
					lc=lchi2(j-1)-lchi2(j);
					ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
				}
				if (ls>=fs*smooth_length)	ind_curv0=j;
			}
			
			if (ind_curv0 && ind_curv0<ind_alpha_vec)
			{
				if (ind_curv>=ind_curv0)
				{
					lalpha=log10(alpha_vec.rows(0,ind_alpha_vec));
					lchi2=log10(chi2_vec.rows(0,ind_alpha_vec));
					
					ind_curv++;
					jmin=ind_curv-1;
					la=lalpha(jmin)-lalpha(jmin+1);
					lc=lchi2(jmin)-lchi2(jmin+1);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && jmin>0)
					{
						jmin--;
						la=lalpha(jmin)-lalpha(jmin+1);
						lc=lchi2(jmin)-lchi2(jmin+1);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					jmax=ind_curv+1;
					la=lalpha(jmax-1)-lalpha(jmax);
					lc=lchi2(jmax-1)-lchi2(jmax);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && jmax<ind_alpha_vec)
					{
						jmax++;
						la=lalpha(jmax-1)-lalpha(jmax);
						lc=lchi2(jmax-1)-lchi2(jmax);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					fit_circle_arc(fs*lalpha.rows(jmin,jmax), lchi2.rows(jmin,jmax), arc_params);
					curv_lchi2_lalpha_1(ind_curv)=1.0/arc_params(0);
					
					xc=arc_params(1);
					yc=arc_params(2);
					x1=fs*lalpha(ind_curv);
					y1=lchi2(ind_curv);
					
					dlchi2_lalpha_1(ind_curv)=(xc-x1)/(y1-yc);
				}
				else
				{
					lalpha=log10(alpha_vec.rows(0,ind_alpha_vec));
					lchi2=log10(chi2_vec.rows(0,ind_alpha_vec));
					
					j=ind_alpha_vec-1;
					la=lalpha(j)-lalpha(j+1);
					lc=lchi2(j)-lchi2(j+1);
					ls=sqrt(pow(fs*la,2)+pow(lc,2));
					while (ls<fs*smooth_length && j>ind_curv0)
					{
						j--;
						la=lalpha(j)-lalpha(j+1);
						lc=lchi2(j)-lchi2(j+1);
						ls=ls+sqrt(pow(fs*la,2)+pow(lc,2));
					}
					if (ls>=fs*smooth_length)	ind_curv=j;
				}
			}
			
			if (ind_curv0 && ind_curv>=ind_curv0)
			{
				dlchi2_lalpha_max=dlchi2_lalpha_1.max(ind_max_dchi2_alpha);
				dlchi2_lalpha_min=dlchi2_lalpha_1(ind_curv);
				
				if (alpha<alpha_min)
				{
					if (dlchi2_lalpha_min/dlchi2_lalpha_max>RMAX_dlchi2_lalpha || ind_alpha_vec<Nalpha_min)
					{
						if (!alpha_min_in.size())
						{
							//cout<<"Reducing alpha_min by a factor "<<f_alpha_min<<".\n";
							alpha_min=alpha_min/f_alpha_min;
						}
						else
						{
							cout<<"warning: minimum value of alpha reached, but the calculation does not seem over.\n";
						}
					}
				}
				else if (dlchi2_lalpha_min/dlchi2_lalpha_max<RMAX_dlchi2_lalpha/10 && ind_alpha_vec>Nalpha_min)
				{
					if (!alpha_min_in.size())
					{
						cout<<"dlog(chi2)/dlog(alpha) is below the minimum value. Stopping calculation.\n";
						alpha_min=alpha_prec;
					}
					else if (!alpha_too_small)
					{
						alpha_too_small=true;
						if (alpha_min_in.size())
							cout<<"warning: minimum value of alpha seems too small.\n";
					}
				}
			}
			
		}
		
		ind_alpha++;
		ind_alpha_vec++;
	}
	
	if (ind_alpha>Nalpha)
		cout<<"maximum number of values alpha reached\n";
	else if (alpha<alpha_min)
		cout<<"minimum value of alpha reached\n";
}


bool OmegaMaxEnt_data::diagonalize_covariance()
{
	mat VM, WM, VG, WG;
	
	if (NM>0)
	{
		if (covm_diag)
		{
			VM.eye(NM,NM);
			WM=diagmat(1.0/errM);
		}
		else
		{
			int ind0_M=1;
			if (boson)	ind0_M=0;
			vec COVM_eig;
			mat VMtmp;
			
		//	cout<<"COVM:\n"<<COVM.submat(ind0_M,ind0_M,NM-1,NM-1)<<endl;
			
			if (!eig_sym(COVM_eig,VMtmp,COVM.submat(ind0_M,ind0_M,NM-1,NM-1)))
			{
				cout<<"diagonalize_covariance() error: diagonalization of moments covariance matrix failed\n";
				return false;
			}
			if (COVM_eig.min()<=0)
			{
				cout<<"diagonalize_covariance() warning: the moments covariance matrix has non-positive eigenvalues\n";
				
				VM.eye(NM,NM);
				vec errM_eig(NM);
				if (!boson) errM_eig(0)=sqrt(COVM(0,0));
				COVM_eig=COVM.submat(ind0_M,ind0_M,NM-1,NM-1).diag();
				errM_eig.rows(ind0_M,NM-1)=sqrt(COVM_eig);
				WM=diagmat(1.0/errM_eig);
				
			//	return false;
			}
			else
			{
				VM.zeros(NM,NM);
				if (!boson) VM(0,0)=1;
				VM.submat(ind0_M,ind0_M,NM-1,NM-1)=VMtmp;
				vec errM_eig(NM);
				if (!boson) errM_eig(0)=sqrt(COVM(0,0));
				errM_eig.rows(ind0_M,NM-1)=sqrt(COVM_eig);
				WM=diagmat(1.0/errM_eig);
			}
//			cout<<"COVM:\n"<<COVM<<endl;
//			cout<<"VM.t()*COVM*VM:\n"<<VM.t()*COVM*VM<<endl;
//			cout<<"COVM_eig:\n"<<COVM_eig<<endl;
		}
		
		if (!boson && covm_diag && M1_in.size() && M3_in.size() && !M2_in.size() && !eval_moments && maxM>2)
		{
			NM=3;
			M_ord.zeros(NM);
			M_ord(0)=0;
			M_ord(1)=1;
			M_ord(2)=3;
			M.zeros(NM);
			M(0)=M0;
			M(1)=M1;
			M(2)=M3;
			errM.zeros(NM);
			errM(0)=errM0;
			errM(1)=errM1;
			errM(2)=errM3;
			WM=diagmat(1.0/errM);
			M_V=WM*M;
			uvec indM={0,1,3};
			KM_V=WM*KM.rows(indM);
		}
		else
		{
			KM=KM.rows(0,NM-1);
			M_V=WM*VM.t()*M;
			KM_V=WM*VM.t()*KM;
		}
		
		KM_V=KM_V.cols(ind0,Nw-2);
		KM=KM.cols(ind0,Nw-2);
	}

	if (cov_diag)
	{
		VG.eye(2*Nn,2*Nn);
		WG=diagmat(1.0/errG);
	}
	else
	{
		vec COVG_eig;
		mat VGtmp;
		if (!eig_sym(COVG_eig,VGtmp,COV))
		{
			cout<<"diagonalize_covariance() error: diagonalization of covariance matrix failed\n";
			return false;
		}
//		cout<<"COVG_eig:\n"<<COVG_eig<<endl;
		mat PV=eye<mat>(2*Nn,2*Nn);
		PV=flipud(PV);
		COVG_eig=flipud(COVG_eig);
		VG=VGtmp*PV;
		vec GVtmp=VG.t()*Gchi2;
		mat SGV=diagmat(sign(GVtmp));
		SGV=SGV-abs(SGV)+eye<mat>(2*Nn,2*Nn);
		VG=VG*SGV;
		WG=diagmat(1.0/sqrt(COVG_eig));
		if (COVG_eig.min()<=0)
		{
			cout<<"diagonalize_covariance() error: the covariance matrix has non-positive eigenvalues\n";
			return false;
		}
	}
	
	G_V=WG*VG.t()*Gchi2;
	KG_V=WG*VG.t()*K.cols(ind0,Nw-2);
	K=K.cols(ind0,Nw-2);
	
	if (NM>0)
	{
		GM=join_vert(M_V, G_V);
		KGM=join_vert(KM_V, KG_V);
	}
	else
	{
		GM=G_V;
		KGM=KG_V;
	}
	NGM=GM.n_rows;
	
	cout<<"number of terms in chi2: "<<NGM<<endl;
	
	return true;
}

bool OmegaMaxEnt_data::Kernel_G_fermions_grid_transf_omega()
{
	bool use_HF_exp=true;
	double fg=1.7;
	int pngmax=100;
	double fi=fg;
	double fr=fi;
	int pnmax=50;
	double fd=fg;
	int pndmax=pngmax;
	
	cout<<"defining kernel matrix...\n";
	
	Kcx.zeros(Nn,Nw);
	KM.zeros(4,Nw);
	
	mat MM;
	spline_matrix_grid_transf(w, MM);
	
/*
	//testing the spline matrix
	rowvec x0={wl, 0, wr/2};
	rowvec s0={SW/10, SW/20, SW/5};
	rowvec wgt={1,1,1};
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
 	vec coeffs=MM*test_A;
	
	mat MM1;
	spline_matrix_G_part(w, Nw_lims, ws, MM1);
	int Ncfs0=MM1.n_rows;
	vec coeffs0=MM1*test_A;
	vec coeffs1(4*(Nw+1));
	coeffs1(0)=coeffs0(0);
	coeffs1(1)=coeffs0(1);
	coeffs1(2)=0;
	coeffs1(3)=0;
	coeffs1(4*(Nw+1)-4)=coeffs0(Ncfs0-2);
	coeffs1(4*(Nw+1)-3)=coeffs0(Ncfs0-1);
	coeffs1(4*(Nw+1)-2)=0;
	coeffs1(4*(Nw+1)-1)=0;
	uvec ind_tmp=linspace<uvec>(1,Nw-1,Nw-1);
	coeffs1.rows(4*ind_tmp)=coeffs0(3*ind_tmp-1);
	coeffs1.rows(4*ind_tmp+1)=coeffs0(3*ind_tmp);
	coeffs1.rows(4*ind_tmp+2)=coeffs0(3*ind_tmp+1);
	ind_tmp=linspace<uvec>(1,Nw_lims(1),Nw_lims(1));
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp-1);
	ind_tmp=linspace<uvec>(Nw_lims(1)+1,Nw-1,Nw-Nw_lims(1)-1);
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp);

 	double Nw2=400;
	vec w2=linspace<vec>(wl-SW,wr+SW,Nw2);

	vec test_A2;
	sum_gaussians(w2, x0, s0, wgt, test_A2);

	vec test_A3, test_A4;
	spline_val_grid_transf(w2, w, coeffs, test_A3);
	spline_val_G_part(w2, w, Nw_lims, ws, coeffs1, test_A4);

	graph_2D g1, g2, g3;
	char xl[]="$\\\\omega$";
	char yl[]="A";
	char attr1[]="'v-',color='b', markeredgecolor='b', markerfacecolor='none'";
	char attr2[]="'^-',color='r', markeredgecolor='r', markerfacecolor='none'";
	char attr3[]="'s-',color='m', markeredgecolor='m', markerfacecolor='none'";
	double xlims[2], ylims[2];
	xlims[0]=w2(0);
	xlims[1]=w2(w2.n_rows-1);
	vec test_A_all=join_vert(test_A3,test_A4);
	ylims[0]=test_A_all.min();
	ylims[1]=1.1*test_A2.max();

	g1.add_data(w2.memptr(),test_A2.memptr(),w2.n_rows);
	g1.add_attribute(attr1);
	g1.add_data(w2.memptr(),test_A3.memptr(),w2.n_rows);
	g1.add_attribute(attr2);
	g1.add_data(w2.memptr(),test_A4.memptr(),w2.n_rows);
	g1.add_attribute(attr3);
	g1.set_axes_labels(xl,yl);
	g1.set_axes_lims(xlims,ylims);
	g1.curve_plot();

	char yl2[]="Delta A";
	//vec Delta_A=(test_A2-test_A3)/test_A2;
	vec Delta_A=(test_A2-test_A3);
	ylims[0]=Delta_A.min();
	ylims[1]=Delta_A.max();
	g2.add_data(w2.memptr(),Delta_A.memptr(),w2.n_rows);
	g2.set_axes_labels(xl,yl2);
	g2.set_axes_lims(xlims,ylims);
	g2.curve_plot();
	
	char attr[]="color='r'";
	vec Delta_A1=(test_A2-test_A4);
	ylims[0]=Delta_A1.min();
	ylims[1]=Delta_A1.max();
	g3.add_data(w2.memptr(),Delta_A1.memptr(),w2.n_rows);
	g3.add_attribute(attr);
	g3.set_axes_labels(xl,yl2);
	g3.set_axes_lims(xlims,ylims);
	g3.curve_plot();

	graph_2D::show_figures();
	*/
	
	dcomplex i(0,1);
	int Nint=Nw-1;
	
	cx_mat Ka=zeros<cx_mat>(Nn,Nint);
	cx_mat Kb=zeros<cx_mat>(Nn,Nint);
	cx_mat Kc=zeros<cx_mat>(Nn,Nint);
	cx_mat Kd=zeros<cx_mat>(Nn,Nint);
	
	mat Wn=wn*ones<rowvec>(Nint);
	mat W=ones<vec>(Nn)*w.t();
	vec vdw=w.rows(1,Nw-1)-w.rows(0,Nw-2);
	
	mat logc=log((pow(Wn,2)+pow(W.cols(1,Nint),2))/(pow(Wn,2)+pow(W.cols(0,Nint-1),2)));
	mat atanc=atan((Wn % (W.cols(1,Nint)-W.cols(0,Nint-1)))/(W.cols(1,Nint) % W.cols(0,Nint-1)+pow(Wn,2)));
	cx_mat logc2=logc/2+i*atanc;
	cx_mat dWn=i*Wn-W.cols(0,Nint-1);
	mat dW=W.cols(1,Nint)-W.cols(0,Nint-1);
	cx_mat Rdwn_dw=dWn/dW;
	
	Ka=-pow(Rdwn_dw,2)-Rdwn_dw/2-1.0/3.0-pow(Rdwn_dw,3) % logc2;
	Kb=-Rdwn_dw-0.5-pow(Rdwn_dw,2) % logc2;
	Kc=-1.0 - Rdwn_dw % logc2;
	Kd=-logc2;
	
	int Pmax=2*pnmax+4;
	imat MP;
	pascal(Pmax+1,MP);
	
	int j;
	if (use_HF_exp)
	{
		double wtmp;
		int jni, jnr, p, l;
		for (j=0; j<Nw-1; j++)
		{
			wtmp=abs(w(j));
			if (abs(w(j+1))>wtmp) wtmp=abs(w(j+1));
			jni=0;
			while (jni<Nn && wn(jni)<fi*wtmp) jni++;
			while (jni<Nn && pow(wn(jni),2*pnmax+1)==0) jni++;
			jnr=1;
			while (jnr<Nn && wn(jnr)<fr*wtmp) jnr++;
			while (jnr<Nn && pow(wn(jnr),2*pnmax)==0) jnr++;
			
			if (jni<Nn || jnr<Nn)
			{
				double wj=w(j);
				double dw1=w(j+1)-wj;
				vec dwp=zeros<vec>(Pmax);
				dwp(0)=dw1;
				dwp(1)=pow(dw1,2)+2*wj*dw1;
				for (p=3; p<=Pmax; p++)
				{
					dwp(p-1)=0;
					for (l=0; l<p; l++)
					{
						dwp(p-1)=dwp(p-1)+MP(l,p-l)*pow(dw1,p-l)*pow(wj,l);
					}
				}
				cx_vec vtmp;
				if (jni<Nn)
				{
					vtmp.zeros(Nn-jni);
					vtmp.set_real(real(Ka.submat(jni,j,Nn-1,j)));
					Ka.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kb.submat(jni,j,Nn-1,j)));
					Kb.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kc.submat(jni,j,Nn-1,j)));
					Kc.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kd.submat(jni,j,Nn-1,j)));
					Kd.submat(jni,j,Nn-1,j)=vtmp;
					
					for (p=2*pnmax+1; p>=1; p-=2)
					{
						Ka.submat(jni,j,Nn-1,j)=Ka.submat(jni,j,Nn-1,j) + (i*pow(-1,(p-1)/2)*(pow(wj,3)*dwp(p-1)/p - 3.0*pow(wj,2)*dwp(p)/(p+1) + 3.0*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jni,Nn-1),p))/pow(dw1,3);
						Kb.submat(jni,j,Nn-1,j)=Kb.submat(jni,j,Nn-1,j) + (i*pow(-1,(p-1)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jni,Nn-1),p))/pow(dw1,2);
						Kc.submat(jni,j,Nn-1,j)=Kc.submat(jni,j,Nn-1,j) + (i*pow(-1,(p-1)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jni,Nn-1),p))/dw1;
						Kd.submat(jni,j,Nn-1,j)=Kd.submat(jni,j,Nn-1,j) - i*pow(-1,(p-1)/2)*dwp(p-1)/(p*pow(wn.rows(jni,Nn-1),p));
					}
				}
				if (jnr<Nn)
				{
					vtmp.zeros(Nn-jnr);
					vtmp.set_imag(imag(Ka.submat(jnr,j,Nn-1,j)));
					Ka.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kb.submat(jnr,j,Nn-1,j)));
					Kb.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kc.submat(jnr,j,Nn-1,j)));
					Kc.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kd.submat(jnr,j,Nn-1,j)));
					Kd.submat(jnr,j,Nn-1,j)=vtmp;
					for (p=2*pnmax; p>=2; p-=2)
					{
						Ka.submat(jnr,j,Nn-1,j)=Ka.submat(jnr,j,Nn-1,j) + (pow(-1,(p-2)/2)*(pow(wj,3)*dwp(p-1)/p - 3*pow(wj,2)*dwp(p)/(p+1) + 3*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jnr,Nn-1),p))/pow(dw1,3);
						Kb.submat(jnr,j,Nn-1,j)=Kb.submat(jnr,j,Nn-1,j) + (pow(-1,(p-2)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jnr,Nn-1),p))/pow(dw1,2);
						Kc.submat(jnr,j,Nn-1,j)=Kc.submat(jnr,j,Nn-1,j) + (pow(-1,(p-2)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jnr,Nn-1),p))/dw1;
						Kd.submat(jnr,j,Nn-1,j)=Kd.submat(jnr,j,Nn-1,j) - pow(-1,(p-2)/2)*dwp(p-1)/(p*pow(wn.rows(jnr,Nn-1),p));
					}
				}
			}
		}
	}
	
	Ka=Ka/(2*PI);
	Kb=Kb/(2*PI);
	Kc=Kc/(2*PI);
	Kd=Kd/(2*PI);
	
	mat	Pa=zeros<mat>(Nint,4*Nint);
	mat	Pb=zeros<mat>(Nint,4*Nint);
	mat	Pc=zeros<mat>(Nint,4*Nint);
	mat	Pd=zeros<mat>(Nint,4*Nint);
	
	for (j=0; j<Nint; j++)
	{
		Pa(j,4*j)=1;
		Pb(j,4*j+1)=1;
		Pc(j,4*j+2)=1;
		Pd(j,4*j+3)=1;
	}
	
	Kcx=(Ka*Pa+Kb*Pb+Kc*Pc+Kd*Pd)*MM;
	
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	K.zeros(2*Nn,Nw);
	K.rows(even_ind)=real(Kcx);
	K.rows(even_ind+1)=imag(Kcx);
	
	/*
	//testing the kernel matrix
	rowvec x0={-2, 0, 3};
	rowvec s0={0.8, 0.3, 1};
	rowvec wgt={1,1,1};
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
	vec Gtest1=K*test_A;
	vec Gr_test1=Gtest1.rows(even_ind);
	vec Gi_test1=Gtest1.rows(even_ind+1);
	cx_vec Gtest=Kcx*test_A;
	vec Gr_test=real(Gtest);
	vec Gi_test=imag(Gtest);
	
	graph_2D g1, g2, g3, g4;
	char attr1[]="'o',color='b',markeredgecolor='b',markerfacecolor='none'";
	char attr2[]="'s',color='r',markeredgecolor='r',markerfacecolor='none'";
	char attr3[]="'.-',color='b'";
	char attr4[]="'.-',color='m'";
	char attr5[]="'.-',color='r'";
	char attr6[]="'.-',color='c'";
	
	g1.add_data(wn.memptr(), Gr.memptr(), Nn);
	g1.add_attribute(attr1);
	g1.add_data(wn.memptr(), Gr_test.memptr(), Nn);
	g1.add_attribute(attr2);
	g1.curve_plot();
	
	g2.add_data(wn.memptr(), Gi.memptr(), Nn);
	g2.add_attribute(attr1);
	g2.add_data(wn.memptr(), Gi_test.memptr(), Nn);
	g2.add_attribute(attr2);
	g2.curve_plot();
	
	vec DGr1=(Gr_test1-Gr)/Gr, DGr=(Gr_test-Gr)/Gr;
	vec DGi1=(Gi_test1-Gi)/Gi, DGi=(Gi_test-Gi)/Gi;
	g3.add_data(wn.memptr(),DGr1.memptr(),Nn);
	g3.add_attribute(attr3);
	g3.add_data(wn.memptr(),DGr.memptr(),Nn);
	g3.add_attribute(attr4);
	g3.curve_plot();
	
	g4.add_data(wn.memptr(),DGi1.memptr(),Nn);
	g4.add_attribute(attr5);
	g4.add_data(wn.memptr(),DGi.memptr(),Nn);
	g4.add_attribute(attr6);
	g4.curve_plot();
	
	//plot(g3,wn,(Gr_test-Gr)/Gr,NULL,NULL,attr3);
	//plot(g4,wn,(Gi_test-Gi)/Gi,NULL,NULL,attr4);
	
	graph_2D::show_figures();
	*/
	
	rowvec Knorm_a=zeros<rowvec>(Nint);
	rowvec Knorm_b=zeros<rowvec>(Nint);
	rowvec Knorm_c=zeros<rowvec>(Nint);
	rowvec Knorm_d=zeros<rowvec>(Nint);
	
	Knorm_a=trans(pow(w.rows(1,Nw-1)-w.rows(0,Nw-2),4)/4);
	Knorm_b=trans(pow(w.rows(1,Nw-1)-w.rows(0,Nw-2),3)/3);
	Knorm_c=trans(pow(w.rows(1,Nw-1)-w.rows(0,Nw-2),2)/2);
	Knorm_d=trans(w.rows(1,Nw-1)-w.rows(0,Nw-2));
	
	rowvec KM0=(Knorm_a*Pa+Knorm_b*Pb+Knorm_c*Pc+Knorm_d*Pd)*MM/(2*PI);
	
	mat Wjc=diagmat(w.rows(0,Nint-1));
	
	rowvec KM1_a_c_tmp=trans(pow(w.rows(1,Nw-1)-w.rows(0,Nw-2),5)/5.0);
	
	rowvec KM1_a_c=KM1_a_c_tmp+Knorm_a*Wjc;
	rowvec KM1_b_c=Knorm_a+Knorm_b*Wjc;
	rowvec KM1_c_c=Knorm_b+Knorm_c*Wjc;
	rowvec KM1_d_c=Knorm_c+Knorm_d*Wjc;
	
	rowvec KM1=(KM1_a_c*Pa+KM1_b_c*Pb+KM1_c_c*Pc+KM1_d_c*Pd)*MM/(2*PI);
	
	rowvec KM2_a_c_tmp=trans(pow(w.rows(1,Nw-1)-w.rows(0,Nw-2),6)/6);
	
	rowvec KM2_a_c=KM2_a_c_tmp+2*KM1_a_c_tmp*Wjc+Knorm_a*pow(Wjc,2);
	rowvec KM2_b_c=KM1_a_c_tmp+2*Knorm_a*Wjc+Knorm_b*pow(Wjc,2);
	rowvec KM2_c_c=Knorm_a+2*Knorm_b*Wjc+Knorm_c*pow(Wjc,2);
	rowvec KM2_d_c=Knorm_b+2*Knorm_c*Wjc+Knorm_d*pow(Wjc,2);
	
	rowvec KM2=(KM2_a_c*Pa+KM2_b_c*Pb+KM2_c_c*Pc+KM2_d_c*Pd)*MM/(2*PI);
	
	rowvec KM3_a_c_tmp=trans(pow(w.rows(1,Nw-1)-w.rows(0,Nw-2),7)/7);
	
	rowvec KM3_a_c=KM3_a_c_tmp+3*KM2_a_c_tmp*Wjc+3*KM1_a_c_tmp*pow(Wjc,2)+Knorm_a*pow(Wjc,3);
	rowvec KM3_b_c=KM2_a_c_tmp+3*KM1_a_c_tmp*Wjc+3*Knorm_a*pow(Wjc,2)+Knorm_b*pow(Wjc,3);
	rowvec KM3_c_c=KM1_a_c_tmp+3*Knorm_a*Wjc+3*Knorm_b*pow(Wjc,2)+Knorm_c*pow(Wjc,3);
	rowvec KM3_d_c=Knorm_a+3*Knorm_b*Wjc+3*Knorm_c*pow(Wjc,2)+Knorm_d*pow(Wjc,3);
	
	rowvec KM3=(KM3_a_c*Pa+KM3_b_c*Pb+KM3_c_c*Pc+KM3_d_c*Pd)*MM/(2*PI);
	
	KM.row(0)=KM0;
	KM.row(1)=KM1;
	KM.row(2)=KM2;
	KM.row(3)=KM3;
	
	/*
	//test moments
	rowvec x0={-2, 0, 3};
	rowvec s0={0.8, 0.3, 1};
	rowvec wgt={1,1,1};
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
	cout<<KM*test_A<<endl;
	*/
	
	cout<<"kernel matrix defined.\n";
	
	return true;
}

bool OmegaMaxEnt_data::Kernel_G_fermions_grid_transf_2()
{
	bool use_HF_exp=true;
	double fg=1.7;
	int pngmax=100;
	double fi=fg;
	double fr=fi;
	int pnmax=50;
	double fd=fg;
	int pndmax=pngmax;
	
	cout<<"defining kernel matrix...\n";
	
	Kcx.zeros(Nn,Nw);
 
	int Nug=Nw_lims(0);
	int Nud=Nw-Nw_lims(1)-1;
	
	vec ug, ud;
	if (Du_constant)
	{
		double dug=dwl/((wl-dwl-w0l)*(wl-w0l));
		vec ug_int=linspace<vec>(1,Nug,Nug);
		ug=-dug*ug_int;
		
		double dud=dwr/((wr-w0r)*(wr+dwr-w0r));
		vec ud_int=linspace<vec>(Nud,1,Nud);
		ud=dud*ud_int;
	}
	else
	{
		ug=1.0/(w.rows(0,Nug-1)-w0l);
		ud=1.0/(w.rows(Nw_lims(1)+1,Nw-1)-w0r);
	}
	
	mat MM;
	spline_matrix_grid_transf_G_part_2(w, Nw_lims, ws, MM);
	
/*
	 //testing the spline matrix
	 rowvec x0={wl, 0, wr/2};
	 rowvec s0={SW/10, SW/20, SW/5};
	 rowvec wgt={1,1,1};
	 
	 vec test_A;
	 sum_gaussians(w, x0, s0, wgt, test_A);
	 
	 vec coeffs=MM*test_A;
	 
	 mat MM1;
	 spline_matrix_grid_transf_G_part(w, Nw_lims, ws, MM1);
	
	vec coeffs1=MM1*test_A;
	
	 double Nw2=400;
	 vec w2=linspace<vec>(wl-SW,wr+SW,Nw2);
	 
	 vec test_A2;
	 sum_gaussians(w2, x0, s0, wgt, test_A2);
	 
	 vec test_A3, test_A4;
	 spline_val_G_part_grid_transf(w2, w, Nw_lims, ws, coeffs, test_A3);
	spline_val_G_part_grid_transf_1(w2, w, Nw_lims, ws, coeffs1, test_A4);
	 
	 graph_2D g1, g2, g3;
	 char xl[]="$\\\\omega$";
	 char yl[]="A";
	 char attr1[]="'v-',color='b', markeredgecolor='b', markerfacecolor='none'";
	 char attr2[]="'^-',color='r', markeredgecolor='r', markerfacecolor='none'";
	 char attr3[]="'s-',color='m', markeredgecolor='m', markerfacecolor='none'";
	 double xlims[2], ylims[2];
	 xlims[0]=w2(0);
	 xlims[1]=w2(w2.n_rows-1);
	 vec test_A_all=join_vert(test_A3,test_A4);
	 ylims[0]=test_A_all.min();
	 ylims[1]=1.1*test_A2.max();
	 
	 g1.add_data(w2.memptr(),test_A2.memptr(),w2.n_rows);
	 g1.add_attribute(attr1);
	 g1.add_data(w2.memptr(),test_A3.memptr(),w2.n_rows);
	 g1.add_attribute(attr2);
	 g1.add_data(w2.memptr(),test_A4.memptr(),w2.n_rows);
	 g1.add_attribute(attr3);
	 g1.set_axes_labels(xl,yl);
	 g1.set_axes_lims(xlims,ylims);
	 g1.curve_plot();
	 
	 char yl2[]="Delta A";
	 //vec Delta_A=(test_A2-test_A3)/test_A2;
	 vec Delta_A=(test_A2-test_A3);
	 ylims[0]=Delta_A.min();
	 ylims[1]=Delta_A.max();
	 g2.add_data(w2.memptr(),Delta_A.memptr(),w2.n_rows);
	 g2.set_axes_labels(xl,yl2);
	 g2.set_axes_lims(xlims,ylims);
	 g2.curve_plot();
	 
	 char attr[]="color='r'";
	 vec Delta_A1=(test_A3-test_A4);
	 ylims[0]=Delta_A1.min();
	 ylims[1]=Delta_A1.max();
	 g3.add_data(w2.memptr(),Delta_A1.memptr(),w2.n_rows);
	 g3.add_attribute(attr);
	 g3.set_axes_labels(xl,yl2);
	 g3.set_axes_lims(xlims,ylims);
	 g3.curve_plot();
	 
	 graph_2D::show_figures();
	*/
	
	
	int Nint=Nw-1;
	int NCfs=4*Nint;
	
	int Nintg=Nug;
	int NCg=4*Nintg;
	
	mat Pa_g=zeros<mat>(Nintg,NCfs);
	mat Pb_g_r=zeros<mat>(Nintg,NCfs);
	mat Pc_g_r=zeros<mat>(Nintg,NCfs);
	mat Pd_g_r=zeros<mat>(Nintg,NCfs);
	
	int j;
	for (j=0; j<Nintg; j++)
	{
		Pa_g(j,4*j)=1;
		Pb_g_r(j,4*j+1)=1;
		Pc_g_r(j,4*j+2)=1;
		Pd_g_r(j,4*j+3)=1;
	}
	
	vec vDg=((w.rows(1,Nw_lims(0))-w0l)%(w.rows(0,Nw_lims(0)-1)-w0l))/(w.rows(0,Nw_lims(0)-1)-w.rows(1,Nw_lims(0)));
	mat Dg=diagmat(vDg);
	vec vDU=(w.rows(1,Nw_lims(0))-w0l)/(w.rows(0,Nw_lims(0)-1)-w.rows(1,Nw_lims(0)));
	mat DU=diagmat(vDU);
	
	mat Pb_g=Pb_g_r-3*DU*Pa_g;
	mat Pc_g=Pc_g_r+3*pow(DU,2)*Pa_g-2*DU*Pb_g_r;
	mat Pd_g=Pd_g_r-pow(DU,3)*Pa_g+pow(DU,2)*Pb_g_r-DU*Pc_g_r;
	
	Pa_g=pow(Dg,3)*Pa_g;
	Pb_g=pow(Dg,2)*Pb_g;
	Pc_g=Dg*Pc_g;
	
	int	Nintc=Nwc-1;
	
	mat	Pa_c=zeros<mat>(Nintc,NCfs);
	mat	Pb_c=zeros<mat>(Nintc,NCfs);
	mat	Pc_c=zeros<mat>(Nintc,NCfs);
	mat	Pd_c=zeros<mat>(Nintc,NCfs);
	
	for (j=0; j<Nintc; j++)
	{
		Pa_c(j,4*j+NCg)=1;
		Pb_c(j,4*j+1+NCg)=1;
		Pc_c(j,4*j+2+NCg)=1;
		Pd_c(j,4*j+3+NCg)=1;
	}
	
	vec vDc=1.0/(w.rows(Nw_lims(0)+1,Nw_lims(1))-w.rows(Nw_lims(0),Nw_lims(1)-1));
	mat Dc=diagmat(vDc);
	
	Pa_c=pow(Dc,3)*Pa_c;
	Pb_c=pow(Dc,2)*Pb_c;
	Pc_c=Dc*Pc_c;
	
	int NCgc=NCg+4*Nintc;
	
	int Nintd=Nud;
	
	mat Pa_d=zeros<mat>(Nintd,NCfs);
	mat Pb_d_r=zeros<mat>(Nintd,NCfs);
	mat Pc_d_r=zeros<mat>(Nintd,NCfs);
	mat Pd_d_r=zeros<mat>(Nintd,NCfs);
	
	for (j=0; j<Nintd; j++)
	{
		Pa_d(j,4*j+NCgc)=1;
		Pb_d_r(j,4*j+1+NCgc)=1;
		Pc_d_r(j,4*j+2+NCgc)=1;
		Pd_d_r(j,4*j+3+NCgc)=1;
	}
	
	vec vDd=((w.rows(Nw_lims(1)+1,Nw-1)-w0r)%(w.rows(Nw_lims(1),Nw-2)-w0r))/(w.rows(Nw_lims(1),Nw-2)-w.rows(Nw_lims(1)+1,Nw-1));
	mat Dd=diagmat(vDd);
	vDU=(w.rows(Nw_lims(1)+1,Nw-1)-w0r)/(w.rows(Nw_lims(1),Nw-2)-w.rows(Nw_lims(1)+1,Nw-1));
	DU=diagmat(vDU);
	
	mat Pb_d=Pb_d_r-3*DU*Pa_d;
	mat Pc_d=Pc_d_r+3*pow(DU,2)*Pa_d-2*DU*Pb_d_r;
	mat Pd_d=Pd_d_r-pow(DU,3)*Pa_d+pow(DU,2)*Pb_d_r-DU*Pc_d_r;
	
	Pa_d=pow(Dd,3)*Pa_d;
	Pb_d=pow(Dd,2)*Pb_d;
	Pc_d=Dd*Pc_d;
	
	cx_mat Ka_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kb_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kc_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kd_g=zeros<cx_mat>(Nn,Nintg);
	
	rowvec ug2(Nug+1);
	ug2.cols(0,Nug-1)=ug.t();
	ug2(Nug)=1.0/(wl-w0l);
	
	mat Wng=wn*ones<rowvec>(Nintg);
	mat Ug=ones<vec>(Nn)*ug2;
	
	dcomplex i(0,1);
	
	mat atang=atan((Wng % (Ug.cols(1,Nintg)-Ug.cols(0,Nintg-1)))/(1+w0l*(Ug.cols(1,Nintg)+Ug.cols(0,Nintg-1))+(pow(w0l,2)+pow(Wng,2)) % Ug.cols(0,Nintg-1) % Ug.cols(1,Nintg)));
	mat logg=log((1.0+2*w0l*Ug.cols(1,Nintg)+pow(Ug.cols(1,Nintg),2) % (pow(Wng,2)+pow(w0l,2)))/(1.0+2*w0l*Ug.cols(0,Nintg-1)+pow(Ug.cols(0,Nintg-1),2) % (pow(Wng,2)+pow(w0l,2))));
	
	Ka_g=-(Ug.cols(1,Nintg)-Ug.cols(0,Nintg-1))/pow(Wng+i*w0l,2)-i*(pow(Ug.cols(1,Nintg),2)-pow(Ug.cols(0,Nintg-1),2))/(2*(Wng+i*w0l))+atang/pow(Wng+i*w0l,3)+i*logg/(2*pow(Wng+i*w0l,3));
	
	Kb_g=-i*(Ug.cols(1,Nintg)-Ug.cols(0,Nintg-1))/(Wng+i*w0l)+i*atang/pow(Wng+i*w0l,2)-logg/(2*pow(Wng+i*w0l,2));
	
	Kc_g=-atang/(Wng+i*w0l)-i*logg/(2*(Wng+i*w0l));
	
	Kd_g=-i*atang+logg/2-log(Ug.cols(1,Nintg)/Ug.cols(0,Nintg-1));
	
	rowvec wg=trans(w.rows(0,Nug));
	//rowvec wg=1/ug2+w0l;
	mat Wg=ones<vec>(Nn)*wg;
	
	if (use_HF_exp)
	{
		double utmp;
		int jn, p;
		for (j=0; j<Nug; j++)
		{
			utmp=ug2(j);
			jn=0;
			while (jn<Nn && abs(wn(jn)*utmp)<fg*abs(1+utmp*w0l)) jn++;
			while (jn<Nn && pow(wn(jn),pngmax)==0) jn++;
			if (jn<Nn)
			{
				Ka_g.submat(jn,j,Nn-1,j)=-i*(pow(Ug.submat(jn,j+1,Nn-1,j+1),2)-pow(Ug.submat(jn,j,Nn-1,j),2))/(2*(wn.rows(jn,Nn-1)+i*w0l))-(Ug.submat(jn,j+1,Nn-1,j+1)-Ug.submat(jn,j,Nn-1,j))/pow(wn.rows(jn,Nn-1)+i*w0l,2)+log(Ug.submat(jn,j+1,Nn-1,j+1)/Ug.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0l,3);
				Kb_g.submat(jn,j,Nn-1,j)=-i*(Ug.submat(jn,j+1,Nn-1,j+1)-Ug.submat(jn,j,Nn-1,j))/(wn.rows(jn,Nn-1)+i*w0l)+log(Ug.submat(jn,j+1,Nn-1,j+1)/Ug.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0l,2);
				Kc_g.submat(jn,j,Nn-1,j)=log(Ug.submat(jn,j+1,Nn-1,j+1)/Ug.submat(jn,j,Nn-1,j))/(i*wn.rows(jn,Nn-1)-w0l);
				Kd_g.submat(jn,j,Nn-1,j)=zeros<cx_vec>(Nn-jn);
				for (p=pngmax; p>=1; p--)
				{
					Ka_g.submat(jn,j,Nn-1,j)=Ka_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0l,3);
					Kb_g.submat(jn,j,Nn-1,j)=Kb_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0l,2);
					Kc_g.submat(jn,j,Nn-1,j)=Kc_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/(i*wn.rows(jn,Nn-1)-w0l);
					Kd_g.submat(jn,j,Nn-1,j)=Kd_g.submat(jn,j,Nn-1,j)+pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p));
				}
			}
		}
	}
	
	Ka_g=-Ka_g/(2*PI);
	Kb_g=-Kb_g/(2*PI);
	Kc_g=-Kc_g/(2*PI);
	Kd_g=-Kd_g/(2*PI);
	
	
	cx_mat Ka_c=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kb_c=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kc_c=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kd_c=zeros<cx_mat>(Nn,Nintc);
	
	mat Wnc=wn*ones<rowvec>(Nintc);
	mat Wc=ones<vec>(Nn)*wc.t();
	
	mat logc=log((pow(Wnc,2)+pow(Wc.cols(1,Nintc),2))/(pow(Wnc,2)+pow(Wc.cols(0,Nintc-1),2)));
	mat atanc2=atan((Wnc % (Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1)))/(Wc.cols(1,Nintc) % Wc.cols(0,Nintc-1)+pow(Wnc,2)));
	cx_mat logc2=logc/2+i*atanc2;
	cx_mat dWn=i*Wnc-Wc.cols(0,Nintc-1);
	mat dWc=Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1);
	
	Ka_c=-pow(dWn,2) % dWc-dWn % pow(dWc,2)/2-pow(dWc,3)/3-pow(dWn,3) % logc2;
	Kb_c=-dWn % dWc-pow(dWc,2)/2-pow(dWn,2) % logc2;
	Kc_c=-dWc-dWn % logc2;
	Kd_c=-logc2;
	
	int Pmax=2*pnmax+4;
	imat MP;
	pascal(Pmax+1,MP);
	
	if (use_HF_exp)
	{
		double wtmp;
		int jni, jnr, p, l;
		for (j=0; j<Nwc-1; j++)
		{
			wtmp=abs(wc(j));
			if (abs(wc(j+1))>wtmp) wtmp=abs(wc(j+1));
			jni=0;
			while (jni<Nn && wn(jni)<fi*wtmp) jni++;
			while (jni<Nn && pow(wn(jni),2*pnmax+1)==0) jni++;
			jnr=1;
			while (jnr<Nn && wn(jnr)<fr*wtmp) jnr++;
			while (jnr<Nn && pow(wn(jnr),2*pnmax)==0) jnr++;
			if (jni<Nn || jnr<Nn)
			{
				double wj=wc(j);
				double dw1=wc(j+1)-wj;
				vec dwp=zeros<vec>(Pmax);
				dwp(0)=dw1;
				dwp(1)=pow(dw1,2)+2*wj*dw1;
				for (p=3; p<=Pmax; p++)
				{
					dwp(p-1)=0;
					for (l=0; l<p; l++)
					{
						dwp(p-1)=dwp(p-1)+MP(l,p-l)*pow(dw1,p-l)*pow(wj,l);
					}
				}
				cx_vec vtmp;
				if (jni<Nn)
				{
					vtmp.zeros(Nn-jni);
					vtmp.set_real(real(Ka_c.submat(jni,j,Nn-1,j)));
					Ka_c.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kb_c.submat(jni,j,Nn-1,j)));
					Kb_c.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kc_c.submat(jni,j,Nn-1,j)));
					Kc_c.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kd_c.submat(jni,j,Nn-1,j)));
					Kd_c.submat(jni,j,Nn-1,j)=vtmp;
					
					for (p=2*pnmax+1; p>=1; p-=2)
					{
						Ka_c.submat(jni,j,Nn-1,j)=Ka_c.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(pow(wj,3)*dwp(p-1)/p - 3.0*pow(wj,2)*dwp(p)/(p+1) + 3.0*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jni,Nn-1),p);
						Kb_c.submat(jni,j,Nn-1,j)=Kb_c.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jni,Nn-1),p);
						Kc_c.submat(jni,j,Nn-1,j)=Kc_c.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jni,Nn-1),p);
						Kd_c.submat(jni,j,Nn-1,j)=Kd_c.submat(jni,j,Nn-1,j) - i*pow(-1,(p-1)/2)*dwp(p-1)/(p*pow(wn.rows(jni,Nn-1),p));
					}
				}
				if (jnr<Nn)
				{
					vtmp.zeros(Nn-jnr);
					vtmp.set_imag(imag(Ka_c.submat(jnr,j,Nn-1,j)));
					Ka_c.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kb_c.submat(jnr,j,Nn-1,j)));
					Kb_c.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kc_c.submat(jnr,j,Nn-1,j)));
					Kc_c.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kd_c.submat(jnr,j,Nn-1,j)));
					Kd_c.submat(jnr,j,Nn-1,j)=vtmp;
					for (p=2*pnmax; p>=2; p-=2)
					{
						Ka_c.submat(jnr,j,Nn-1,j)=Ka_c.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(pow(wj,3)*dwp(p-1)/p - 3*pow(wj,2)*dwp(p)/(p+1) + 3*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jnr,Nn-1),p);
						Kb_c.submat(jnr,j,Nn-1,j)=Kb_c.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jnr,Nn-1),p);
						Kc_c.submat(jnr,j,Nn-1,j)=Kc_c.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jnr,Nn-1),p);
						Kd_c.submat(jnr,j,Nn-1,j)=Kd_c.submat(jnr,j,Nn-1,j) - pow(-1,(p-2)/2)*dwp(p-1)/(p*pow(wn.rows(jnr,Nn-1),p));
					}
				}
			}
		}
	}
	
	Ka_c=Ka_c/(2*PI);
	Kb_c=Kb_c/(2*PI);
	Kc_c=Kc_c/(2*PI);
	Kd_c=Kd_c/(2*PI);
	
	cx_mat Ka_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kb_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kc_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kd_d=zeros<cx_mat>(Nn,Nintd);
	
	rowvec ud2(Nud+1);
	ud2.cols(1,Nud)=ud.t();
	ud2(0)=1.0/(wr-w0r);
	
	mat Wnd=wn*ones<rowvec>(Nintd);
	mat Ud=ones<vec>(Nn)*ud2;
	
	mat atand=atan((Wnd % (Ud.cols(1,Nintd)-Ud.cols(0,Nintd-1)))/(1.0+w0r*(Ud.cols(1,Nintd)+Ud.cols(0,Nintd-1))+(pow(w0r,2)+pow(Wnd,2)) % Ud.cols(0,Nintd-1) % Ud.cols(1,Nintd)));
	mat logd=log((1+2*w0r*Ud.cols(1,Nintd)+pow(Ud.cols(1,Nintd),2) % (pow(Wnd,2)+pow(w0r,2)))/(1+2*w0r*Ud.cols(0,Nintd-1)+pow(Ud.cols(0,Nintd-1),2) % (pow(Wnd,2)+pow(w0r,2))));
	
	Ka_d=-(Ud.cols(1,Nintd)-Ud.cols(0,Nintd-1))/pow(Wnd+i*w0r,2) - i*(pow(Ud.cols(1,Nintd),2)-pow(Ud.cols(0,Nintd-1),2))/(2*(Wnd+i*w0r))+atand/pow(Wnd+i*w0r,3) +i*logd/(2*pow(Wnd+i*w0r,3));
	
	Kb_d=-i*(Ud.cols(1,Nintd)-Ud.cols(0,Nintd-1))/(Wnd+i*w0r)+i*atand/pow(Wnd+i*w0r,2)-logd/(2*pow(Wnd+i*w0r,2));
	
	Kc_d=-atand/(Wnd+i*w0r)-i*logd/(2*(Wnd+i*w0r));
	
	Kd_d=-i*atand-log(Ud.cols(1,Nintd)/Ud.cols(0,Nintd-1))+logd/2;
	
	//rowvec wd=1/ud2+w0r;
	rowvec wd=trans(w.rows(Nw_lims(1),Nw-1));
	mat Wd=ones<vec>(Nn)*wd;
	
	if (use_HF_exp)
	{
		double utmp;
		int jn, p;
		for (j=0; j<Nud; j++)
		{
			utmp=ud2(j);
			jn=0;
			while (jn<Nn && abs(wn(jn)*utmp)<fd*abs(1+utmp*w0r)) jn++;
			while (jn<Nn && pow(wn(jn),pndmax)) jn++;
			if (jn<Nn)
			{
				Ka_d.submat(jn,j,Nn-1,j)=-(Ud.submat(jn,j+1,Nn-1,j+1)-Ud.submat(jn,j,Nn-1,j))/pow(wn.rows(jn,Nn-1)+i*w0r,2) - i*(pow(Ud.submat(jn,j+1,Nn-1,j+1),2)-pow(Ud.submat(jn,j,Nn-1,j),2))/(2*(wn.rows(jn,Nn-1)+i*w0r))+log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0r,3);
				Kb_d.submat(jn,j,Nn-1,j)=-i*(Ud.submat(jn,j+1,Nn-1,j+1)-Ud.submat(jn,j,Nn-1,j))/(wn.rows(jn,Nn-1)+i*w0r)+log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0r,2);
				Kc_d.submat(jn,j,Nn-1,j)=log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/(i*wn.rows(jn,Nn-1)-w0r);
				Kd_d.submat(jn,j,Nn-1,j)=zeros<cx_vec>(Nn-jn);
				for (p=pndmax; p>=1; p--)
				{
					Ka_d.submat(jn,j,Nn-1,j)=Ka_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0r,3);
					Kb_d.submat(jn,j,Nn-1,j)=Kb_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0r,2);
					Kc_d.submat(jn,j,Nn-1,j)=Kc_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/(i*wn.rows(jn,Nn-1)-w0r);
					Kd_d.submat(jn,j,Nn-1,j)=Kd_d.submat(jn,j,Nn-1,j)+pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p));
				}
			}
		}
	}
	
	Ka_d=-Ka_d/(2*PI);
	Kb_d=-Kb_d/(2*PI);
	Kc_d=-Kc_d/(2*PI);
	Kd_d=-Kd_d/(2*PI);
	
	cx_mat KG=(Ka_g*Pa_g+Kb_g*Pb_g+Kc_g*Pc_g+Kd_g*Pd_g)*MM;
	cx_mat KC=(Ka_c*Pa_c+Kb_c*Pb_c+Kc_c*Pc_c+Kd_c*Pd_c)*MM;
	cx_mat KD=(Ka_d*Pa_d+Kb_d*Pb_d+Kc_d*Pc_d+Kd_d*Pd_d)*MM;
	
	Kcx=KG+KC+KD;
	
	/*
	 //test the kernel matrix
	 rowvec x0={-2, 0, 3};
	 rowvec s0={0.8, 0.3, 1};
	 rowvec wgt={1,1,1};
	 
	 vec test_A;
	 sum_gaussians(w, x0, s0, wgt, test_A);
	 
	 uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	 
	 vec Gtest1=K*test_A;
	 vec Gr_test1=Gtest1.rows(even_ind);
	 vec Gi_test1=Gtest1.rows(even_ind+1);
	 cx_vec Gtest=Kcx*test_A;
	 vec Gr_test=real(Gtest);
	 vec Gi_test=imag(Gtest);
	 
	 graph_2D g1, g2, g3, g4, g5, g6;
	 char attr1[]="'o',color='b',markeredgecolor='b',markerfacecolor='none'";
	 char attr2[]="'s',color='r',markeredgecolor='r',markerfacecolor='none'";
	 char attr3[]="'.-',color='b'";
	 char attr4[]="'.-',color='m'";
	 char attr5[]="'.-',color='r'";
	 char attr6[]="'.-',color='c'";
	 
	 g1.add_data(wn.memptr(), Gr.memptr(), Nn);
	 g1.add_attribute(attr1);
	 g1.add_data(wn.memptr(), Gr_test.memptr(), Nn);
	 g1.add_attribute(attr2);
	 g1.curve_plot();
	 
	 g2.add_data(wn.memptr(), Gi.memptr(), Nn);
	 g2.add_attribute(attr1);
	 g2.add_data(wn.memptr(), Gi_test.memptr(), Nn);
	 g2.add_attribute(attr2);
	 g2.curve_plot();
	 
	 vec DGr1=(Gr_test-Gr)/Gr, DGr2=(Gr_test1-Gr)/Gr, DGr=(Gr_test-Gr_test1);
	 vec DGi1=(Gi_test-Gi)/Gi, DGi2=(Gi_test1-Gi)/Gi, DGi=(Gi_test-Gi_test1);
	 g3.add_data(wn.memptr(),DGr1.memptr(),Nn);
	 g3.add_attribute(attr3);
	 g3.add_data(wn.memptr(),DGr2.memptr(),Nn);
	 g3.add_attribute(attr4);
	 g3.curve_plot();
	 
	 g4.add_data(wn.memptr(),DGr.memptr(),Nn);
	 g4.add_attribute(attr4);
	 g4.curve_plot();
	 
	 g5.add_data(wn.memptr(),DGi1.memptr(),Nn);
	 g5.add_attribute(attr5);
	 g5.add_data(wn.memptr(),DGi2.memptr(),Nn);
	 g5.add_attribute(attr6);
	 g5.curve_plot();
	 
	 g6.add_data(wn.memptr(),DGi.memptr(),Nn);
	 g6.add_attribute(attr6);
	 g6.curve_plot();
	 
	 graph_2D::show_figures();
	*/
	
	
	rowvec Knorm_a_g=zeros<rowvec>(Nintg);
	rowvec Knorm_b_g=zeros<rowvec>(Nintg);
	rowvec Knorm_c_g=zeros<rowvec>(Nintg);
	rowvec Knorm_d_g=zeros<rowvec>(Nintg);
	
	Knorm_a_g=-(pow(ug2.cols(1,Nug),2)-pow(ug2.cols(0,Nug-1),2))/2;
	Knorm_b_g=-(ug2.cols(1,Nug)-ug2.cols(0,Nug-1));
	Knorm_c_g=-log(ug2.cols(1,Nug)/ug2.cols(0,Nug-1));
	Knorm_d_g=1.0/ug2.cols(1,Nug)-1.0/ug2.cols(0,Nug-1);
	
	rowvec KM0g=(Knorm_a_g*Pa_g+Knorm_b_g*Pb_g+Knorm_c_g*Pc_g+Knorm_d_g*Pd_g)*MM/(2*PI);
	
	rowvec Knorm_a_c=zeros<rowvec>(Nintc);
	rowvec Knorm_b_c=zeros<rowvec>(Nintc);
	rowvec Knorm_c_c=zeros<rowvec>(Nintc);
	rowvec Knorm_d_c=zeros<rowvec>(Nintc);
	
	Knorm_a_c=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),4)/4);
	Knorm_b_c=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),3)/3);
	Knorm_c_c=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),2)/2);
	Knorm_d_c=trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2));
	
	rowvec KM0c=(Knorm_a_c*Pa_c+Knorm_b_c*Pb_c+Knorm_c_c*Pc_c+Knorm_d_c*Pd_c)*MM/(2*PI);
	
	rowvec Knorm_a_d=zeros<rowvec>(Nintd);
	rowvec Knorm_b_d=zeros<rowvec>(Nintd);
	rowvec Knorm_c_d=zeros<rowvec>(Nintd);
	rowvec Knorm_d_d=zeros<rowvec>(Nintd);
	
	Knorm_a_d=-(pow(ud2.cols(1,Nud),2)-pow(ud2.cols(0,Nud-1),2))/2;
	Knorm_b_d=-(ud2.cols(1,Nud)-ud2.cols(0,Nud-1));
	Knorm_c_d=-log(ud2.cols(1,Nud)/ud2.cols(0,Nud-1));
	Knorm_d_d=1.0/ud2.cols(1,Nud)-1.0/ud2.cols(0,Nud-1);
	
	rowvec KM0d=(Knorm_a_d*Pa_d+Knorm_b_d*Pb_d+Knorm_c_d*Pc_d+Knorm_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM0=KM0g+KM0c+KM0d;
	
	
	rowvec KM1_a_g=zeros<rowvec>(Nintg);
	rowvec KM1_b_g=zeros<rowvec>(Nintg);
	rowvec KM1_c_g=zeros<rowvec>(Nintg);
	rowvec KM1_d_g=zeros<rowvec>(Nintg);
	
	KM1_a_g=Knorm_b_g;
	KM1_b_g=Knorm_c_g;
	KM1_c_g=Knorm_d_g;
	KM1_d_g=(1.0/pow(ug2.cols(1,Nug),2)-1.0/pow(ug2.cols(0,Nug-1),2))/2;
	
	rowvec KM1g_tmp=(KM1_a_g*Pa_g+KM1_b_g*Pb_g+KM1_c_g*Pc_g+KM1_d_g*Pd_g)*MM/(2*PI);
	
	rowvec KM1g=w0l*KM0g+KM1g_tmp;
	
	mat Wjc=diagmat(wc.rows(0,Nintc-1));
	
	rowvec KM1_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),5)/5.0);
	
	rowvec KM1_a_c=KM1_a_c_tmp+Knorm_a_c*Wjc;
	rowvec KM1_b_c=Knorm_a_c+Knorm_b_c*Wjc;
	rowvec KM1_c_c=Knorm_b_c+Knorm_c_c*Wjc;
	rowvec KM1_d_c=Knorm_c_c+Knorm_d_c*Wjc;
	
	rowvec KM1c=(KM1_a_c*Pa_c+KM1_b_c*Pb_c+KM1_c_c*Pc_c+KM1_d_c*Pd_c)*MM/(2*PI);
	
	rowvec KM1_a_d=zeros<rowvec>(Nintd);
	rowvec KM1_b_d=zeros<rowvec>(Nintd);
	rowvec KM1_c_d=zeros<rowvec>(Nintd);
	rowvec KM1_d_d=zeros<rowvec>(Nintd);
	
	KM1_a_d=Knorm_b_d;
	KM1_b_d=Knorm_c_d;
	KM1_c_d=Knorm_d_d;
	KM1_d_d=(1.0/pow(ud2.cols(1,Nud),2)-1.0/pow(ud2.cols(0,Nud-1),2))/2;
	
	rowvec KM1d_tmp=(KM1_a_d*Pa_d+KM1_b_d*Pb_d+KM1_c_d*Pc_d+KM1_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM1d=w0r*KM0d+KM1d_tmp;
	
	rowvec KM1=KM1g+KM1c+KM1d;
	
	rowvec KM2_a_g=zeros<rowvec>(Nintg);
	rowvec KM2_b_g=zeros<rowvec>(Nintg);
	rowvec KM2_c_g=zeros<rowvec>(Nintg);
	rowvec KM2_d_g=zeros<rowvec>(Nintg);
	
	KM2_a_g=Knorm_c_g;
	KM2_b_g=Knorm_d_g;
	KM2_c_g=KM1_d_g;
	KM2_d_g=(1.0/pow(ug2.cols(1,Nug),3)-1.0/pow(ug2.cols(0,Nug-1),3))/3;
	
	rowvec KM2g_tmp=(KM2_a_g*Pa_g+KM2_b_g*Pb_g+KM2_c_g*Pc_g+KM2_d_g*Pd_g)*MM/(2*PI);
	
	rowvec KM2g=pow(w0l,2)*KM0g+2*w0l*KM1g_tmp+KM2g_tmp;
	
	rowvec KM2_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),6)/6);
	
	rowvec KM2_a_c=KM2_a_c_tmp+2*KM1_a_c_tmp*Wjc+Knorm_a_c*pow(Wjc,2);
	rowvec KM2_b_c=KM1_a_c_tmp+2*Knorm_a_c*Wjc+Knorm_b_c*pow(Wjc,2);
	rowvec KM2_c_c=Knorm_a_c+2*Knorm_b_c*Wjc+Knorm_c_c*pow(Wjc,2);
	rowvec KM2_d_c=Knorm_b_c+2*Knorm_c_c*Wjc+Knorm_d_c*pow(Wjc,2);
	
	rowvec KM2c=(KM2_a_c*Pa_c+KM2_b_c*Pb_c+KM2_c_c*Pc_c+KM2_d_c*Pd_c)*MM/(2*PI);
	
	rowvec KM2_a_d=zeros<rowvec>(Nintd);
	rowvec KM2_b_d=zeros<rowvec>(Nintd);
	rowvec KM2_c_d=zeros<rowvec>(Nintd);
	rowvec KM2_d_d=zeros<rowvec>(Nintd);
	
	KM2_a_d=Knorm_c_d;
	KM2_b_d=Knorm_d_d;
	KM2_c_d=KM1_d_d;
	KM2_d_d=(1.0/pow(ud2.cols(1,Nud),3)-1.0/pow(ud2.cols(0,Nud-1),3))/3;
	
	rowvec KM2d_tmp=(KM2_a_d*Pa_d+KM2_b_d*Pb_d+KM2_c_d*Pc_d+KM2_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM2d=pow(w0r,2)*KM0d+2*w0r*KM1d_tmp+KM2d_tmp;
	
	rowvec KM2=KM2g+KM2c+KM2d;
	
	rowvec KM3_a_g=zeros<rowvec>(Nintg);
	rowvec KM3_b_g=zeros<rowvec>(Nintg);
	rowvec KM3_c_g=zeros<rowvec>(Nintg);
	rowvec KM3_d_g=zeros<rowvec>(Nintg);
	
	KM3_a_g=Knorm_d_g;
	KM3_b_g=KM1_d_g;
	KM3_c_g=KM2_d_g;
	KM3_d_g=(1.0/pow(ug2.cols(1,Nug),4)-1.0/pow(ug2.cols(0,Nug-1),4))/4;
	
	rowvec KM3g_tmp=(KM3_a_g*Pa_g+KM3_b_g*Pb_g+KM3_c_g*Pc_g+KM3_d_g*Pd_g)*MM/(2*PI);
	
	rowvec KM3g=pow(w0l,3)*KM0g+3*pow(w0l,2)*KM1g_tmp+3*w0l*KM2g_tmp+KM3g_tmp;
	
	rowvec KM3_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),7)/7);
	
	rowvec KM3_a_c=KM3_a_c_tmp+3*KM2_a_c_tmp*Wjc+3*KM1_a_c_tmp*pow(Wjc,2)+Knorm_a_c*pow(Wjc,3);
	rowvec KM3_b_c=KM2_a_c_tmp+3*KM1_a_c_tmp*Wjc+3*Knorm_a_c*pow(Wjc,2)+Knorm_b_c*pow(Wjc,3);
	rowvec KM3_c_c=KM1_a_c_tmp+3*Knorm_a_c*Wjc+3*Knorm_b_c*pow(Wjc,2)+Knorm_c_c*pow(Wjc,3);
	rowvec KM3_d_c=Knorm_a_c+3*Knorm_b_c*Wjc+3*Knorm_c_c*pow(Wjc,2)+Knorm_d_c*pow(Wjc,3);
	
	rowvec KM3c=(KM3_a_c*Pa_c+KM3_b_c*Pb_c+KM3_c_c*Pc_c+KM3_d_c*Pd_c)*MM/(2*PI);
	
	rowvec KM3_a_d=zeros<rowvec>(Nintd);
	rowvec KM3_b_d=zeros<rowvec>(Nintd);
	rowvec KM3_c_d=zeros<rowvec>(Nintd);
	rowvec KM3_d_d=zeros<rowvec>(Nintd);
	
	KM3_a_d=Knorm_d_d;
	KM3_b_d=KM1_d_d;
	KM3_c_d=KM2_d_d;
	KM3_d_d=(1.0/pow(ud2.cols(1,Nud),4)-1.0/pow(ud2.cols(0,Nud-1),4))/4;
	
	rowvec KM3d_tmp=(KM3_a_d*Pa_d+KM3_b_d*Pb_d+KM3_c_d*Pc_d+KM3_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM3d=pow(w0r,3)*KM0d+3*pow(w0r,2)*KM1d_tmp+3*w0r*KM2d_tmp+KM3d_tmp;
	
	rowvec KM3=KM3g+KM3c+KM3d;
	
	KM.zeros(4,Nw);
	
	KM.row(0)=KM0;
	KM.row(1)=KM1;
	KM.row(2)=KM2;
	KM.row(3)=KM3;
	
	K.zeros(2*Nn,Nw);
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	K.rows(even_ind)=real(Kcx);
	K.rows(even_ind+1)=imag(Kcx);
	
	/*
	 //check moments
	 rowvec x0={-2, 0, 3};
	 rowvec s0={0.8, 0.3, 1};
	 rowvec wgt={1,1,1};
	 
	 vec test_A;
	 sum_gaussians(w, x0, s0, wgt, test_A);
	 
 	cout<<KM*test_A<<endl;
	*/
	
	return true;
}

bool OmegaMaxEnt_data::Kernel_G_fermions_Riemann_integ()
{
//	cout<<"defining kernel matrix...\n";
	
	int i,j;
	
	Kcx.zeros(Nn,Nw);
	
	int jl=Nw_lims(0);
	int jr=Nw_lims(1);
	
	vec Dwc=0.5*(w.rows(jl+1,jr+1)-w.rows(jl-1,jr-1));
	
	cx_mat KC(Nn,Nwc);
	
	for (i=0; i<Nn; i++)
	{
		for (j=0; j<Nwc; j++)
		{
			KC(i,j)=Dwc(j)/(2*PI*(dcomplex(0,wn(i))-wc(j)));
		}
	}
	
	int Nug=Nw_lims(0);
	int Nud=Nw-Nw_lims(1)-1;
	
	double dug,dud;
	
	vec ug, ud;
	if (Du_constant)
	{
		dug=dwl/((wl-dwl-w0l)*(wl-w0l));
		vec ug_int=linspace<vec>(1,Nug,Nug);
		ug=-dug*ug_int;
		vec dul=dug*ones<vec>(Nug);
		
		dud=dwr/((wr-w0r)*(wr+dwr-w0r));
		vec ud_int=linspace<vec>(Nud,1,Nud);
		ud=dud*ud_int;
		vec dur=dud*ones<vec>(Nud);
	}
	else
	{
		cout<<"Kernel_G_fermions_Riemann_integ(): this kernel can be defined only for uniform u=1/(w-w0)\n";
		return false;
//		ug=1.0/(w.rows(0,Nug-1)-w0l);
//		ud=1.0/(w.rows(Nw_lims(1)+1,Nw-1)-w0r);
	}
	
	cx_mat KG(Nn,Nug);
	cx_mat KD(Nn,Nud);
	
	for (i=0; i<Nn; i++)
	{
		for (j=0; j<Nug; j++)
		{
			KG(i,j)=dug/(2*PI*((dcomplex(0,wn(i))-w0l)*ug(j)*ug(j)-ug(j)));
		}
		for (j=0; j<Nud; j++)
		{
			KD(i,j)=dud/(2*PI*((dcomplex(0,wn(i))-w0r)*ud(j)*ud(j)-ud(j)));
		}
	}
	
	Kcx=join_rows(KG,KC);
	Kcx=join_rows(Kcx,KD);
	
	K.zeros(2*Nn,Nw);
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	K.rows(even_ind)=real(Kcx);
	K.rows(even_ind+1)=imag(Kcx);
	
	rowvec KM0g(Nug);
	rowvec KM1g(Nug);
	rowvec KM2g(Nug);
	rowvec KM3g(Nug);
	rowvec KM0c(Nwc);
	rowvec KM1c(Nwc);
	rowvec KM2c(Nwc);
	rowvec KM3c(Nwc);
	rowvec KM0d(Nud);
	rowvec KM1d(Nud);
	rowvec KM2d(Nud);
	rowvec KM3d(Nud);
	
	for (i=0; i<Nug; i++)
	{
		KM0g(i)=dug*(w(i)-w0l)*(w(i)-w0l)/(2*PI);
		KM1g(i)=dug*(w(i)-w0l)*(w(i)-w0l)*w(i)/(2*PI);
		KM2g(i)=dug*(w(i)-w0l)*(w(i)-w0l)*w(i)*w(i)/(2*PI);
		KM3g(i)=dug*(w(i)-w0l)*(w(i)-w0l)*w(i)*w(i)*w(i)/(2*PI);
	}
	for (i=0; i<Nwc; i++)
	{
		KM0c(i)=Dwc(i)/(2*PI);
		KM1c(i)=Dwc(i)*wc(i)/(2*PI);
		KM2c(i)=Dwc(i)*wc(i)*wc(i)/(2*PI);
		KM3c(i)=Dwc(i)*wc(i)*wc(i)*wc(i)/(2*PI);
	}
	vec wd=w.rows(Nw_lims(1)+1,Nw-1);
	for (i=0; i<Nud; i++)
	{
		KM0d(i)=dud*(wd(i)-w0r)*(wd(i)-w0r)/(2*PI);
		KM1d(i)=dud*(wd(i)-w0r)*(wd(i)-w0r)*wd(i)/(2*PI);
		KM2d(i)=dud*(wd(i)-w0r)*(wd(i)-w0r)*wd(i)*wd(i)/(2*PI);
		KM3d(i)=dud*(wd(i)-w0r)*(wd(i)-w0r)*wd(i)*wd(i)*wd(i)/(2*PI);
	}
	
	rowvec KM0=join_rows(KM0g,KM0c);
	KM0=join_rows(KM0,KM0d);
	rowvec KM1=join_rows(KM1g,KM1c);
	KM1=join_rows(KM1,KM1d);
	rowvec KM2=join_rows(KM2g,KM2c);
	KM2=join_rows(KM2,KM2d);
	rowvec KM3=join_rows(KM3g,KM3c);
	KM3=join_rows(KM3,KM3d);
	
	KM.zeros(4,Nw);
	
	KM.row(0)=KM0;
	KM.row(1)=KM1;
	KM.row(2)=KM2;
	KM.row(3)=KM3;
	
	
/*
	 //test the kernel matrix and the moments vector
	ifstream file("peaks.dat");
	mat peaks_data;
	
	peaks_data.load(file);
	int Npeaks=peaks_data.n_rows;
	
	rowvec x0, s0, wgt;
	
	x0=trans(peaks_data.col(0));
	s0=trans(peaks_data.col(1));
	wgt=trans(peaks_data.col(2));
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
	cout<<KM*test_A<<endl;
	
	 vec Gtest1=K*test_A;
	 vec Gr_test1=Gtest1.rows(even_ind);
	 vec Gi_test1=Gtest1.rows(even_ind+1);
	 cx_vec Gtest=Kcx*test_A;
	 vec Gr_test=real(Gtest);
	 vec Gi_test=imag(Gtest);
	 
	 graph_2D g1, g2, g3, g4, g5, g6;
	 char attr1[]="'o',color='b',markeredgecolor='b',markerfacecolor='none'";
	 char attr2[]="'s',color='r',markeredgecolor='r',markerfacecolor='none'";
	 char attr3[]="'.-',color='m'";
	 char attr4[]="'.-',color='c'";
	 
	 g1.add_data(wn.memptr(), Gr.memptr(), Nn);
	 g1.add_attribute(attr1);
	 g1.add_data(wn.memptr(), Gr_test.memptr(), Nn);
	 g1.add_attribute(attr2);
	g1.add_title("Gr et Gr_test");
	 g1.curve_plot();
	 
	 g2.add_data(wn.memptr(), Gi.memptr(), Nn);
	 g2.add_attribute(attr1);
	 g2.add_data(wn.memptr(), Gi_test.memptr(), Nn);
	 g2.add_attribute(attr2);
	g2.add_title("Gi et Gi_test");
	 g2.curve_plot();
	 
	 vec DGr1=(Gr_test-Gr)/Gr, DGr=(Gr_test-Gr_test1);
	 vec DGi1=(Gi_test-Gi)/Gi, DGi=(Gi_test-Gi_test1);
	
	g3.add_data(wn.memptr(),DGr1.memptr(),Nn);
	 g3.add_attribute(attr3);
	g3.add_title("DGr1");
	 g3.curve_plot();
	 
	 g4.add_data(wn.memptr(),DGr.memptr(),Nn);
	 g4.add_attribute(attr3);
	g4.add_title("DGr");
	 g4.curve_plot();
	 
	 g5.add_data(wn.memptr(),DGi1.memptr(),Nn);
	 g5.add_attribute(attr4);
	g5.add_title("DGi1");
	 g5.curve_plot();
	 
	 g6.add_data(wn.memptr(),DGi.memptr(),Nn);
	 g6.add_attribute(attr4);
	g6.add_title("DGi");
	 g6.curve_plot();
	 
	 graph_2D::show_figures();
*/
	
//	cout<<"kernel matrix defined.\n";
	
	return true;
}

bool OmegaMaxEnt_data::Kernel_G_fermions_grid_transf()
{
	bool use_HF_exp=true;
	double fg=1.7;
	int pngmax=100;
	double fi=fg;
	double fr=fi;
	int pnmax=50;
	double fd=fg;
	int pndmax=pngmax;
	
	cout<<"defining kernel matrix...\n";
	
	Kcx.zeros(Nn,Nw);
 
	int Nug=Nw_lims(0);
	int Nud=Nw-Nw_lims(1)-1;
	
	vec ug, ud;
	if (Du_constant)
	{
		double dug=dwl/((wl-dwl-w0l)*(wl-w0l));
		vec ug_int=linspace<vec>(1,Nug,Nug);
		ug=-dug*ug_int;
		
		double dud=dwr/((wr-w0r)*(wr+dwr-w0r));
		vec ud_int=linspace<vec>(Nud,1,Nud);
		ud=dud*ud_int;
	}
	else
	{
		ug=1.0/(w.rows(0,Nug-1)-w0l);
		ud=1.0/(w.rows(Nw_lims(1)+1,Nw-1)-w0r);
	}

	mat MM;
	spline_matrix_grid_transf_G_part(w, Nw_lims, ws, MM);

	/*
	//testing the spline matrix
	rowvec x0={wl, 0, wr/2};
	rowvec s0={SW/10, SW/20, SW/5};
	rowvec wgt={1,1,1};
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
	vec coeffs=MM*test_A;
	
	mat MM1;
	spline_matrix_G_part(w, Nw_lims, ws, MM1);
	int Ncfs0=MM1.n_rows;
	vec coeffs0=MM1*test_A;
	vec coeffs1(4*(Nw+1));
	coeffs1(0)=coeffs0(0);
	coeffs1(1)=coeffs0(1);
	coeffs1(2)=0;
	coeffs1(3)=0;
	coeffs1(4*(Nw+1)-4)=coeffs0(Ncfs0-2);
	coeffs1(4*(Nw+1)-3)=coeffs0(Ncfs0-1);
	coeffs1(4*(Nw+1)-2)=0;
	coeffs1(4*(Nw+1)-1)=0;
	uvec ind_tmp=linspace<uvec>(1,Nw-1,Nw-1);
	coeffs1.rows(4*ind_tmp)=coeffs0(3*ind_tmp-1);
	coeffs1.rows(4*ind_tmp+1)=coeffs0(3*ind_tmp);
	coeffs1.rows(4*ind_tmp+2)=coeffs0(3*ind_tmp+1);
	ind_tmp=linspace<uvec>(1,Nw_lims(1),Nw_lims(1));
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp-1);
	ind_tmp=linspace<uvec>(Nw_lims(1)+1,Nw-1,Nw-Nw_lims(1)-1);
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp);
	
	double Nw2=400;
	vec w2=linspace<vec>(wl-SW,wr+SW,Nw2);
	
	vec test_A2;
	sum_gaussians(w2, x0, s0, wgt, test_A2);
	
	vec test_A3, test_A4;
	spline_val_G_part_grid_transf_1(w2, w, Nw_lims, ws, coeffs, test_A3);
	spline_val_G_part(w2, w, Nw_lims, ws, coeffs1, test_A4);
	
	graph_2D g1, g2, g3;
	char xl[]="$\\\\omega$";
	char yl[]="A";
	char attr1[]="'v-',color='b', markeredgecolor='b', markerfacecolor='none'";
	char attr2[]="'^-',color='r', markeredgecolor='r', markerfacecolor='none'";
	char attr3[]="'s-',color='m', markeredgecolor='m', markerfacecolor='none'";
	double xlims[2], ylims[2];
	xlims[0]=w2(0);
	xlims[1]=w2(w2.n_rows-1);
	vec test_A_all=join_vert(test_A3,test_A4);
	ylims[0]=test_A_all.min();
	ylims[1]=1.1*test_A2.max();
	
	g1.add_data(w2.memptr(),test_A2.memptr(),w2.n_rows);
	g1.add_attribute(attr1);
	g1.add_data(w2.memptr(),test_A3.memptr(),w2.n_rows);
	g1.add_attribute(attr2);
	g1.add_data(w2.memptr(),test_A4.memptr(),w2.n_rows);
	g1.add_attribute(attr3);
	g1.set_axes_labels(xl,yl);
	g1.set_axes_lims(xlims,ylims);
	g1.curve_plot();
	
	char yl2[]="Delta A";
	//vec Delta_A=(test_A2-test_A3)/test_A2;
	vec Delta_A=(test_A2-test_A3);
	ylims[0]=Delta_A.min();
	ylims[1]=Delta_A.max();
	g2.add_data(w2.memptr(),Delta_A.memptr(),w2.n_rows);
	g2.set_axes_labels(xl,yl2);
	g2.set_axes_lims(xlims,ylims);
	g2.curve_plot();
	
	char attr[]="color='r'";
	vec Delta_A1=(test_A3-test_A4);
	ylims[0]=Delta_A1.min();
	ylims[1]=Delta_A1.max();
	g3.add_data(w2.memptr(),Delta_A1.memptr(),w2.n_rows);
	g3.add_attribute(attr);
	g3.set_axes_labels(xl,yl2);
	g3.set_axes_lims(xlims,ylims);
	g3.curve_plot();
	
	graph_2D::show_figures();
	*/
	
	int Nint=Nw-1;
	int NCfs=4*Nint;
	
	int Nintg=Nug;
	int NCg=4*Nintg;
	
	mat Pa_g=zeros<mat>(Nintg,NCfs);
	mat Pb_g_r=zeros<mat>(Nintg,NCfs);
	mat Pc_g_r=zeros<mat>(Nintg,NCfs);
	mat Pd_g_r=zeros<mat>(Nintg,NCfs);
	
	int j;
	for (j=0; j<Nintg; j++)
	{
		Pa_g(j,4*j)=1;
		Pb_g_r(j,4*j+1)=1;
		Pc_g_r(j,4*j+2)=1;
		Pd_g_r(j,4*j+3)=1;
	}
	
	vec vDg=((w.rows(1,Nw_lims(0))-w0l)%(w.rows(0,Nw_lims(0)-1)-w0l))/(w.rows(0,Nw_lims(0)-1)-w.rows(1,Nw_lims(0)));
	mat Dg=diagmat(vDg);
	vec vDU=(w.rows(1,Nw_lims(0))-w0l)/(w.rows(0,Nw_lims(0)-1)-w.rows(1,Nw_lims(0)));
	mat DU=diagmat(vDU);
	
	mat Pb_g=Pb_g_r-3*DU*Pa_g;
	mat Pc_g=Pc_g_r+3*pow(DU,2)*Pa_g-2*DU*Pb_g_r;
	mat Pd_g=Pd_g_r-pow(DU,3)*Pa_g+pow(DU,2)*Pb_g_r-DU*Pc_g_r;
	
	Pa_g=pow(Dg,3)*Pa_g;
	Pb_g=pow(Dg,2)*Pb_g;
	Pc_g=Dg*Pc_g;
	
	int	Nintc=Nwc-1;
	
	mat	Pa_c=zeros<mat>(Nintc,NCfs);
	mat	Pb_c=zeros<mat>(Nintc,NCfs);
	mat	Pc_c=zeros<mat>(Nintc,NCfs);
	mat	Pd_c=zeros<mat>(Nintc,NCfs);
	
	for (j=0; j<Nintc; j++)
	{
		Pa_c(j,4*j+NCg)=1;
		Pb_c(j,4*j+1+NCg)=1;
		Pc_c(j,4*j+2+NCg)=1;
		Pd_c(j,4*j+3+NCg)=1;
	}
	
	vec vDc=1.0/(w.rows(Nw_lims(0)+1,Nw_lims(1))-w.rows(Nw_lims(0),Nw_lims(1)-1));
	mat Dc=diagmat(vDc);
	
	Pa_c=pow(Dc,3)*Pa_c;
	Pb_c=pow(Dc,2)*Pb_c;
	Pc_c=Dc*Pc_c;
	
	int NCgc=NCg+4*Nintc;
	
	int Nintd=Nud;
	
	mat Pa_d=zeros<mat>(Nintd,NCfs);
	mat Pb_d_r=zeros<mat>(Nintd,NCfs);
	mat Pc_d_r=zeros<mat>(Nintd,NCfs);
	mat Pd_d_r=zeros<mat>(Nintd,NCfs);
	
	for (j=0; j<Nintd; j++)
	{
		Pa_d(j,4*j+NCgc)=1;
		Pb_d_r(j,4*j+1+NCgc)=1;
		Pc_d_r(j,4*j+2+NCgc)=1;
		Pd_d_r(j,4*j+3+NCgc)=1;
	}
	
	vec vDd=((w.rows(Nw_lims(1)+1,Nw-1)-w0r)%(w.rows(Nw_lims(1),Nw-2)-w0r))/(w.rows(Nw_lims(1)+1,Nw-1)-w.rows(Nw_lims(1),Nw-2));
	mat Dd=diagmat(vDd);
	vDU=(w.rows(Nw_lims(1),Nw-2)-w0r)/(w.rows(Nw_lims(1)+1,Nw-1)-w.rows(Nw_lims(1),Nw-2));
	DU=diagmat(vDU);
	
	mat Pb_d=Pb_d_r-3*DU*Pa_d;
	mat Pc_d=Pc_d_r+3*pow(DU,2)*Pa_d-2*DU*Pb_d_r;
	mat Pd_d=Pd_d_r-pow(DU,3)*Pa_d+pow(DU,2)*Pb_d_r-DU*Pc_d_r;
	
	Pa_d=pow(Dd,3)*Pa_d;
	Pb_d=pow(Dd,2)*Pb_d;
	Pc_d=Dd*Pc_d;
	
	cx_mat Ka_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kb_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kc_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kd_g=zeros<cx_mat>(Nn,Nintg);
	
	rowvec ug2(Nug+1);
	ug2.cols(0,Nug-1)=ug.t();
	ug2(Nug)=1.0/(wl-w0l);
	
	mat Wng=wn*ones<rowvec>(Nintg);
	mat Ug=ones<vec>(Nn)*ug2;
	
	dcomplex i(0,1);
	
//	cout<<"left side\n";
	
	mat atang=atan((Wng % (Ug.cols(1,Nintg)-Ug.cols(0,Nintg-1)))/(1+w0l*(Ug.cols(1,Nintg)+Ug.cols(0,Nintg-1))+(pow(w0l,2)+pow(Wng,2)) % Ug.cols(0,Nintg-1) % Ug.cols(1,Nintg)));
	mat logg=log((1.0+2*w0l*Ug.cols(1,Nintg)+pow(Ug.cols(1,Nintg),2) % (pow(Wng,2)+pow(w0l,2)))/(1.0+2*w0l*Ug.cols(0,Nintg-1)+pow(Ug.cols(0,Nintg-1),2) % (pow(Wng,2)+pow(w0l,2))));
	
	Ka_g=-(Ug.cols(1,Nintg)-Ug.cols(0,Nintg-1))/pow(Wng+i*w0l,2)-i*(pow(Ug.cols(1,Nintg),2)-pow(Ug.cols(0,Nintg-1),2))/(2*(Wng+i*w0l))+atang/pow(Wng+i*w0l,3)+i*logg/(2*pow(Wng+i*w0l,3));
	
	Kb_g=-i*(Ug.cols(1,Nintg)-Ug.cols(0,Nintg-1))/(Wng+i*w0l)+i*atang/pow(Wng+i*w0l,2)-logg/(2*pow(Wng+i*w0l,2));
	
	Kc_g=-atang/(Wng+i*w0l)-i*logg/(2*(Wng+i*w0l));
	
	Kd_g=-i*atang+logg/2-log(Ug.cols(1,Nintg)/Ug.cols(0,Nintg-1));
	
	rowvec wg=trans(w.rows(0,Nug));
	//rowvec wg=1/ug2+w0l;
	mat Wg=ones<vec>(Nn)*wg;
	
	if (use_HF_exp)
	{
		double utmp;
		int jn, p;
		for (j=0; j<Nug; j++)
		{
			utmp=ug2(j);
			jn=0;
			while (jn<Nn && abs(wn(jn)*utmp)<fg*abs(1+utmp*w0l)) jn++;
			while (jn<Nn && pow(wn(jn),pngmax)==0) jn++;
			if (jn<Nn)
			{
				Ka_g.submat(jn,j,Nn-1,j)=-i*(pow(Ug.submat(jn,j+1,Nn-1,j+1),2)-pow(Ug.submat(jn,j,Nn-1,j),2))/(2*(wn.rows(jn,Nn-1)+i*w0l))-(Ug.submat(jn,j+1,Nn-1,j+1)-Ug.submat(jn,j,Nn-1,j))/pow(wn.rows(jn,Nn-1)+i*w0l,2)+log(Ug.submat(jn,j+1,Nn-1,j+1)/Ug.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0l,3);
				Kb_g.submat(jn,j,Nn-1,j)=-i*(Ug.submat(jn,j+1,Nn-1,j+1)-Ug.submat(jn,j,Nn-1,j))/(wn.rows(jn,Nn-1)+i*w0l)+log(Ug.submat(jn,j+1,Nn-1,j+1)/Ug.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0l,2);
				Kc_g.submat(jn,j,Nn-1,j)=log(Ug.submat(jn,j+1,Nn-1,j+1)/Ug.submat(jn,j,Nn-1,j))/(i*wn.rows(jn,Nn-1)-w0l);
				Kd_g.submat(jn,j,Nn-1,j)=zeros<cx_vec>(Nn-jn);
				for (p=pngmax; p>=1; p--)
				{
					Ka_g.submat(jn,j,Nn-1,j)=Ka_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0l,3);
					Kb_g.submat(jn,j,Nn-1,j)=Kb_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0l,2);
					Kc_g.submat(jn,j,Nn-1,j)=Kc_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/(i*wn.rows(jn,Nn-1)-w0l);
					Kd_g.submat(jn,j,Nn-1,j)=Kd_g.submat(jn,j,Nn-1,j)+pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j+1,Nn-1,j+1),p)-pow(Wg.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p));
				}
			}
		}
	}
	
	Ka_g=-Ka_g/(2*PI);
	Kb_g=-Kb_g/(2*PI);
	Kc_g=-Kc_g/(2*PI);
	Kd_g=-Kd_g/(2*PI);
	
	
	cx_mat Ka_c=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kb_c=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kc_c=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kd_c=zeros<cx_mat>(Nn,Nintc);
	
	mat Wnc=wn*ones<rowvec>(Nintc);
	mat Wc=ones<vec>(Nn)*wc.t();
	
//	cout<<"center\n";
	
	mat logc=log((pow(Wnc,2)+pow(Wc.cols(1,Nintc),2))/(pow(Wnc,2)+pow(Wc.cols(0,Nintc-1),2)));
	mat atanc2=atan((Wnc % (Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1)))/(Wc.cols(1,Nintc) % Wc.cols(0,Nintc-1)+pow(Wnc,2)));
	cx_mat logc2=logc/2+i*atanc2;
	cx_mat dWn=i*Wnc-Wc.cols(0,Nintc-1);
	mat dWc=Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1);
	
	Ka_c=-pow(dWn,2) % dWc-dWn % pow(dWc,2)/2-pow(dWc,3)/3-pow(dWn,3) % logc2;
	Kb_c=-dWn % dWc-pow(dWc,2)/2-pow(dWn,2) % logc2;
	Kc_c=-dWc-dWn % logc2;
	Kd_c=-logc2;
	
	int Pmax=2*pnmax+4;
	imat MP;
	pascal(Pmax+1,MP);
	
	if (use_HF_exp)
	{
		double wtmp;
		int jni, jnr, p, l;
		for (j=0; j<Nwc-1; j++)
		{
			wtmp=abs(wc(j));
			if (abs(wc(j+1))>wtmp) wtmp=abs(wc(j+1));
			jni=0;
			while (jni<Nn && wn(jni)<fi*wtmp) jni++;
			while (jni<Nn && pow(wn(jni),2*pnmax+1)==0) jni++;
			jnr=1;
			while (jnr<Nn && wn(jnr)<fr*wtmp) jnr++;
			while (jnr<Nn && pow(wn(jnr),2*pnmax)==0) jnr++;
			if (jni<Nn || jnr<Nn)
			{
				double wj=wc(j);
				double dw1=wc(j+1)-wj;
				vec dwp=zeros<vec>(Pmax);
				dwp(0)=dw1;
				dwp(1)=pow(dw1,2)+2*wj*dw1;
				for (p=3; p<=Pmax; p++)
				{
					dwp(p-1)=0;
					for (l=0; l<p; l++)
					{
						dwp(p-1)=dwp(p-1)+MP(l,p-l)*pow(dw1,p-l)*pow(wj,l);
					}
				}
				cx_vec vtmp;
				if (jni<Nn)
				{
					vtmp.zeros(Nn-jni);
					vtmp.set_real(real(Ka_c.submat(jni,j,Nn-1,j)));
					Ka_c.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kb_c.submat(jni,j,Nn-1,j)));
					Kb_c.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kc_c.submat(jni,j,Nn-1,j)));
					Kc_c.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kd_c.submat(jni,j,Nn-1,j)));
					Kd_c.submat(jni,j,Nn-1,j)=vtmp;
					
					for (p=2*pnmax+1; p>=1; p-=2)
					{
						Ka_c.submat(jni,j,Nn-1,j)=Ka_c.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(pow(wj,3)*dwp(p-1)/p - 3.0*pow(wj,2)*dwp(p)/(p+1) + 3.0*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jni,Nn-1),p);
						Kb_c.submat(jni,j,Nn-1,j)=Kb_c.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jni,Nn-1),p);
						Kc_c.submat(jni,j,Nn-1,j)=Kc_c.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jni,Nn-1),p);
						Kd_c.submat(jni,j,Nn-1,j)=Kd_c.submat(jni,j,Nn-1,j) - i*pow(-1,(p-1)/2)*dwp(p-1)/(p*pow(wn.rows(jni,Nn-1),p));
					}
				}
				if (jnr<Nn)
				{
					vtmp.zeros(Nn-jnr);
					vtmp.set_imag(imag(Ka_c.submat(jnr,j,Nn-1,j)));
					Ka_c.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kb_c.submat(jnr,j,Nn-1,j)));
					Kb_c.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kc_c.submat(jnr,j,Nn-1,j)));
					Kc_c.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kd_c.submat(jnr,j,Nn-1,j)));
					Kd_c.submat(jnr,j,Nn-1,j)=vtmp;
					for (p=2*pnmax; p>=2; p-=2)
					{
						Ka_c.submat(jnr,j,Nn-1,j)=Ka_c.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(pow(wj,3)*dwp(p-1)/p - 3*pow(wj,2)*dwp(p)/(p+1) + 3*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jnr,Nn-1),p);
						Kb_c.submat(jnr,j,Nn-1,j)=Kb_c.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jnr,Nn-1),p);
						Kc_c.submat(jnr,j,Nn-1,j)=Kc_c.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jnr,Nn-1),p);
						Kd_c.submat(jnr,j,Nn-1,j)=Kd_c.submat(jnr,j,Nn-1,j) - pow(-1,(p-2)/2)*dwp(p-1)/(p*pow(wn.rows(jnr,Nn-1),p));
					}
				}
			}
		}
	}
	
	Ka_c=Ka_c/(2*PI);
	Kb_c=Kb_c/(2*PI);
	Kc_c=Kc_c/(2*PI);
	Kd_c=Kd_c/(2*PI);
	
	cx_mat Ka_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kb_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kc_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kd_d=zeros<cx_mat>(Nn,Nintd);
	
	rowvec ud2(Nud+1);
	ud2.cols(1,Nud)=ud.t();
	ud2(0)=1.0/(wr-w0r);
	
	mat Wnd=wn*ones<rowvec>(Nintd);
	mat Ud=ones<vec>(Nn)*ud2;
	
//	cout<<"right side\n";
	
	mat atand=atan((Wnd % (Ud.cols(1,Nintd)-Ud.cols(0,Nintd-1)))/(1.0+w0r*(Ud.cols(1,Nintd)+Ud.cols(0,Nintd-1))+(pow(w0r,2)+pow(Wnd,2)) % Ud.cols(0,Nintd-1) % Ud.cols(1,Nintd)));
	mat logd=log((1+2*w0r*Ud.cols(1,Nintd)+pow(Ud.cols(1,Nintd),2) % (pow(Wnd,2)+pow(w0r,2)))/(1+2*w0r*Ud.cols(0,Nintd-1)+pow(Ud.cols(0,Nintd-1),2) % (pow(Wnd,2)+pow(w0r,2))));
	
	Ka_d=-(Ud.cols(1,Nintd)-Ud.cols(0,Nintd-1))/pow(Wnd+i*w0r,2) - i*(pow(Ud.cols(1,Nintd),2)-pow(Ud.cols(0,Nintd-1),2))/(2*(Wnd+i*w0r))+atand/pow(Wnd+i*w0r,3) +i*logd/(2*pow(Wnd+i*w0r,3));
	
	Kb_d=-i*(Ud.cols(1,Nintd)-Ud.cols(0,Nintd-1))/(Wnd+i*w0r)+i*atand/pow(Wnd+i*w0r,2)-logd/(2*pow(Wnd+i*w0r,2));
	
	Kc_d=-atand/(Wnd+i*w0r)-i*logd/(2*(Wnd+i*w0r));
	
	Kd_d=-i*atand-log(Ud.cols(1,Nintd)/Ud.cols(0,Nintd-1))+logd/2;
	
	//rowvec wd=1/ud2+w0r;
	rowvec wd=trans(w.rows(Nw_lims(1),Nw-1));
	mat Wd=ones<vec>(Nn)*wd;
	
	if (use_HF_exp)
	{
		double utmp;
		int jn, p;
		for (j=0; j<Nud; j++)
		{
			utmp=ud2(j);
			jn=0;
			while (jn<Nn && abs(wn(jn)*utmp)<fd*abs(1+utmp*w0r)) jn++;
			while (jn<Nn && pow(wn(jn),pndmax)) jn++;
			if (jn<Nn)
			{
				Ka_d.submat(jn,j,Nn-1,j)=-(Ud.submat(jn,j+1,Nn-1,j+1)-Ud.submat(jn,j,Nn-1,j))/pow(wn.rows(jn,Nn-1)+i*w0r,2) - i*(pow(Ud.submat(jn,j+1,Nn-1,j+1),2)-pow(Ud.submat(jn,j,Nn-1,j),2))/(2*(wn.rows(jn,Nn-1)+i*w0r))+log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0r,3);
				Kb_d.submat(jn,j,Nn-1,j)=-i*(Ud.submat(jn,j+1,Nn-1,j+1)-Ud.submat(jn,j,Nn-1,j))/(wn.rows(jn,Nn-1)+i*w0r)+log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0r,2);
				Kc_d.submat(jn,j,Nn-1,j)=log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/(i*wn.rows(jn,Nn-1)-w0r);
				Kd_d.submat(jn,j,Nn-1,j)=zeros<cx_vec>(Nn-jn);
				for (p=pndmax; p>=1; p--)
				{
					Ka_d.submat(jn,j,Nn-1,j)=Ka_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0r,3);
					Kb_d.submat(jn,j,Nn-1,j)=Kb_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0r,2);
					Kc_d.submat(jn,j,Nn-1,j)=Kc_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/(i*wn.rows(jn,Nn-1)-w0r);
					Kd_d.submat(jn,j,Nn-1,j)=Kd_d.submat(jn,j,Nn-1,j)+pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p));
				}
			}
		}
	}
	
	Ka_d=-Ka_d/(2*PI);
	Kb_d=-Kb_d/(2*PI);
	Kc_d=-Kc_d/(2*PI);
	Kd_d=-Kd_d/(2*PI);
	
	cx_mat KG=(Ka_g*Pa_g+Kb_g*Pb_g+Kc_g*Pc_g+Kd_g*Pd_g)*MM;
	cx_mat KC=(Ka_c*Pa_c+Kb_c*Pb_c+Kc_c*Pc_c+Kd_c*Pd_c)*MM;
	cx_mat KD=(Ka_d*Pa_d+Kb_d*Pb_d+Kc_d*Pc_d+Kd_d*Pd_d)*MM;
	
	Kcx=KG+KC+KD;
	
/*
	//test the kernel matrix
	rowvec x0={-2, 0, 3};
	rowvec s0={0.8, 0.3, 1};
	rowvec wgt={1,1,1};
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	
	vec Gtest1=K*test_A;
	vec Gr_test1=Gtest1.rows(even_ind);
	vec Gi_test1=Gtest1.rows(even_ind+1);
	cx_vec Gtest=Kcx*test_A;
	vec Gr_test=real(Gtest);
	vec Gi_test=imag(Gtest);
	
	graph_2D g1, g2, g3, g4, g5, g6;
	char attr1[]="'o',color='b',markeredgecolor='b',markerfacecolor='none'";
	char attr2[]="'s',color='r',markeredgecolor='r',markerfacecolor='none'";
	char attr3[]="'.-',color='b'";
	char attr4[]="'.-',color='m'";
	char attr5[]="'.-',color='r'";
	char attr6[]="'.-',color='c'";
	
	g1.add_data(wn.memptr(), Gr.memptr(), Nn);
	g1.add_attribute(attr1);
	g1.add_data(wn.memptr(), Gr_test.memptr(), Nn);
	g1.add_attribute(attr2);
	g1.curve_plot();
	
	g2.add_data(wn.memptr(), Gi.memptr(), Nn);
	g2.add_attribute(attr1);
	g2.add_data(wn.memptr(), Gi_test.memptr(), Nn);
	g2.add_attribute(attr2);
	g2.curve_plot();
	
	vec DGr1=(Gr_test-Gr)/Gr, DGr2=(Gr_test1-Gr)/Gr, DGr=(Gr_test-Gr_test1);
	vec DGi1=(Gi_test-Gi)/Gi, DGi2=(Gi_test1-Gi)/Gi, DGi=(Gi_test-Gi_test1);
	g3.add_data(wn.memptr(),DGr1.memptr(),Nn);
	g3.add_attribute(attr3);
	g3.add_data(wn.memptr(),DGr2.memptr(),Nn);
	g3.add_attribute(attr4);
	g3.curve_plot();
	
	g4.add_data(wn.memptr(),DGr.memptr(),Nn);
	g4.add_attribute(attr4);
	g4.curve_plot();

	g5.add_data(wn.memptr(),DGi1.memptr(),Nn);
	g5.add_attribute(attr5);
	g5.add_data(wn.memptr(),DGi2.memptr(),Nn);
	g5.add_attribute(attr6);
	g5.curve_plot();

	g6.add_data(wn.memptr(),DGi.memptr(),Nn);
	g6.add_attribute(attr6);
	g6.curve_plot();
	
	graph_2D::show_figures();
	*/
	
//	cout<<"moments\n";

	rowvec Knorm_a_g=zeros<rowvec>(Nintg);
	rowvec Knorm_b_g=zeros<rowvec>(Nintg);
	rowvec Knorm_c_g=zeros<rowvec>(Nintg);
	rowvec Knorm_d_g=zeros<rowvec>(Nintg);
	
	Knorm_a_g=-(pow(ug2.cols(1,Nug),2)-pow(ug2.cols(0,Nug-1),2))/2;
	Knorm_b_g=-(ug2.cols(1,Nug)-ug2.cols(0,Nug-1));
	Knorm_c_g=-log(ug2.cols(1,Nug)/ug2.cols(0,Nug-1));
	Knorm_d_g=1.0/ug2.cols(1,Nug)-1.0/ug2.cols(0,Nug-1);
	
	rowvec KM0g=(Knorm_a_g*Pa_g+Knorm_b_g*Pb_g+Knorm_c_g*Pc_g+Knorm_d_g*Pd_g)*MM/(2*PI);
	
	rowvec Knorm_a_c=zeros<rowvec>(Nintc);
	rowvec Knorm_b_c=zeros<rowvec>(Nintc);
	rowvec Knorm_c_c=zeros<rowvec>(Nintc);
	rowvec Knorm_d_c=zeros<rowvec>(Nintc);
	
	Knorm_a_c=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),4)/4);
	Knorm_b_c=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),3)/3);
	Knorm_c_c=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),2)/2);
	Knorm_d_c=trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2));
	
	rowvec KM0c=(Knorm_a_c*Pa_c+Knorm_b_c*Pb_c+Knorm_c_c*Pc_c+Knorm_d_c*Pd_c)*MM/(2*PI);
	
	rowvec Knorm_a_d=zeros<rowvec>(Nintd);
	rowvec Knorm_b_d=zeros<rowvec>(Nintd);
	rowvec Knorm_c_d=zeros<rowvec>(Nintd);
	rowvec Knorm_d_d=zeros<rowvec>(Nintd);
	
	Knorm_a_d=-(pow(ud2.cols(1,Nud),2)-pow(ud2.cols(0,Nud-1),2))/2;
	Knorm_b_d=-(ud2.cols(1,Nud)-ud2.cols(0,Nud-1));
	Knorm_c_d=-log(ud2.cols(1,Nud)/ud2.cols(0,Nud-1));
	Knorm_d_d=1.0/ud2.cols(1,Nud)-1.0/ud2.cols(0,Nud-1);
	
	rowvec KM0d=(Knorm_a_d*Pa_d+Knorm_b_d*Pb_d+Knorm_c_d*Pc_d+Knorm_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM0=KM0g+KM0c+KM0d;
	
	
	rowvec KM1_a_g=zeros<rowvec>(Nintg);
	rowvec KM1_b_g=zeros<rowvec>(Nintg);
	rowvec KM1_c_g=zeros<rowvec>(Nintg);
	rowvec KM1_d_g=zeros<rowvec>(Nintg);
	
	KM1_a_g=Knorm_b_g;
	KM1_b_g=Knorm_c_g;
	KM1_c_g=Knorm_d_g;
	KM1_d_g=(1.0/pow(ug2.cols(1,Nug),2)-1.0/pow(ug2.cols(0,Nug-1),2))/2;
	
	rowvec KM1g_tmp=(KM1_a_g*Pa_g+KM1_b_g*Pb_g+KM1_c_g*Pc_g+KM1_d_g*Pd_g)*MM/(2*PI);
	
	rowvec KM1g=w0l*KM0g+KM1g_tmp;
	
	mat Wjc=diagmat(wc.rows(0,Nintc-1));
	
	rowvec KM1_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),5)/5.0);
	
	rowvec KM1_a_c=KM1_a_c_tmp+Knorm_a_c*Wjc;
	rowvec KM1_b_c=Knorm_a_c+Knorm_b_c*Wjc;
	rowvec KM1_c_c=Knorm_b_c+Knorm_c_c*Wjc;
	rowvec KM1_d_c=Knorm_c_c+Knorm_d_c*Wjc;
	
	rowvec KM1c=(KM1_a_c*Pa_c+KM1_b_c*Pb_c+KM1_c_c*Pc_c+KM1_d_c*Pd_c)*MM/(2*PI);
	
	rowvec KM1_a_d=zeros<rowvec>(Nintd);
	rowvec KM1_b_d=zeros<rowvec>(Nintd);
	rowvec KM1_c_d=zeros<rowvec>(Nintd);
	rowvec KM1_d_d=zeros<rowvec>(Nintd);
	
	KM1_a_d=Knorm_b_d;
	KM1_b_d=Knorm_c_d;
	KM1_c_d=Knorm_d_d;
	KM1_d_d=(1.0/pow(ud2.cols(1,Nud),2)-1.0/pow(ud2.cols(0,Nud-1),2))/2;
	
	rowvec KM1d_tmp=(KM1_a_d*Pa_d+KM1_b_d*Pb_d+KM1_c_d*Pc_d+KM1_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM1d=w0r*KM0d+KM1d_tmp;
	
	rowvec KM1=KM1g+KM1c+KM1d;
	
	rowvec KM2_a_g=zeros<rowvec>(Nintg);
	rowvec KM2_b_g=zeros<rowvec>(Nintg);
	rowvec KM2_c_g=zeros<rowvec>(Nintg);
	rowvec KM2_d_g=zeros<rowvec>(Nintg);
	
	KM2_a_g=Knorm_c_g;
	KM2_b_g=Knorm_d_g;
	KM2_c_g=KM1_d_g;
	KM2_d_g=(1.0/pow(ug2.cols(1,Nug),3)-1.0/pow(ug2.cols(0,Nug-1),3))/3;
	
	rowvec KM2g_tmp=(KM2_a_g*Pa_g+KM2_b_g*Pb_g+KM2_c_g*Pc_g+KM2_d_g*Pd_g)*MM/(2*PI);
	
	rowvec KM2g=pow(w0l,2)*KM0g+2*w0l*KM1g_tmp+KM2g_tmp;
	
	rowvec KM2_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),6)/6);
	
	rowvec KM2_a_c=KM2_a_c_tmp+2*KM1_a_c_tmp*Wjc+Knorm_a_c*pow(Wjc,2);
	rowvec KM2_b_c=KM1_a_c_tmp+2*Knorm_a_c*Wjc+Knorm_b_c*pow(Wjc,2);
	rowvec KM2_c_c=Knorm_a_c+2*Knorm_b_c*Wjc+Knorm_c_c*pow(Wjc,2);
	rowvec KM2_d_c=Knorm_b_c+2*Knorm_c_c*Wjc+Knorm_d_c*pow(Wjc,2);
	
	rowvec KM2c=(KM2_a_c*Pa_c+KM2_b_c*Pb_c+KM2_c_c*Pc_c+KM2_d_c*Pd_c)*MM/(2*PI);
	
	rowvec KM2_a_d=zeros<rowvec>(Nintd);
	rowvec KM2_b_d=zeros<rowvec>(Nintd);
	rowvec KM2_c_d=zeros<rowvec>(Nintd);
	rowvec KM2_d_d=zeros<rowvec>(Nintd);
	
	KM2_a_d=Knorm_c_d;
	KM2_b_d=Knorm_d_d;
	KM2_c_d=KM1_d_d;
	KM2_d_d=(1.0/pow(ud2.cols(1,Nud),3)-1.0/pow(ud2.cols(0,Nud-1),3))/3;
	
	rowvec KM2d_tmp=(KM2_a_d*Pa_d+KM2_b_d*Pb_d+KM2_c_d*Pc_d+KM2_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM2d=pow(w0r,2)*KM0d+2*w0r*KM1d_tmp+KM2d_tmp;
	
	rowvec KM2=KM2g+KM2c+KM2d;
	
	rowvec KM3_a_g=zeros<rowvec>(Nintg);
	rowvec KM3_b_g=zeros<rowvec>(Nintg);
	rowvec KM3_c_g=zeros<rowvec>(Nintg);
	rowvec KM3_d_g=zeros<rowvec>(Nintg);
	
	KM3_a_g=Knorm_d_g;
	KM3_b_g=KM1_d_g;
	KM3_c_g=KM2_d_g;
	KM3_d_g=(1.0/pow(ug2.cols(1,Nug),4)-1.0/pow(ug2.cols(0,Nug-1),4))/4;
	
	rowvec KM3g_tmp=(KM3_a_g*Pa_g+KM3_b_g*Pb_g+KM3_c_g*Pc_g+KM3_d_g*Pd_g)*MM/(2*PI);
	
	rowvec KM3g=pow(w0l,3)*KM0g+3*pow(w0l,2)*KM1g_tmp+3*w0l*KM2g_tmp+KM3g_tmp;
	
	rowvec KM3_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),7)/7);
	
	rowvec KM3_a_c=KM3_a_c_tmp+3*KM2_a_c_tmp*Wjc+3*KM1_a_c_tmp*pow(Wjc,2)+Knorm_a_c*pow(Wjc,3);
	rowvec KM3_b_c=KM2_a_c_tmp+3*KM1_a_c_tmp*Wjc+3*Knorm_a_c*pow(Wjc,2)+Knorm_b_c*pow(Wjc,3);
	rowvec KM3_c_c=KM1_a_c_tmp+3*Knorm_a_c*Wjc+3*Knorm_b_c*pow(Wjc,2)+Knorm_c_c*pow(Wjc,3);
	rowvec KM3_d_c=Knorm_a_c+3*Knorm_b_c*Wjc+3*Knorm_c_c*pow(Wjc,2)+Knorm_d_c*pow(Wjc,3);
	
	rowvec KM3c=(KM3_a_c*Pa_c+KM3_b_c*Pb_c+KM3_c_c*Pc_c+KM3_d_c*Pd_c)*MM/(2*PI);
	
	rowvec KM3_a_d=zeros<rowvec>(Nintd);
	rowvec KM3_b_d=zeros<rowvec>(Nintd);
	rowvec KM3_c_d=zeros<rowvec>(Nintd);
	rowvec KM3_d_d=zeros<rowvec>(Nintd);
	
	KM3_a_d=Knorm_d_d;
	KM3_b_d=KM1_d_d;
	KM3_c_d=KM2_d_d;
	KM3_d_d=(1.0/pow(ud2.cols(1,Nud),4)-1.0/pow(ud2.cols(0,Nud-1),4))/4;
	
	rowvec KM3d_tmp=(KM3_a_d*Pa_d+KM3_b_d*Pb_d+KM3_c_d*Pc_d+KM3_d_d*Pd_d)*MM/(2*PI);
	
	rowvec KM3d=pow(w0r,3)*KM0d+3*pow(w0r,2)*KM1d_tmp+3*w0r*KM2d_tmp+KM3d_tmp;
	
	rowvec KM3=KM3g+KM3c+KM3d;
	
	KM.zeros(4,Nw);
	 
	KM.row(0)=KM0;
	KM.row(1)=KM1;
	KM.row(2)=KM2;
	KM.row(3)=KM3;
	
	K.zeros(2*Nn,Nw);
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	K.rows(even_ind)=real(Kcx);
	K.rows(even_ind+1)=imag(Kcx);
	
	cout<<"kernel matrix defined.\n";
	
	/*
	 //check moments
	 
	 rowvec x0={-2, 0, 3};
	 rowvec s0={0.8, 0.3, 1};
	 rowvec wgt={1,1,1};
	 
	 vec test_A;
	 sum_gaussians(w, x0, s0, wgt, test_A);
	
 	cout<<KM*test_A<<endl;
	 */
	
	return true;
}

bool OmegaMaxEnt_data::Kernel_G_fermions()
{
	bool use_HF_exp=true;
	double fg=1.7;
	int pngmax=100;
	double fi=fg;
	double fr=fi;
	int pnmax=50;
	double fd=fg;
	int pndmax=pngmax;
	
	cout<<"defining kernel matrix...\n";
	
	Kcx.zeros(Nn,Nw);
 	KM.zeros(4,Nw);
 
	int Nug=Nw_lims(0);
 	int Nud=Nw-Nw_lims(1)-1;
 
	vec ug, ud;
	if (Du_constant)
	{
 		double dug=dwl/((wl-dwl-w0l)*(wl-w0l));
		vec ug_int=linspace<vec>(1,Nug,Nug);
		ug=-dug*ug_int;
		
 		double dud=dwr/((wr-w0r)*(wr+dwr-w0r));
		vec ud_int=linspace<vec>(Nud,1,Nud);
		ud=dud*ud_int;
	}
	else
	{
		ug=1.0/(w.rows(0,Nug-1)-w0l);
		ud=1.0/(w.rows(Nw_lims(1)+1,Nw-1)-w0r);
	}
 
	mat MM;
	spline_matrix_G_part(w, Nw_lims, ws, MM);
	
	int Ncfs0=MM.n_rows;

//testing the spline matrix
//	rowvec x0={wl, 0, wr/2};
//	rowvec s0={SW/10, SW/20, SW/5};
//	rowvec wgt={1,1,1};
	
//	vec test_A;
//	sum_gaussians(w, x0, s0, wgt, test_A);
/*
	vec test_A=A_ref;
	
	vec coeffs0=MM*test_A;

	vec coeffs1(4*(Nw+1));
	coeffs1(0)=coeffs0(0);
	coeffs1(1)=coeffs0(1);
	coeffs1(2)=0;
	coeffs1(3)=0;
	coeffs1(4*(Nw+1)-4)=coeffs0(Ncfs0-2);
	coeffs1(4*(Nw+1)-3)=coeffs0(Ncfs0-1);
	coeffs1(4*(Nw+1)-2)=0;
	coeffs1(4*(Nw+1)-1)=0;
	
	uvec ind_tmp=linspace<uvec>(1,Nw-1,Nw-1);
	coeffs1.rows(4*ind_tmp)=coeffs0(3*ind_tmp-1);
	coeffs1.rows(4*ind_tmp+1)=coeffs0(3*ind_tmp);
	coeffs1.rows(4*ind_tmp+2)=coeffs0(3*ind_tmp+1);
	
	ind_tmp=linspace<uvec>(1,Nw_lims(1),Nw_lims(1));
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp-1);
	ind_tmp=linspace<uvec>(Nw_lims(1)+1,Nw-1,Nw-Nw_lims(1)-1);
	coeffs1.rows(4*ind_tmp+3)=test_A.rows(ind_tmp);
	

 	double Nw2=400;
	vec w2=linspace<vec>(wl-SW,wr+SW,Nw2);
	
	vec test_A2;
	sum_gaussians(w2, x0, s0, wgt, test_A2);

	vec test_A3;
	spline_val_G_part(w2, w, Nw_lims, ws, coeffs1, test_A3);

	graph_2D g1, g2;
	char xl[]="$\\\\omega$";
	char yl[]="A";
	double xlims[2], ylims[2];
	xlims[0]=w2(0);
	xlims[1]=w2(w2.n_rows-1);
	ylims[0]=0;
	ylims[1]=1.1*test_A2.max();
	
	g1.add_data(w2.memptr(),test_A2.memptr(),w2.n_rows);
	g1.add_data(w2.memptr(),test_A3.memptr(),w2.n_rows);
	g1.set_axes_labels(xl,yl);
	g1.set_axes_lims(xlims,ylims);
	g1.curve_plot();
	
	char yl2[]="Delta A";
	vec Delta_A=test_A2-test_A3;
	ylims[0]=Delta_A.min();
	ylims[1]=Delta_A.max();
	g2.add_data(w2.memptr(),Delta_A.memptr(),w2.n_rows);
	g2.set_axes_labels(xl,yl2);
	g2.set_axes_lims(xlims,ylims);
	g2.curve_plot();
	
	graph_2D::show_figures();
*/

	mat MC(Ncfs0+Nw,Nw);
	
	MC.submat(0,0,Ncfs0-1,Nw-1)=MM;
	MC.submat(Ncfs0,0,Ncfs0+Nw-1,Nw-1)=eye<mat>(Nw,Nw);
	 
	int Nint=Nw+1;
	int Nintg=Nug+1;

	mat Pa_g=zeros<mat>(Nintg,3*Nint-2+Nw);
	mat Pb_g_r=zeros<mat>(Nintg,3*Nint-2+Nw);
	mat Pc_g_r=zeros<mat>(Nintg,3*Nint-2+Nw);
	mat Pd_g_r=zeros<mat>(Nintg,3*Nint-2+Nw);

	Pa_g(0,0)=1;
	Pb_g_r(0,1)=1;
	
	int j;
	for (j=2; j<=Nintg; j++)
	{
		Pa_g(j-1,3*j-4)=1;
		Pb_g_r(j-1,3*j-3)=1;
		Pc_g_r(j-1,3*j-2)=1;
		Pd_g_r(j-1,j+3*Nint-4)=1;
	}
	
	mat U=zeros<mat>(Nug+1,Nug+1);
	vec ug1=zeros<vec>(Nug+1);
	ug1.rows(1,Nug)=ug;
	
	U.diag()=ug1;
		
	mat Pb_g=Pb_g_r-3*U*Pa_g;
	mat Pc_g=Pc_g_r+3*pow(U,2)*Pa_g-2*U*Pb_g_r;
	mat Pd_g=Pd_g_r-pow(U,3)*Pa_g+pow(U,2)*Pb_g_r-U*Pc_g_r;
	
	int	Nintc=Nwc-1;
		
	mat	Pa_c=zeros<mat>(Nintc,3*Nint-2+Nw);
	mat	Pb_c_r=zeros<mat>(Nintc,3*Nint-2+Nw);
	mat	Pc_c_r=zeros<mat>(Nintc,3*Nint-2+Nw);
	mat	Pd_c_r=zeros<mat>(Nintc,3*Nint-2+Nw);
		
	for (j=1; j<=Nintc; j++)
	{
	 	Pa_c(j-1,3*j-4+3*Nintg)=1;
		Pb_c_r(j-1,3*j-3+3*Nintg)=1;
		Pc_c_r(j-1,3*j-2+3*Nintg)=1;
		Pd_c_r(j-1,j+3*Nint-3+Nug)=1;
	}
	
	int Nintd=Nud+1;
	
	mat Pa_d=zeros<mat>(Nintd,3*Nint-2+Nw);
	mat Pb_d_r=zeros<mat>(Nintd,3*Nint-2+Nw);
	mat Pc_d_r=zeros<mat>(Nintd,3*Nint-2+Nw);
	mat Pd_d_r=zeros<mat>(Nintd,3*Nint-2+Nw);
	
	for (j=1; j<Nintd; j++)
	{
		Pa_d(j-1,3*j-4+3*Nintg+3*Nintc)=1;
		Pb_d_r(j-1,3*j-3+3*Nintg+3*Nintc)=1;
		Pc_d_r(j-1,3*j-2+3*Nintg+3*Nintc)=1;
		Pd_d_r(j-1,j+3*Nint-3+Nug+Nwc)=1;
	}
	
	j=Nintd;
	Pa_d(j-1,3*j-4+3*Nintg+3*Nintc)=1;
	Pb_d_r(j-1,3*j-3+3*Nintg+3*Nintc)=1;
	
	U.zeros(Nud+1,Nud+1);
	vec ud1=zeros<vec>(Nud+1);
	ud1.rows(0,Nud-1)=ud;
	U.diag()=ud1;
	
	mat Pb_d=Pb_d_r-3*U*Pa_d;
	mat Pc_d=Pc_d_r+3*pow(U,2)*Pa_d-2*U*Pb_d_r;
	mat Pd_d=Pd_d_r-pow(U,3)*Pa_d+pow(U,2)*Pb_d_r-U*Pc_d_r;

// check projectors
//	vec a=Pa_g*MC*test_A;
//	vec b=Pb_g_r*MC*test_A;
//	vec c=Pc_g_r*MC*test_A;
//	vec d=Pd_g_r*MC*test_A;
//	ind_tmp=linspace<uvec>(0,Nug,Nug+1);
//	vec a=Pa_c*MC*test_A;
//	vec b=Pb_c_r*MC*test_A;
//	vec c=Pc_c_r*MC*test_A;
//	vec d=Pd_c_r*MC*test_A;
//	ind_tmp=linspace<uvec>(Nug+1,Nug+Nintc,Nintc);
//	vec a=Pa_d*MC*test_A;
//	vec b=Pb_d_r*MC*test_A;
//	vec c=Pc_d_r*MC*test_A;
//	vec d=Pd_d_r*MC*test_A;
//	ind_tmp=linspace<uvec>(Nug+Nintc+1,Nug+Nintc+Nud+1,Nud+1);
	
//	cout<<"a:\n"<<a<<endl;
//	cout<<"diff(a):\n"<<a-coeffs1.rows(4*ind_tmp)<<endl;
//	cout<<"b:\n"<<b<<endl;
//	cout<<"diff(b):\n"<<b-coeffs1.rows(4*ind_tmp+1)<<endl;
//	cout<<"c:\n"<<c<<endl;
//	cout<<"diff(c):\n"<<c-coeffs1.rows(4*ind_tmp+2)<<endl;
//	cout<<"d:\n"<<d<<endl;
//	cout<<"diff(d):\n"<<d-coeffs1.rows(4*ind_tmp+3)<<endl;
	

	cx_mat Ka_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kb_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kc_g=zeros<cx_mat>(Nn,Nintg);
	cx_mat Kd_g=zeros<cx_mat>(Nn,Nintg);
	
	rowvec ug2(Nug+1);
	ug2.cols(0,Nug-1)=ug.t();
	ug2(Nug)=1.0/(wl-w0l);
	
	mat Wng=wn*ones<rowvec>(Nintg-1);
	mat Ug=ones<vec>(Nn)*ug2;
	
	dcomplex i(0,1);
	
	Ka_g.col(0)=-ug(0)/pow(wn+i*w0l,2)-i*pow(ug(0),2)/(2*(wn+i*w0l))+atan((wn*ug(0))/(1.0+ug(0)*w0l))/pow(wn+i*w0l,3)+i*log(1.0+2*w0l*ug(0)+pow(ug(0),2)*(pow(wn,2)+pow(w0l,2)))/(2*pow(wn+i*w0l,3));
	Kb_g.col(0)=-i*ug(0)/(wn+i*w0l)+i*atan((wn*ug(0))/(1+ug(0)*w0l))/pow(wn+i*w0l,2)-log(1.0+2*w0l*ug(0)+pow(ug(0),2)*(pow(wn,2)+pow(w0l,2)))/(2*pow(wn+i*w0l,2));

	mat atang=atan((Wng % (Ug.cols(1,Nintg-1)-Ug.cols(0,Nintg-2)))/(1+w0l*(Ug.cols(1,Nintg-1)+Ug.cols(0,Nintg-2))+(pow(w0l,2)+pow(Wng,2)) % Ug.cols(0,Nintg-2) % Ug.cols(1,Nintg-1)));
	mat logg=log((1.0+2*w0l*Ug.cols(1,Nintg-1)+pow(Ug.cols(1,Nintg-1),2) % (pow(Wng,2)+pow(w0l,2)))/(1.0+2*w0l*Ug.cols(0,Nintg-2)+pow(Ug.cols(0,Nintg-2),2) % (pow(Wng,2)+pow(w0l,2))));
	 
	Ka_g.cols(1,Nintg-1)=-(Ug.cols(1,Nintg-1)-Ug.cols(0,Nintg-2))/pow(Wng+i*w0l,2)-i*(pow(Ug.cols(1,Nintg-1),2)-pow(Ug.cols(0,Nintg-2),2))/(2*(Wng+i*w0l))+atang/pow(Wng+i*w0l,3)+i*logg/(2*pow(Wng+i*w0l,3));
	 
	Kb_g.cols(1,Nintg-1)=-i*(Ug.cols(1,Nintg-1)-Ug.cols(0,Nintg-2))/(Wng+i*w0l)+i*atang/pow(Wng+i*w0l,2)-logg/(2*pow(Wng+i*w0l,2));
	 
	Kc_g.cols(1,Nintg-1)=-atang/(Wng+i*w0l)-i*logg/(2*(Wng+i*w0l));
	 
	Kd_g.cols(1,Nintg-1)=-i*atang+logg/2-log(Ug.cols(1,Nintg-1)/Ug.cols(0,Nintg-2));

	rowvec wg=1/ug2+w0l;
	mat Wg=ones<vec>(Nn)*wg;
	
	if (use_HF_exp)
	{
		double utmp;
		int jn, p;
		for (j=1; j<=Nug; j++)
		{
			utmp=ug2(j-1);
			jn=0;
			while (jn<Nn && abs(wn(jn)*utmp)<fg*abs(1+utmp*w0l)) jn++;
			while (jn<Nn && pow(wn(jn),pngmax)==0) jn++;
			if (jn<Nn)
			{
				Ka_g.submat(jn,j,Nn-1,j)=-i*(pow(Ug.submat(jn,j,Nn-1,j),2)-pow(Ug.submat(jn,j-1,Nn-1,j-1),2))/(2*(wn.rows(jn,Nn-1)+i*w0l))-(Ug.submat(jn,j,Nn-1,j)-Ug.submat(jn,j-1,Nn-1,j-1))/pow(wn.rows(jn,Nn-1)+i*w0l,2)+log(Ug.submat(jn,j,Nn-1,j)/Ug.submat(jn,j-1,Nn-1,j-1))/pow(i*wn.rows(jn,Nn-1)-w0l,3);
				Kb_g.submat(jn,j,Nn-1,j)=-i*(Ug.submat(jn,j,Nn-1,j)-Ug.submat(jn,j-1,Nn-1,j-1))/(wn.rows(jn,Nn-1)+i*w0l)+log(Ug.submat(jn,j,Nn-1,j)/Ug.submat(jn,j-1,Nn-1,j-1))/pow(i*wn.rows(jn,Nn-1)-w0l,2);
				Kc_g.submat(jn,j,Nn-1,j)=log(Ug.submat(jn,j,Nn-1,j)/Ug.submat(jn,j-1,Nn-1,j-1))/(i*wn.rows(jn,Nn-1)-w0l);
				Kd_g.submat(jn,j,Nn-1,j)=zeros<cx_vec>(Nn-jn);
				for (p=pngmax; p>=1; p--)
				{
					Ka_g.submat(jn,j,Nn-1,j)=Ka_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j,Nn-1,j),p)-pow(Wg.submat(jn,j-1,Nn-1,j-1),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0l,3);
					Kb_g.submat(jn,j,Nn-1,j)=Kb_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j,Nn-1,j),p)-pow(Wg.submat(jn,j-1,Nn-1,j-1),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0l,2);
					Kc_g.submat(jn,j,Nn-1,j)=Kc_g.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j,Nn-1,j),p)-pow(Wg.submat(jn,j-1,Nn-1,j-1),p))/(p*pow(wn.rows(jn,Nn-1),p)))/(i*wn.rows(jn,Nn-1)-w0l);
					Kd_g.submat(jn,j,Nn-1,j)=Kd_g.submat(jn,j,Nn-1,j)+pow(i,p)*pow(-1,p+1)*(pow(Wg.submat(jn,j,Nn-1,j),p)-pow(Wg.submat(jn,j-1,Nn-1,j-1),p))/(p*pow(wn.rows(jn,Nn-1),p));
				}
			}
		}
	}
	 
	Ka_g=-Ka_g/(2*PI);
	Kb_g=-Kb_g/(2*PI);
	Kc_g=-Kc_g/(2*PI);
	Kd_g=-Kd_g/(2*PI);
	 

	cx_mat Ka_c_r=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kb_c_r=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kc_c_r=zeros<cx_mat>(Nn,Nintc);
	cx_mat Kd_c_r=zeros<cx_mat>(Nn,Nintc);

	mat Wnc=wn*ones<rowvec>(Nintc);
	mat Wc=ones<vec>(Nn)*wc.t();
	 
	mat logc=log((pow(Wnc,2)+pow(Wc.cols(1,Nintc),2))/(pow(Wnc,2)+pow(Wc.cols(0,Nintc-1),2)));
	mat atanc2=atan((Wnc % (Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1)))/(Wc.cols(1,Nintc) % Wc.cols(0,Nintc-1)+pow(Wnc,2)));
	cx_mat logc2=logc/2+i*atanc2;
	cx_mat dWn=i*Wnc-Wc.cols(0,Nintc-1);
	mat dWc=Wc.cols(1,Nintc)-Wc.cols(0,Nintc-1);
	 
	Ka_c_r=-pow(dWn,2) % dWc-dWn % pow(dWc,2)/2-pow(dWc,3)/3-pow(dWn,3) % logc2;
	Kb_c_r=-dWn % dWc-pow(dWc,2)/2-pow(dWn,2) % logc2;
	Kc_c_r=-dWc-dWn % logc2;
	Kd_c_r=-logc2;
	
	int Pmax=2*pnmax+4;
	imat MP;
	pascal(Pmax+1,MP);

	if (use_HF_exp)
	{
		double wtmp;
		int jni, jnr, p, l;
		for (j=0; j<Nwc-1; j++)
		{
			wtmp=abs(wc(j));
			if (abs(wc(j+1))>wtmp) wtmp=abs(wc(j+1));
			jni=0;
			while (jni<Nn && wn(jni)<fi*wtmp) jni++;
			while (jni<Nn && pow(wn(jni),2*pnmax+1)==0) jni++;
			jnr=1;
			while (jnr<Nn && wn(jnr)<fr*wtmp) jnr++;
			while (jnr<Nn && pow(wn(jnr),2*pnmax)==0) jnr++;
			if (jni<Nn || jnr<Nn)
			{
				double wj=wc(j);
				double dw1=wc(j+1)-wj;
				vec dwp=zeros<vec>(Pmax);
				dwp(0)=dw1;
				dwp(1)=pow(dw1,2)+2*wj*dw1;
				for (p=3; p<=Pmax; p++)
				{
					dwp(p-1)=0;
					for (l=0; l<p; l++)
					{
						dwp(p-1)=dwp(p-1)+MP(l,p-l)*pow(dw1,p-l)*pow(wj,l);
					}
				}
				cx_vec vtmp;
				if (jni<Nn)
				{
					vtmp.zeros(Nn-jni);
					vtmp.set_real(real(Ka_c_r.submat(jni,j,Nn-1,j)));
					Ka_c_r.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kb_c_r.submat(jni,j,Nn-1,j)));
					Kb_c_r.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kc_c_r.submat(jni,j,Nn-1,j)));
					Kc_c_r.submat(jni,j,Nn-1,j)=vtmp;
					vtmp.set_real(real(Kd_c_r.submat(jni,j,Nn-1,j)));
					Kd_c_r.submat(jni,j,Nn-1,j)=vtmp;

					for (p=2*pnmax+1; p>=1; p-=2)
					{
						Ka_c_r.submat(jni,j,Nn-1,j)=Ka_c_r.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(pow(wj,3)*dwp(p-1)/p - 3.0*pow(wj,2)*dwp(p)/(p+1) + 3.0*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jni,Nn-1),p);
						Kb_c_r.submat(jni,j,Nn-1,j)=Kb_c_r.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jni,Nn-1),p);
						Kc_c_r.submat(jni,j,Nn-1,j)=Kc_c_r.submat(jni,j,Nn-1,j) + i*pow(-1,(p-1)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jni,Nn-1),p);
						Kd_c_r.submat(jni,j,Nn-1,j)=Kd_c_r.submat(jni,j,Nn-1,j) - i*pow(-1,(p-1)/2)*dwp(p-1)/(p*pow(wn.rows(jni,Nn-1),p));
					}
				}
				if (jnr<Nn)
				{
					vtmp.zeros(Nn-jnr);
					vtmp.set_imag(imag(Ka_c_r.submat(jnr,j,Nn-1,j)));
					Ka_c_r.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kb_c_r.submat(jnr,j,Nn-1,j)));
					Kb_c_r.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kc_c_r.submat(jnr,j,Nn-1,j)));
					Kc_c_r.submat(jnr,j,Nn-1,j)=vtmp;
					vtmp.set_imag(imag(Kd_c_r.submat(jnr,j,Nn-1,j)));
					Kd_c_r.submat(jnr,j,Nn-1,j)=vtmp;
					for (p=2*pnmax; p>=2; p-=2)
					{
						Ka_c_r.submat(jnr,j,Nn-1,j)=Ka_c_r.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(pow(wj,3)*dwp(p-1)/p - 3*pow(wj,2)*dwp(p)/(p+1) + 3*wj*dwp(p+1)/(p+2) - dwp(p+2)/(p+3))/pow(wn.rows(jnr,Nn-1),p);
						Kb_c_r.submat(jnr,j,Nn-1,j)=Kb_c_r.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(-pow(wj,2)*dwp(p-1)/p + 2*wj*dwp(p)/(p+1) - dwp(p+1)/(p+2))/pow(wn.rows(jnr,Nn-1),p);
						Kc_c_r.submat(jnr,j,Nn-1,j)=Kc_c_r.submat(jnr,j,Nn-1,j) + pow(-1,(p-2)/2)*(wj*dwp(p-1)/p - dwp(p)/(p+1))/pow(wn.rows(jnr,Nn-1),p);
						Kd_c_r.submat(jnr,j,Nn-1,j)=Kd_c_r.submat(jnr,j,Nn-1,j) - pow(-1,(p-2)/2)*dwp(p-1)/(p*pow(wn.rows(jnr,Nn-1),p));
					}
				}
			}
	 	}
	}
	
	Ka_c_r=Ka_c_r/(2*PI);
	Kb_c_r=Kb_c_r/(2*PI);
	Kc_c_r=Kc_c_r/(2*PI);
	Kd_c_r=Kd_c_r/(2*PI);

	cx_mat Ka_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kb_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kc_d=zeros<cx_mat>(Nn,Nintd);
	cx_mat Kd_d=zeros<cx_mat>(Nn,Nintd);
	
	rowvec ud2(Nud+1);
	ud2.cols(1,Nud)=ud.t();
	ud2(0)=1.0/(wr-w0r);
	 
	mat Wnd=wn*ones<rowvec>(Nintd-1);
	mat Ud=ones<vec>(Nn)*ud2;

	mat atand=atan((Wnd % (Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2)))/(1.0+w0r*(Ud.cols(1,Nintd-1)+Ud.cols(0,Nintd-2))+(pow(w0r,2)+pow(Wnd,2)) % Ud.cols(0,Nintd-2) % Ud.cols(1,Nintd-1)));
	mat logd=log((1+2*w0r*Ud.cols(1,Nintd-1)+pow(Ud.cols(1,Nintd-1),2) % (pow(Wnd,2)+pow(w0r,2)))/(1+2*w0r*Ud.cols(0,Nintd-2)+pow(Ud.cols(0,Nintd-2),2) % (pow(Wnd,2)+pow(w0r,2))));
	 
	Ka_d.cols(0,Nintd-2)=-(Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2))/pow(Wnd+i*w0r,2) - i*(pow(Ud.cols(1,Nintd-1),2)-pow(Ud.cols(0,Nintd-2),2))/(2*(Wnd+i*w0r))+atand/pow(Wnd+i*w0r,3) +i*logd/(2*pow(Wnd+i*w0r,3));
	 
	Kb_d.cols(0,Nintd-2)=-i*(Ud.cols(1,Nintd-1)-Ud.cols(0,Nintd-2))/(Wnd+i*w0r)+i*atand/pow(Wnd+i*w0r,2)-logd/(2*pow(Wnd+i*w0r,2));
	 
	Kc_d.cols(0,Nintd-2)=-atand/(Wnd+i*w0r)-i*logd/(2*(Wnd+i*w0r));
	 
	Kd_d.cols(0,Nintd-2)=-i*atand-log(Ud.cols(1,Nintd-1)/Ud.cols(0,Nintd-2))+logd/2;
	 
	Ka_d.col(Nintd-1)=ud(Nud-1)/pow(wn+i*w0r,2)+i*pow(ud(Nud-1),2)/(2*(wn+i*w0r))-atan((ud(Nud-1)*wn)/(1+ud(Nud-1)*w0r))/pow(wn+i*w0r,3)-i*log(1+2*ud(Nud-1)*w0r+pow(ud(Nud-1),2)*(pow(wn,2)+pow(w0r,2)))/(2*pow(wn+i*w0r,3));
	Kb_d.col(Nintd-1)=i*ud(Nud-1)/(wn+i*w0r)-i*atan((ud(Nud-1)*wn)/(1+ud(Nud-1)*w0r))/pow(wn+i*w0r,2)+log(1+2*ud(Nud-1)*w0r+pow(ud(Nud-1),2)*(pow(wn,2)+pow(w0r,2)))/(2*pow(wn+i*w0r,2));
	 
	rowvec wd=1/ud2+w0r;
	mat Wd=ones<vec>(Nn)*wd;
	 
	if (use_HF_exp)
	{
		double utmp;
		int jn, p;
		for (j=0; j<Nud; j++)
		{
		 	utmp=ud2(j);
			jn=0;
			while (jn<Nn && abs(wn(jn)*utmp)<fd*abs(1+utmp*w0r)) jn++;
			while (jn<Nn && pow(wn(jn),pndmax)) jn++;
			if (jn<Nn)
		 	{
				Ka_d.submat(jn,j,Nn-1,j)=-(Ud.submat(jn,j+1,Nn-1,j+1)-Ud.submat(jn,j,Nn-1,j))/pow(wn.rows(jn,Nn-1)+i*w0r,2) - i*(pow(Ud.submat(jn,j+1,Nn-1,j+1),2)-pow(Ud.submat(jn,j,Nn-1,j),2))/(2*(wn.rows(jn,Nn-1)+i*w0r))+log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0r,3);
				Kb_d.submat(jn,j,Nn-1,j)=-i*(Ud.submat(jn,j+1,Nn-1,j+1)-Ud.submat(jn,j,Nn-1,j))/(wn.rows(jn,Nn-1)+i*w0r)+log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/pow(i*wn.rows(jn,Nn-1)-w0r,2);
				Kc_d.submat(jn,j,Nn-1,j)=log(Ud.submat(jn,j+1,Nn-1,j+1)/Ud.submat(jn,j,Nn-1,j))/(i*wn.rows(jn,Nn-1)-w0r);
				Kd_d.submat(jn,j,Nn-1,j)=zeros<cx_vec>(Nn-jn);
				for (p=pndmax; p>=1; p--)
			 	{
					Ka_d.submat(jn,j,Nn-1,j)=Ka_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0r,3);
					Kb_d.submat(jn,j,Nn-1,j)=Kb_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/pow(i*wn.rows(jn,Nn-1)-w0r,2);
					Kc_d.submat(jn,j,Nn-1,j)=Kc_d.submat(jn,j,Nn-1,j)+(pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p)))/(i*wn.rows(jn,Nn-1)-w0r);
					Kd_d.submat(jn,j,Nn-1,j)=Kd_d.submat(jn,j,Nn-1,j)+pow(i,p)*pow(-1,p+1)*(pow(Wd.submat(jn,j+1,Nn-1,j+1),p)-pow(Wd.submat(jn,j,Nn-1,j),p))/(p*pow(wn.rows(jn,Nn-1),p));
			 	}
		 	}
		}
	}
	
	Ka_d=-Ka_d/(2*PI);
	Kb_d=-Kb_d/(2*PI);
	Kc_d=-Kc_d/(2*PI);
	Kd_d=-Kd_d/(2*PI);
	

	cx_mat KG=(Ka_g*Pa_g+Kb_g*Pb_g+Kc_g*Pc_g+Kd_g*Pd_g)*MC;
	cx_mat KC=(Ka_c_r*Pa_c+Kb_c_r*Pb_c_r+Kc_c_r*Pc_c_r+Kd_c_r*Pd_c_r)*MC;
	cx_mat KD=(Ka_d*Pa_d+Kb_d*Pb_d+Kc_d*Pc_d+Kd_d*Pd_d)*MC;
	 
	Kcx=KG+KC+KD;


	rowvec Knorm_a_g=zeros<rowvec>(Nintg);
	rowvec Knorm_b_g=zeros<rowvec>(Nintg);
	rowvec Knorm_c_g=zeros<rowvec>(Nintg);
	rowvec Knorm_d_g=zeros<rowvec>(Nintg);
	 
	Knorm_a_g(0)=-pow(ug(0),2)/2;
	Knorm_b_g(0)=-ug(0);
	 
	Knorm_a_g.cols(1,Nintg-1)=-(pow(ug2.cols(1,Nug),2)-pow(ug2.cols(0,Nug-1),2))/2;
	Knorm_b_g.cols(1,Nintg-1)=-(ug2.cols(1,Nug)-ug2.cols(0,Nug-1));
	Knorm_c_g.cols(1,Nintg-1)=-log(ug2.cols(1,Nug)/ug2.cols(0,Nug-1));
	Knorm_d_g.cols(1,Nintg-1)=1.0/ug2.cols(1,Nug)-1.0/ug2.cols(0,Nug-1);
	 
	rowvec KM0g=(Knorm_a_g*Pa_g+Knorm_b_g*Pb_g+Knorm_c_g*Pc_g+Knorm_d_g*Pd_g)*MC/(2*PI);

	rowvec Knorm_a_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_b_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_c_c_r=zeros<rowvec>(Nintc);
	rowvec Knorm_d_c_r=zeros<rowvec>(Nintc);
	 
	Knorm_a_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),4)/4);
	Knorm_b_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),3)/3);
	Knorm_c_c_r=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),2)/2);
	Knorm_d_c_r=trans(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2));
	 
	rowvec KM0c=(Knorm_a_c_r*Pa_c+Knorm_b_c_r*Pb_c_r+Knorm_c_c_r*Pc_c_r+Knorm_d_c_r*Pd_c_r)*MC/(2*PI);

	rowvec Knorm_a_d=zeros<rowvec>(Nintd);
	rowvec Knorm_b_d=zeros<rowvec>(Nintd);
	rowvec Knorm_c_d=zeros<rowvec>(Nintd);
	rowvec Knorm_d_d=zeros<rowvec>(Nintd);
	
	Knorm_a_d.cols(0,Nintd-2)=-(pow(ud2.cols(1,Nud),2)-pow(ud2.cols(0,Nud-1),2))/2;
	Knorm_b_d.cols(0,Nintd-2)=-(ud2.cols(1,Nud)-ud2.cols(0,Nud-1));
	Knorm_c_d.cols(0,Nintd-2)=-log(ud2.cols(1,Nud)/ud2.cols(0,Nud-1));
	Knorm_d_d.cols(0,Nintd-2)=1.0/ud2.cols(1,Nud)-1.0/ud2.cols(0,Nud-1);
	
	Knorm_a_d(Nintd-1)=pow(ud2(Nud-1),2)/2;
	Knorm_b_d(Nintd-1)=ud2(Nud-1);
	
	rowvec KM0d=(Knorm_a_d*Pa_d+Knorm_b_d*Pb_d+Knorm_c_d*Pc_d+Knorm_d_d*Pd_d)*MC/(2*PI);
	
	rowvec KM0=KM0g+KM0c+KM0d;
	

	rowvec KM1_a_g=zeros<rowvec>(Nintg);
	rowvec KM1_b_g=zeros<rowvec>(Nintg);
	rowvec KM1_c_g=zeros<rowvec>(Nintg);
	rowvec KM1_d_g=zeros<rowvec>(Nintg);
	
	KM1_a_g.cols(1,Nintg-1)=Knorm_b_g.cols(1,Nintg-1);
	KM1_b_g.cols(1,Nintg-1)=Knorm_c_g.cols(1,Nintg-1);
	KM1_c_g.cols(1,Nintg-1)=Knorm_d_g.cols(1,Nintg-1);
	KM1_d_g.cols(1,Nintg-1)=(1.0/pow(ug2.cols(1,Nug),2)-1.0/pow(ug2.cols(0,Nug-1),2))/2;
	
	rowvec KM1g_tmp=(KM1_a_g*Pa_g+KM1_b_g*Pb_g+KM1_c_g*Pc_g+KM1_d_g*Pd_g)*MC/(2*PI);
	
	rowvec KM1g=w0l*KM0g+KM1g_tmp;

	mat Wjc=diagmat(wc.rows(0,Nintc-1));
	
	rowvec KM1_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),5)/5.0);
	
	rowvec KM1_a_c=KM1_a_c_tmp+Knorm_a_c_r*Wjc;
	rowvec KM1_b_c=Knorm_a_c_r+Knorm_b_c_r*Wjc;
	rowvec KM1_c_c=Knorm_b_c_r+Knorm_c_c_r*Wjc;
	rowvec KM1_d_c=Knorm_c_c_r+Knorm_d_c_r*Wjc;
	
	rowvec KM1c=(KM1_a_c*Pa_c+KM1_b_c*Pb_c_r+KM1_c_c*Pc_c_r+KM1_d_c*Pd_c_r)*MC/(2*PI);
	 
	rowvec KM1_a_d=zeros<rowvec>(Nintd);
	rowvec KM1_b_d=zeros<rowvec>(Nintd);
	rowvec KM1_c_d=zeros<rowvec>(Nintd);
	rowvec KM1_d_d=zeros<rowvec>(Nintd);
	 
	KM1_a_d.cols(0,Nintd-2)=Knorm_b_d.cols(0,Nintd-2);
	KM1_b_d.cols(0,Nintd-2)=Knorm_c_d.cols(0,Nintd-2);
	KM1_c_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM1_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),2)-1.0/pow(ud2.cols(0,Nud-1),2))/2;
	
	rowvec KM1d_tmp=(KM1_a_d*Pa_d+KM1_b_d*Pb_d+KM1_c_d*Pc_d+KM1_d_d*Pd_d)*MC/(2*PI);
	 
	rowvec KM1d=w0r*KM0d+KM1d_tmp;
	 
	rowvec KM1=KM1g+KM1c+KM1d;


	rowvec KM2_a_g=zeros<rowvec>(Nintg);
	rowvec KM2_b_g=zeros<rowvec>(Nintg);
	rowvec KM2_c_g=zeros<rowvec>(Nintg);
	rowvec KM2_d_g=zeros<rowvec>(Nintg);
	 
	KM2_a_g.cols(1,Nintg-1)=Knorm_c_g.cols(1,Nintg-1);
	KM2_b_g.cols(1,Nintg-1)=Knorm_d_g.cols(1,Nintg-1);
	KM2_c_g.cols(1,Nintg-1)=KM1_d_g.cols(1,Nintg-1);
	KM2_d_g.cols(1,Nintg-1)=(1.0/pow(ug2.cols(1,Nug),3)-1.0/pow(ug2.cols(0,Nug-1),3))/3;
	 
	rowvec KM2g_tmp=(KM2_a_g*Pa_g+KM2_b_g*Pb_g+KM2_c_g*Pc_g+KM2_d_g*Pd_g)*MC/(2*PI);
	 
	rowvec KM2g=pow(w0l,2)*KM0g+2*w0l*KM1g_tmp+KM2g_tmp;
	 
	rowvec KM2_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),6)/6);
	 
	rowvec KM2_a_c=KM2_a_c_tmp+2*KM1_a_c_tmp*Wjc+Knorm_a_c_r*pow(Wjc,2);
	rowvec KM2_b_c=KM1_a_c_tmp+2*Knorm_a_c_r*Wjc+Knorm_b_c_r*pow(Wjc,2);
	rowvec KM2_c_c=Knorm_a_c_r+2*Knorm_b_c_r*Wjc+Knorm_c_c_r*pow(Wjc,2);
	rowvec KM2_d_c=Knorm_b_c_r+2*Knorm_c_c_r*Wjc+Knorm_d_c_r*pow(Wjc,2);
	 
	rowvec KM2c=(KM2_a_c*Pa_c+KM2_b_c*Pb_c_r+KM2_c_c*Pc_c_r+KM2_d_c*Pd_c_r)*MC/(2*PI);
	 
	rowvec KM2_a_d=zeros<rowvec>(Nintd);
	rowvec KM2_b_d=zeros<rowvec>(Nintd);
	rowvec KM2_c_d=zeros<rowvec>(Nintd);
	rowvec KM2_d_d=zeros<rowvec>(Nintd);
	 
	KM2_a_d.cols(0,Nintd-2)=Knorm_c_d.cols(0,Nintd-2);
	KM2_b_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM2_c_d.cols(0,Nintd-2)=KM1_d_d.cols(0,Nintd-2);
	KM2_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),3)-1.0/pow(ud2.cols(0,Nud-1),3))/3;
	 
	rowvec KM2d_tmp=(KM2_a_d*Pa_d+KM2_b_d*Pb_d+KM2_c_d*Pc_d+KM2_d_d*Pd_d)*MC/(2*PI);
	 
	rowvec KM2d=pow(w0r,2)*KM0d+2*w0r*KM1d_tmp+KM2d_tmp;
	 
	rowvec KM2=KM2g+KM2c+KM2d;

	rowvec KM3_a_g=zeros<rowvec>(Nintg);
	rowvec KM3_b_g=zeros<rowvec>(Nintg);
	rowvec KM3_c_g=zeros<rowvec>(Nintg);
	rowvec KM3_d_g=zeros<rowvec>(Nintg);
	 
	KM3_a_g.cols(1,Nintg-1)=Knorm_d_g.cols(1,Nintg-1);
	KM3_b_g.cols(1,Nintg-1)=KM1_d_g.cols(1,Nintg-1);
	KM3_c_g.cols(1,Nintg-1)=KM2_d_g.cols(1,Nintg-1);
	KM3_d_g.cols(1,Nintg-1)=(1.0/pow(ug2.cols(1,Nug),4)-1.0/pow(ug2.cols(0,Nug-1),4))/4;
	 
	rowvec KM3g_tmp=(KM3_a_g*Pa_g+KM3_b_g*Pb_g+KM3_c_g*Pc_g+KM3_d_g*Pd_g)*MC/(2*PI);
	 
	rowvec KM3g=pow(w0l,3)*KM0g+3*pow(w0l,2)*KM1g_tmp+3*w0l*KM2g_tmp+KM3g_tmp;
	 
	rowvec KM3_a_c_tmp=trans(pow(wc.rows(1,Nwc-1)-wc.rows(0,Nwc-2),7)/7);
	 
	rowvec KM3_a_c=KM3_a_c_tmp+3*KM2_a_c_tmp*Wjc+3*KM1_a_c_tmp*pow(Wjc,2)+Knorm_a_c_r*pow(Wjc,3);
	rowvec KM3_b_c=KM2_a_c_tmp+3*KM1_a_c_tmp*Wjc+3*Knorm_a_c_r*pow(Wjc,2)+Knorm_b_c_r*pow(Wjc,3);
	rowvec KM3_c_c=KM1_a_c_tmp+3*Knorm_a_c_r*Wjc+3*Knorm_b_c_r*pow(Wjc,2)+Knorm_c_c_r*pow(Wjc,3);
	rowvec KM3_d_c=Knorm_a_c_r+3*Knorm_b_c_r*Wjc+3*Knorm_c_c_r*pow(Wjc,2)+Knorm_d_c_r*pow(Wjc,3);
	 
	rowvec KM3c=(KM3_a_c*Pa_c+KM3_b_c*Pb_c_r+KM3_c_c*Pc_c_r+KM3_d_c*Pd_c_r)*MC/(2*PI);
	 
	rowvec KM3_a_d=zeros<rowvec>(Nintd);
	rowvec KM3_b_d=zeros<rowvec>(Nintd);
	rowvec KM3_c_d=zeros<rowvec>(Nintd);
	rowvec KM3_d_d=zeros<rowvec>(Nintd);
	 
	KM3_a_d.cols(0,Nintd-2)=Knorm_d_d.cols(0,Nintd-2);
	KM3_b_d.cols(0,Nintd-2)=KM1_d_d.cols(0,Nintd-2);
	KM3_c_d.cols(0,Nintd-2)=KM2_d_d.cols(0,Nintd-2);
	KM3_d_d.cols(0,Nintd-2)=(1.0/pow(ud2.cols(1,Nud),4)-1.0/pow(ud2.cols(0,Nud-1),4))/4;
	 
	rowvec KM3d_tmp=(KM3_a_d*Pa_d+KM3_b_d*Pb_d+KM3_c_d*Pc_d+KM3_d_d*Pd_d)*MC/(2*PI);
	 
	rowvec KM3d=pow(w0r,3)*KM0d+3*pow(w0r,2)*KM1d_tmp+3*w0r*KM2d_tmp+KM3d_tmp;
	 
	rowvec KM3=KM3g+KM3c+KM3d;

	KM.row(0)=KM0;
	KM.row(1)=KM1;
	KM.row(2)=KM2;
	KM.row(3)=KM3;
	
	K.zeros(2*Nn,Nw);
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	K.rows(even_ind)=real(Kcx);
	K.rows(even_ind+1)=imag(Kcx);

/*
	//test K
//	rowvec x0={-3, 0, 2};
//	rowvec s0={0.5, 0.002, 0.3};
	
	rowvec x0={-2, 0, 3};
	rowvec s0={0.8, 0.3, 1};
	
	rowvec wgt={1,1,1};
	
	vec test_A;
	sum_gaussians(w, x0, s0, wgt, test_A);
	
	cx_vec Gtest=Kcx*test_A;
	vec Gr_test=real(Gtest);
	vec Gi_test=imag(Gtest);
	
	graph_2D g1, g2, g3, g4;
	char attr1[]="'o',color='r',markerfacecolor='none'";
	char attr2[]="'s',color='b',markerfacecolor='none'";
	char attr3[]="'.-',color='r'";
	char attr4[]="'.-',color='b'";
	
	g1.add_data(wn.memptr(), Gr.memptr(), Nn);
	g1.add_attribute(attr1);
	g1.add_data(wn.memptr(), Gr_test.memptr(), Nn);
	g1.add_attribute(attr2);
	g1.curve_plot();
	
	g2.add_data(wn.memptr(), Gi.memptr(), Nn);
	g2.add_attribute(attr1);
	g2.add_data(wn.memptr(), Gi_test.memptr(), Nn);
	g2.add_attribute(attr2);
	g2.curve_plot();
	
	plot(g3,wn,(Gr_test-Gr)/Gr,NULL,NULL,attr3);
	plot(g4,wn,(Gi_test-Gi)/Gi,NULL,NULL,attr4);
	
	graph_2D::show_figures();
*/
/*
//check moments
 	cout<<KM*test_A<<endl;
 
	void *par[5];
	par[0]=&w;
	par[1]=&Nw_lims;
	par[2]=&ws;
	par[3]=&coeffs1;
	j=1;
	par[4]=&j;
	
	fctPtr1 Ptr=static_cast<fctPtr1> (&OmegaMaxEnt_data::spline_val_G_part_int);
	double int_lims[2];
	double tol=1e-10;
	int nbEval[1];
	
	int_lims[0]=w(0);
	int_lims[1]=w(Nw_lims[0]);
	double M_g=quadInteg1D(Ptr, int_lims, tol, nbEval, par)/(2*PI);
	int_lims[0]=w(Nw_lims[0]);
	int_lims[1]=w(Nw_lims[1]);
	double M_c=quadInteg1D(Ptr, int_lims, tol, nbEval, par)/(2*PI);
	int_lims[0]=w(Nw_lims[1]);
	int_lims[1]=w(Nw-1);
	double M_d=quadInteg1D(Ptr, int_lims, tol, nbEval, par)/(2*PI);
	
	cout<<"Mg:\n"<<KM1g*test_A<<M_g<<endl;
	cout<<"Mc:\n"<<KM1c*test_A<<M_c<<endl;
	cout<<"Md:\n"<<KM1d*test_A<<M_d<<endl;
	cout<<"M: "<<M_g+M_c+M_d<<endl;
*/
	
//	cout<<"kernel matrix defined.\n";
	
	return true;
}

bool OmegaMaxEnt_data::set_A_ref()
{
	if (Aref_data.n_cols<2)
	{
		cout<<"number of columns in file "<<A_ref_file<<" is smaller than 2\n";
		return false;
	}
	
	w_ref=Aref_data.col(0);
	A_ref=Aref_data.col(1);
	
	/*
	if (displ_prep_figs)
	{
		graph_2D g1;
		char xl[]="$\\\\omega$";
		char yl[]="A_ref";
		char attr[]="marker='o', markeredgecolor='b'";
		plot(g1,w_ref,A_ref,xl,yl,attr);
		graph_2D::show_figures();
	}
	*/
	
	return true;
}

double OmegaMaxEnt_data::spline_val_G_part_int(double x, void *par[])
{
	vec *x0=reinterpret_cast<vec*>(par[0]);
	uvec *ind_xlims=reinterpret_cast<uvec*>(par[1]);
	vec *xs=reinterpret_cast<vec*>(par[2]);
	vec *coeffs=reinterpret_cast<vec*>(par[3]);
	int *p=reinterpret_cast<int*>(par[4]);
	
	return pow(x,p[0])*spline_val_G_part(x, *x0, *ind_xlims, *xs, *coeffs);
}

double OmegaMaxEnt_data::spline_val_G_part(double x, vec &x0, uvec &ind_xlims, vec &xs, vec &coeffs)
{
	int Nx0=x0.n_rows;
	
	double xl=x0(ind_xlims(0));
	double xr=x0(ind_xlims(1));
	
	double x0l=xs(0);
	double x0r=xs(1);
	
	double a,b,c,d, Dx, Du, sv;
	
	int l;
	sv=0;
	
	//if (x>=x0(0) && x<=x0(Nx0-1))
	//{
		if (x<xl)
		{
			l=0;
			while (x>=x0(l) && l<ind_xlims(0)) l++;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			if (l>0)
				Du=(x0(l-1)-x)/((x-x0l)*(x0(l-1)-x0l));
			else
				Du=1.0/(x-x0l);
			sv=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
		}
		else if (x>=xl && x<xr)
		{
			l=ind_xlims(0)+1;
			while (x>=x0(l) && l<ind_xlims(1)) l++;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			Dx=x-x0(l-1);
			sv=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
		}
		else
		{
			l=ind_xlims(1)+1;
			while (x>x0(l) && l<Nx0-1) l++;
			if (x>x0(Nx0-1)) l=Nx0;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			if (l<Nx0)
				Du=(x0(l)-x)/((x-x0r)*(x0(l)-x0r));
			else
				Du=1.0/(x-x0r);
			sv=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
		}
	//}
	
	return sv;
}

bool OmegaMaxEnt_data::spline_val_G_part_grid_transf(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv)
{
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;
	
	int Ncfs=4*(Nx0-1);
	if (coeffs.n_rows<Ncfs)
	{
		cout<<"spline_val_G_part_grid_transf(): number of elements in vectors \"coeffs\" and \"x0\" are not consistant\n";
		return false;
	}
	
	double xl=x0(ind_xlims(0));
	double xr=x0(ind_xlims(1));
	
	double x0l=xs(0);
	double x0r=xs(1);
	
	double a,b,c,d, Dx, Du;
	
	sv.zeros(Nx);
	
	int j, l;
	for (j=0; j<Nx; j++)
	{
		if (x(j)>=x0(0) && x(j)<=x0(Nx0-1))
		{
			if (x(j)<xl)
			{
				l=0;
				while (x(j)>=x0(l+1) && l<ind_xlims(0)-1) l++;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				
				Du=((x0(l+1)-x0l)/(x(j)-x0l))*((x0(l)-x(j))/(x0(l)-x0(l+1)));
				
				sv(j)=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
			}
			else if (x(j)>=xl && x(j)<=xr)
			{
				if (x(j)<xr)
				{
					l=ind_xlims(0);
					while (x(j)>=x0(l+1) && l<ind_xlims(1)-1) l++;
					a=coeffs(4*l);
					b=coeffs(4*l+1);
					c=coeffs(4*l+2);
					d=coeffs(4*l+3);
					Dx=(x(j)-x0(l))/(x0(l+1)-x0(l));
					sv(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
				}
				else
				{
					l=ind_xlims(1)-1;
					a=coeffs(4*l);
					b=coeffs(4*l+1);
					c=coeffs(4*l+2);
					d=coeffs(4*l+3);
					sv(j)=a+b+c+d;
				}
			}
			else
			{
				l=ind_xlims(1);
				while (x(j)>x0(l+1) && l<Nx0-2) l++;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				Du=((x0(l+1)-x0r)/(x(j)-x0r))*((x0(l)-x(j))/(x0(l)-x0(l+1)));
				sv(j)=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
			}
		}
	}
	
	return true;
}

bool OmegaMaxEnt_data::spline_val_G_part_grid_transf_1(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv)
{
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;
	
	int Ncfs=4*(Nx0-1);
	if (coeffs.n_rows<Ncfs)
	{
		cout<<"spline_val_G_part_grid_transf(): number of elements in vectors \"coeffs\" and \"x0\" are not consistant\n";
		return false;
	}
	
	double xl=x0(ind_xlims(0));
	double xr=x0(ind_xlims(1));
	
	double x0l=xs(0);
	double x0r=xs(1);
	
	double a,b,c,d, Dx, Du;
	
	sv.zeros(Nx);
	
	int j, l;
	for (j=0; j<Nx; j++)
	{
		if (x(j)>=x0(0) && x(j)<=x0(Nx0-1))
		{
			if (x(j)<xl)
			{
				l=0;
				while (x(j)>=x0(l+1) && l<ind_xlims(0)-1) l++;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				
				Du=((x0(l+1)-x0l)/(x(j)-x0l))*((x0(l)-x(j))/(x0(l)-x0(l+1)));
				
				sv(j)=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
			}
			else if (x(j)>=xl && x(j)<=xr)
			{
				if (x(j)<xr)
				{
					l=ind_xlims(0);
					while (x(j)>=x0(l+1) && l<ind_xlims(1)-1) l++;
					a=coeffs(4*l);
					b=coeffs(4*l+1);
					c=coeffs(4*l+2);
					d=coeffs(4*l+3);
					Dx=(x(j)-x0(l))/(x0(l+1)-x0(l));
					sv(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
				}
				else
				{
					l=ind_xlims(1)-1;
					a=coeffs(4*l);
					b=coeffs(4*l+1);
					c=coeffs(4*l+2);
					d=coeffs(4*l+3);
					sv(j)=a+b+c+d;
				}
			}
			else
			{
				l=ind_xlims(1);
				while (x(j)>x0(l+1) && l<Nx0-2) l++;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				Du=((x0(l)-x0r)/(x(j)-x0r))*((x0(l+1)-x(j))/(x0(l+1)-x0(l)));
				sv(j)=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
			}
		}
	}
	
	return true;
}

bool OmegaMaxEnt_data::spline_val_G_part(vec x, vec x0, uvec ind_xlims, vec xs, vec coeffs, vec &sv)
{
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;
	
	int Ncfs=4*(Nx0+1);
	if (coeffs.n_rows<Ncfs)
	{
		cout<<"spline_val_G_part(): number of elements in vectors \"coeffs\" and \"x0\" are not consistant\n";
		return false;
	}
	
//	int Ng=ind_xlims(0)+1;
//	int Nc=ind_xlims(1)-ind_xlims(0);
//	int Nd=Nx0-ind_xlims(1);
	
	double xl=x0(ind_xlims(0));
	double xr=x0(ind_xlims(1));
	
	double x0l=xs(0);
	double x0r=xs(1);
	
	double a,b,c,d, Dx, Du;
	
	sv.zeros(Nx);
	
	int j, l;
	for (j=0; j<Nx; j++)
	{
		//if (x(j)>=x0(0) && x(j)<=x0(Nx0-1))
		//{
			if (x(j)<xl)
			{
				l=0;
				while (x(j)>=x0(l) && l<ind_xlims(0)) l++;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				if (l>0)
					Du=(x0(l-1)-x(j))/((x(j)-x0l)*(x0(l-1)-x0l));
				else
					Du=1.0/(x(j)-x0l);
				sv(j)=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
			}
			else if (x(j)>=xl && x(j)<xr)
			{
				l=ind_xlims(0)+1;
				while (x(j)>=x0(l) && l<ind_xlims(1)) l++;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				Dx=x(j)-x0(l-1);
				sv(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
			}
			else
			{
				l=ind_xlims(1)+1;
				while (x(j)>x0(l) && l<Nx0-1) l++;
				if (x(j)>x0(Nx0-1)) l=Nx0;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				if (l<Nx0)
					Du=(x0(l)-x(j))/((x(j)-x0r)*(x0(l)-x0r));
				else
					Du=1.0/(x(j)-x0r);
				sv(j)=a*pow(Du,3)+b*pow(Du,2)+c*Du+d;
			}
		//}
	}
	
	return true;
}

bool OmegaMaxEnt_data::spline_G_omega_u(vec x, uvec ind_xlims, vec xs, vec F, vec &coeffs)
{
	int Nx=x.n_rows;
	
	if (F.n_rows!=Nx)
	{
		cout<<"spline_G() error: function vector and position vector do not have the same size.\n";
		return false;
	}
	
	int Ng=ind_xlims(0)+1;
	int Nc=ind_xlims(1)-ind_xlims(0);
	int Nd=Nx-ind_xlims(1);
	int Nint=Nx+1;
	
	double x0g=xs(0);
	double x0d=xs(1);
	
	//	 solve the spline in the interval -inf,wl]
	int NS=3*Nint-2;
	mat Aspl=zeros<mat>(NS,NS);
	
	vec D=zeros<vec>(NS);
	D(0)=F(0);
	
	double u=1.0/(x(0)-x0g);
	
	Aspl(0,0)=pow(u,3);
	Aspl(0,1)=pow(u,2);
	Aspl(1,0)=3*pow(u,2);
	Aspl(1,1)=2*u;
	Aspl(1,4)=-1;
	Aspl(2,0)=6*u;
	Aspl(2,1)=2;
	Aspl(2,3)=-2;
	
	int j;
	for (j=1; j<ind_xlims(0); j++)
	{
		u=1/(x(j)-x0g)-1/(x(j-1)-x0g);
	 
		D(3*j)=F(j)-F(j-1);
	 
		Aspl(3*j,3*j-1)=pow(u,3);
		Aspl(3*j,3*j)=pow(u,2);
		Aspl(3*j,3*j+1)=u;
	 
		Aspl(3*j+1,3*j-1)=3*pow(u,2);
		Aspl(3*j+1,3*j)=2*u;
		Aspl(3*j+1,3*j+1)=1;
		Aspl(3*j+1,3*j+4)=-1;
	 
		Aspl(3*j+2,3*j-1)=6*u;
		Aspl(3*j+2,3*j)=2;
		Aspl(3*j+2,3*j+3)=-2;
	}
	
	j=ind_xlims(0);
	
	u=1/(x(j)-x0g)-1/(x(j-1)-x0g);
	double ug=1/(x(j)-x0g);
	
	D(3*j)=F(j)-F(j-1);
	Aspl(3*j,3*j-1)=pow(u,3);
	Aspl(3*j,3*j)=pow(u,2);
	Aspl(3*j,3*j+1)=u;
	
	Aspl(3*j+1,3*j-1)=-3*pow(ug,2)*pow(u,2);
	Aspl(3*j+1,3*j)=-2*pow(ug,2)*u;
	Aspl(3*j+1,3*j+1)=-pow(ug,2);
	Aspl(3*j+1,3*j+4)=-1;
	
	Aspl(3*j+2,3*j-1)=6*pow(ug,3)*pow(u,2)+6*pow(ug,4)*u;
	Aspl(3*j+2,3*j)=4*pow(ug,3)*u+2*pow(ug,4);
	Aspl(3*j+2,3*j+1)=2*pow(ug,3);
	Aspl(3*j+2,3*j+3)=-2;
	
	double Dx;
	for (j=ind_xlims(0)+1; j<ind_xlims(1); j++)
	{
		Dx=x(j)-x(j-1);
	 
		D(3*j)=F(j)-F(j-1);
	 
		Aspl(3*j,3*j-1)=pow(Dx,3);
		Aspl(3*j,3*j)=pow(Dx,2);
		Aspl(3*j,3*j+1)=Dx;
	 
		Aspl(3*j+1,3*j-1)=3*pow(Dx,2);
		Aspl(3*j+1,3*j)=2*Dx;
		Aspl(3*j+1,3*j+1)=1;
		Aspl(3*j+1,3*j+4)=-1;
	 
		Aspl(3*j+2,3*j-1)=6*Dx;
		Aspl(3*j+2,3*j)=2;
		Aspl(3*j+2,3*j+3)=-2;
	}
	
	j=ind_xlims(1);
	
	Dx=x(j)-x(j-1);
	
	D(3*j)=F(j)-F(j-1);
	Aspl(3*j,3*j-1)=pow(Dx,3);
	Aspl(3*j,3*j)=pow(Dx,2);
	Aspl(3*j,3*j+1)=Dx;
	
	u=1/(x(j)-x0d)-1/(x(j+1)-x0d);
	double ud=1/(x(j)-x0d);
	
	Aspl(3*j+1,3*j-1)=3*pow(Dx,2);
	Aspl(3*j+1,3*j)=2*Dx;
	Aspl(3*j+1,3*j+1)=1;
	Aspl(3*j+1,3*j+2)=3*pow(ud,2)*pow(u,2);
	Aspl(3*j+1,3*j+3)=2*pow(ud,2)*u;
	Aspl(3*j+1,3*j+4)=pow(ud,2);
	
	Aspl(3*j+2,3*j-1)=6*Dx;
	Aspl(3*j+2,3*j)=2;
	Aspl(3*j+2,3*j+2)=-6*pow(ud,3)*pow(u,2)-6*pow(ud,4)*u;
	Aspl(3*j+2,3*j+3)=-4*pow(ud,3)*u-2*pow(ud,4);
	Aspl(3*j+2,3*j+4)=-2*pow(ud,3);
	
	D(3*j+3)=F(j)-F(j+1);
	Aspl(3*j+3,3*j+2)=pow(u,3);
	Aspl(3*j+3,3*j+3)=pow(u,2);
	Aspl(3*j+3,3*j+4)=u;
	
	for (j=ind_xlims(1)+1; j<Nx-1; j++)
	{
		u=1/(x(j)-x0d)-1/(x(j+1)-x0d);
		
		Aspl(3*j+1,3*j+1)=-1;
		Aspl(3*j+1,3*j+2)=3*pow(u,2);
		Aspl(3*j+1,3*j+3)=2*u;
		Aspl(3*j+1,3*j+4)=1;
		
		Aspl(3*j+2,3*j)=-2;
		Aspl(3*j+2,3*j+2)=6*u;
		Aspl(3*j+2,3*j+3)=2;
		
		D(3*j+3)=F(j)-F(j+1);
		Aspl(3*j+3,3*j+2)=pow(u,3);
		Aspl(3*j+3,3*j+3)=pow(u,2);
		Aspl(3*j+3,3*j+4)=u;
	}
	
	j=Nx-1;
	u=1/(x(j)-x0d);
	
	Aspl(3*j+1,3*j+1)=-1;
	Aspl(3*j+1,3*j+2)=3*pow(u,2);
	Aspl(3*j+1,3*j+3)=2*u;
	
	Aspl(3*j+2,3*j)=-2;
	Aspl(3*j+2,3*j+2)=6*u;
	Aspl(3*j+2,3*j+3)=2;
	
	D(3*j+3)=F(j);
	Aspl(3*j+3,3*j+2)=pow(u,3);
	Aspl(3*j+3,3*j+3)=pow(u,2);
	
	vec C=solve(Aspl,D);
	
	coeffs.zeros(4*(Nx+1));
	
	coeffs(0)=C(0);
	coeffs(1)=C(1);
	
	uvec ind_tmp=linspace<uvec>(1,Nx-1,Nx-1);
	coeffs.rows(4*ind_tmp)=C(3*ind_tmp-1);
	coeffs.rows(4*ind_tmp+1)=C(3*ind_tmp);
	coeffs.rows(4*ind_tmp+2)=C(3*ind_tmp+1);
	
	ind_tmp=linspace<uvec>(1,ind_xlims(1),ind_xlims(1));
	
	coeffs.rows(4*ind_tmp+3)=F.rows(ind_tmp-1);
	
	ind_tmp=linspace<uvec>(ind_xlims(1)+1,Nx-1,Nx-ind_xlims(1)-1);
	coeffs.rows(4*ind_tmp+3)=F.rows(ind_tmp);
	
	coeffs(4*Nx)=C(3*Nx-1);
	coeffs(4*Nx+1)=C(3*Nx);
	
	return true;
}

bool OmegaMaxEnt_data::spline_G_part(vec x, uvec ind_xlims, vec xs, vec F, vec &coeffs)
{
	int Nx=x.n_rows;
	
	if (F.n_rows!=Nx)
	{
		cout<<"spline_G_part() error: function vector and position vector do not have the same size.\n";
		return false;
	}
	
	int Ng=ind_xlims(0)+1;
	int Nc=ind_xlims(1)-ind_xlims(0);
	int Nd=Nx-ind_xlims(1);
	
	double x0g=xs(0);
	double x0d=xs(1);
	
	//	 solve the spline in the interval -inf,wl]
	int NSg=3*Ng-1;
	mat Aspl=zeros<mat>(NSg,NSg);
	
	vec Dg=zeros<vec>(NSg);
	Dg(0)=F(0);
	
	double u=1.0/(x(0)-x0g);
	
	Aspl(0,0)=pow(u,3);
	Aspl(0,1)=pow(u,2);
	Aspl(1,0)=3*pow(u,2);
	Aspl(1,1)=2*u;
	Aspl(1,4)=-1;
	Aspl(2,0)=6*u;
	Aspl(2,1)=2;
	Aspl(2,3)=-2;
	
	int j;
	for (j=2; j<Ng; j++)
	{
		u=1/(x(j-1)-x0g)-1/(x(j-2)-x0g);
	 
		Dg(3*j-3)=F(j-1)-F(j-2);
	 
		Aspl(3*j-3,3*j-4)=pow(u,3);
		Aspl(3*j-3,3*j-3)=pow(u,2);
		Aspl(3*j-3,3*j-2)=u;
	 
		Aspl(3*j-2,3*j-4)=3*pow(u,2);
		Aspl(3*j-2,3*j-3)=2*u;
		Aspl(3*j-2,3*j-2)=1;
		Aspl(3*j-2,3*j+1)=-1;
	 
		Aspl(3*j-1,3*j-4)=6*u;
		Aspl(3*j-1,3*j-3)=2;
		Aspl(3*j-1,3*j)=-2;
	}
	
	j=ind_xlims(0)+1;
	Dg(3*j-3)=F(j-1)-F(j-2);
	u=1/(x(j-1)-x0g)-1/(x(j-2)-x0g);
	double ug=1/(x(j-1)-x0g);
	
	Aspl(3*j-3,3*j-4)=pow(u,3);
	Aspl(3*j-3,3*j-3)=pow(u,2);
	Aspl(3*j-3,3*j-2)=u;
	
	Aspl(3*j-2,3*j-4)=-3*pow(ug,2)*pow(u,2);
	Aspl(3*j-2,3*j-3)=-2*pow(ug,2)*u;
	Aspl(3*j-2,3*j-2)=-pow(ug,2);
	
	Dg(3*j-2)=(F(j)-F(j-2))/(x(j)-x(j-2));
	
	vec Cg=solve(Aspl,Dg);
/*
	mat T(NSg,Nx);
	T.zeros();
	T(0,0)=1;
	for (j=2; j<=Ng; j++)
	{
	 T(3*j-3,j-2)=-1;
	 T(3*j-3,j-1)=1;
	}
	T(3*Ng-2,Ng-2)=-1/(x(Ng)-x(Ng-2));
	T(3*Ng-2,Ng)=1/(x(Ng)-x(Ng-2));
	
	Cg=Cg*T;
*/
	
	//solve the spline in the interval [wl,wr]
	int NSc=3*Nc-1;
	Aspl.zeros(NSc,NSc);
	
	double Dx=x(Ng)-x(Ng-1);
	
	vec Dc=zeros<vec>(NSc);
	Dc(0)=F(Ng)-Dx*(F(Ng)-F(Ng-2))/(x(Ng)-x(Ng-2))-F(Ng-1);
	
	Aspl(0,0)=pow(Dx,3);
	Aspl(0,1)=pow(Dx,2);
	
	Dc(1)=-(F(Ng)-F(Ng-2))/(x(Ng)-x(Ng-2));
	
	Aspl(1,0)=3*pow(Dx,2);
	Aspl(1,1)=2*Dx;
	Aspl(1,4)=-1;
	
	Aspl(2,0)=6*Dx;
	Aspl(2,1)=2;
	Aspl(2,3)=-2;
	
	for (j=2; j<Nc; j++)
	{
		Dx=x(Ng+j-1)-x(Ng+j-2);
	 
		Dc(3*j-3)=F(Ng+j-1)-F(Ng+j-2);
	 
		Aspl(3*j-3,3*j-4)=pow(Dx,3);
		Aspl(3*j-3,3*j-3)=pow(Dx,2);
		Aspl(3*j-3,3*j-2)=Dx;
	 
		Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
		Aspl(3*j-2,3*j-3)=2*Dx;
		Aspl(3*j-2,3*j-2)=1;
		Aspl(3*j-2,3*j+1)=-1;
	 
		Aspl(3*j-1,3*j-4)=6*Dx;
		Aspl(3*j-1,3*j-3)=2;
		Aspl(3*j-1,3*j)=-2;
	}
	
	j=Nc;
	Dx=x(Ng+j-1)-x(Ng+j-2);
	Dc(3*j-3)=F(Ng+j-1)-F(Ng+j-2);
	Aspl(3*j-3,3*j-4)=pow(Dx,3);
	Aspl(3*j-3,3*j-3)=pow(Dx,2);
	Aspl(3*j-3,3*j-2)=Dx;
	
	Dc(3*j-2)=(F(Ng+j)-F(Ng+j-2))/(x(Ng+j)-x(Ng+j-2));
	Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
	Aspl(3*j-2,3*j-3)=2*Dx;
	Aspl(3*j-2,3*j-2)=1;
	
	vec Cc=solve(Aspl,Dc);
	
/*
	T.zeros(NSc,Nx);
	T(0,Ng-2)=(x(Ng)-x(Ng-1))/(x(Ng)-x(Ng-2));
	T(0,Ng-1)=-1;
	T(0,Ng)=1-(x(Ng)-x(Ng-1))/(x(Ng)-x(Ng-2));
	T(1,Ng-2)=1/(x(Ng)-x(Ng-2));
	T(1,Ng)=-1/(x(Ng)-x(Ng-2));
	for (j=2; j<=Nc; j++)
	{
		T(3*j-3,Ng+j-2)=-1;
		T(3*j-3,Ng+j-1)=1;
	}
	T(3*Nc-2,ind_xlims(1)-1)=-1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	T(3*Nc-2,ind_xlims(1)+1)=1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	
	Cc=Cc*T;
	
	mat Cc2(NSc+1,Nx);
	Cc2.zeros();
	
	Cc2.rows(0,1)=Cc.rows(0,1);
	Cc2.rows(3,NSc)=Cc.rows(2,NSc-1);
	Cc2(2,Ng-2)=-1/(x(Ng)-x(Ng-2));
	Cc2(2,Ng)=1/(x(Ng)-x(Ng-2));
*/
	
	//solve the spline in the interval [wr,inf
	int NSd=3*Nd-1;
	Aspl.zeros(NSd,NSd);
	vec Dd=zeros<vec>(NSd);
	
	u=1/(x(ind_xlims(1))-x0d)-1/(x(ind_xlims(1)+1)-x0d);
	double ud=1/(x(ind_xlims(1))-x0d);
	
	Dd(0)=(F(ind_xlims(1)+1)-F(ind_xlims(1)-1))/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	Aspl(0,0)=-3*pow(ud,2)*pow(u,2);
	Aspl(0,1)=-2*pow(ud,2)*u;
	Aspl(0,2)=-pow(ud,2);
	
	Dd(1)=F(ind_xlims(1))-F(ind_xlims(1)+1);
	Aspl(1,0)=pow(u,3);
	Aspl(1,1)=pow(u,2);
	Aspl(1,2)=u;
	
	for (j=1; j<Nd-1; j++)
	{
		u=1/(x(ind_xlims(1)+j)-x0d)-1/(x(ind_xlims(1)+j+1)-x0d);
		
		Aspl(3*j-1,3*j-1)=-1;
		Aspl(3*j-1,3*j)=3*pow(u,2);
		Aspl(3*j-1,3*j+1)=2*u;
		Aspl(3*j-1,3*j+2)=1;
		
		Aspl(3*j,3*j-2)=-2;
		Aspl(3*j,3*j)=6*u;
		Aspl(3*j,3*j+1)=2;
	 
		Dd(3*j+1)=F(ind_xlims(1)+j)-F(ind_xlims(1)+j+1);
		Aspl(3*j+1,3*j)=pow(u,3);
		Aspl(3*j+1,3*j+1)=pow(u,2);
		Aspl(3*j+1,3*j+2)=u;
	}
	
	j=Nd-1;
	u=1/(x(ind_xlims(1)+j)-x0d);
	
	Aspl(3*j-1,3*j-1)=-1;
	Aspl(3*j-1,3*j)=3*pow(u,2);
	Aspl(3*j-1,3*j+1)=2*u;
	
	Aspl(3*j,3*j-2)=-2;
	Aspl(3*j,3*j)=6*u;
	Aspl(3*j,3*j+1)=2;
	
	Dd(3*j+1)=F(ind_xlims(1)+j);
	Aspl(3*j+1,3*j)=pow(u,3);
	Aspl(3*j+1,3*j+1)=pow(u,2);
	
	vec Cd=solve(Aspl,Dd);
	
	int Nint_tot=Nx+1;
	int NS_tot=4*Nint_tot;
	coeffs.zeros(NS_tot);
	
	coeffs(0)=Cg(0);
	coeffs(1)=Cg(1);
	
	uvec ind_tmp=linspace<uvec>(1,Ng-1,Ng-1);
	coeffs.rows(4*ind_tmp)=Cg(3*ind_tmp-1);
	coeffs.rows(4*ind_tmp+1)=Cg(3*ind_tmp);
	coeffs.rows(4*ind_tmp+2)=Cg(3*ind_tmp+1);
	coeffs.rows(4*ind_tmp+3)=F.rows(0,ind_xlims(0)-1);
	
	coeffs(4*Ng)=Cc(0);
	coeffs(4*Ng+1)=Cc(1);
	coeffs(4*Ng+2)=(F(Ng)-F(Ng-2))/(x(Ng)-x(Ng-2));
	coeffs(4*Ng+3)=F(ind_xlims(0));
	
	ind_tmp=linspace<uvec>(1,Nc-1,Nc-1);
	coeffs.rows(4*ind_tmp+4*Ng)=Cc(3*ind_tmp-1);
	coeffs.rows(4*ind_tmp+4*Ng+1)=Cc(3*ind_tmp);
	coeffs.rows(4*ind_tmp+4*Ng+2)=Cc(3*ind_tmp+1);
	coeffs.rows(4*ind_tmp+4*Ng+3)=F.rows(ind_xlims(0)+1,ind_xlims(1)-1);
	
	ind_tmp=linspace<uvec>(0,Nd-2,Nd-1);
	coeffs.rows(4*ind_tmp+4*(Ng+Nc))=Cd(3*ind_tmp);
	coeffs.rows(4*ind_tmp+4*(Ng+Nc)+1)=Cd(3*ind_tmp+1);
	coeffs.rows(4*ind_tmp+4*(Ng+Nc)+2)=Cd(3*ind_tmp+2);
	coeffs.rows(4*ind_tmp+4*(Ng+Nc)+3)=F.rows(ind_xlims(1)+1,Nx-1);
	
	coeffs(NS_tot-4)=Cd(3*Nd-3);
	coeffs(NS_tot-3)=Cd(3*Nd-2);
	
/*
	T.zeros(NSd,Nx);
	T(0,ind_xlims(1)-1)=-1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	T(0,ind_xlims(1)+1)=1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	for (j=1; j<Nd; j++)
	{
		T(3*j-2,ind_xlims(1)+j-1)=1;
		T(3*j-2,ind_xlims(1)+j)=-1;
	}
	j=Nd;
	T(3*j-2,ind_xlims(1)+j-1)=1;
	
	Cd=Cd*T;
	
	int NS_tot=NSg+NSc+NSd+1;
	
	Mspl.zeros(NS_tot,Nx);
	Mspl.rows(0,NSg-1)=Cg;
	Mspl.rows(NSg,NSg+NSc)=Cc2;
	Mspl.rows(NSg+NSc+1,NS_tot-1)=Cd;
*/
	return true;
}

bool OmegaMaxEnt_data::spline_val_grid_transf(vec x, vec x0, vec coeffs, vec &s)
{
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;
	int Ns=Nx0-1;
	int Nc=4*Ns;
	
	vec D=1.0/(x0.rows(1,Nx0-1)-x0.rows(0,Nx0-2));
	
	if (coeffs.n_rows<Nc)
	{
		cout<<"spline_val_grid_transf(): either the coefficients vector is incomplete or the position vector is too long\n";
		return false;
	}
	
	s.zeros(Nx);
	
	double a, b, c, d, Dx;
	
	int j, l;
	for (j=0; j<Nx; j++)
	{
		if (x(j)>=x0(0) && x(j)<x0(Nx0-1))
		{
			l=0;
			while ( x(j)>=x0(l+1) && l<Ns-1) l++;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			
			Dx=D(l)*(x(j)-x0(l));
			s(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
		}
		else if (x(j)==x0(Nx0-1))
		{
			if (coeffs.n_rows>Nc)
			{
				s(j)=coeffs[Nc];
			}
			else
			{
				l=Ns-1;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				
				s(j)=a+b+c+d;
			}
		}
		else
		{
			cout<<"spline_val_grid_transf(): the provided position vector has a value outside the boundaries of the spline.\n";
			return false;
		}
	}
	
	return true;
}

bool OmegaMaxEnt_data::spline_matrix_grid_transf(vec w0, mat &M)
{
	int N=w0.n_rows;
	vec D=1.0/(w0.rows(1,N-1)-w0.rows(0,N-2));
	
	int Nint=N-1;
	
	mat B=zeros<mat>(3*Nint-1,3*Nint-1);
	//vec vTl(3*Nint-1), vTr(3*Nint-1);
	mat Ps=zeros<mat>(3*Nint-1, N);
	mat Pg=zeros<mat>(4*Nint,4*Nint-1);
	
	//double N1p5=pow(1.0*Nint,1.5), N0p5=pow(1.0*Nint,0.5), Nm0p5=pow(1.0*Nint,-0.5);
	
	B(0,0)=1;
	B(0,1)=1;
	B(1,0)=3;
	B(1,1)=2;
	B(1,4)=-D(1)/D(0);
	B(2,0)=6;
	B(2,1)=2;
	B(2,3)=-2*pow(D(1)/D(0),2);
	
//	vTl(0)=N1p5;
//	vTl(1)=N0p5;
//	vTl(2)=Nm0p5;
	
//	vTr(0)=N1p5;
//	vTr(1)=N0p5;
	
	Ps(0,0)=-1;
	Ps(0,1)=1;
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(3,3*Nint-1)=1;
	
	int j;
	for (j=1; j<Nint-1; j++)
	{
		B(3*j,3*j-1)=1;
		B(3*j,3*j)=1;
		B(3*j,3*j+1)=1;
		B(3*j+1,3*j-1)=3;
		B(3*j+1,3*j)=2;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+4)=-D(j+1)/D(j);
		B(3*j+2,3*j-1)=6;
		B(3*j+2,3*j)=2;
		B(3*j+2,3*j+3)=-2*pow(D(j+1)/D(j),2);
		
//		vTl(3*j)=N1p5;
//		vTl(3*j+1)=N0p5;
//		vTl(3*j+2)=Nm0p5;
		
//		vTr(3*j-1)=N1p5;
//		vTr(3*j)=N0p5;
//		vTr(3*j+1)=Nm0p5;
		
		Ps(3*j,j)=-1;
		Ps(3*j,j+1)=1;
		
		Pg(4*j,3*j-1)=1;
		Pg(4*j+1,3*j)=1;
		Pg(4*j+2,3*j+1)=1;
		Pg(4*j+3,3*Nint-1+j)=1;
	}
	
	j=Nint-1;
	B(3*j,3*j-1)=1;
	B(3*j,3*j)=1;
	B(3*j,3*j+1)=1;
	B(3*j+1,3*j-1)=3;
	B(3*j+1,3*j)=2;
	B(3*j+1,3*j+1)=1;
	
//	vTl(3*j)=N1p5;
//	vTl(3*j+1)=N0p5;
	
//	vTr(3*j-1)=N1p5;
//	vTr(3*j)=N0p5;
//	vTr(3*j+1)=Nm0p5;
	
	Ps(3*j,j)=-1;
	Ps(3*j,j+1)=1;
	
	Pg(4*j,3*j-1)=1;
	Pg(4*j+1,3*j)=1;
	Pg(4*j+2,3*j+1)=1;
	Pg(4*j+3,3*Nint-1+j)=1;
	
//	mat Tl=diagmat(vTl);
//	mat Tr=diagmat(vTr);
	
	mat IB=eye(3*Nint-1,3*Nint-1);
	mat invB=solve(B,IB);
	
	mat IA=eye(N,N);
	mat PA=IA.submat(0,0,N-2,N-1);
	
	mat Lg=join_vert(invB*Ps,PA);
	
//	mat Lg=join_vert(Tr*invB*Tl*Ps,PA);
	
	M=Pg*Lg;
	
	return true;
}

bool OmegaMaxEnt_data::spline_matrix_grid_transf_G_part_2(vec x, uvec ind_xlims, vec xs, mat &M)
{
	int Nx=x.n_rows;
	
	int Ng=ind_xlims(0);
	int Nc=ind_xlims(1)-ind_xlims(0);
	int Nd=Nx-ind_xlims(1)-1;
	
	double x0g=xs(0);
	double x0d=xs(1);
	
	//	 solve the spline in the interval -inf,wl]
	//vec Dg=-((x.rows(1,ind_xlims(0))-x0g)%(x.rows(0,ind_xlims(0)-1)-x0g))/(x.rows(1,ind_xlims(0))-x.rows(0,ind_xlims(0)-1));
	
	vec RDg=((x.rows(2,ind_xlims(0))-x0g)/(x.rows(0,ind_xlims(0)-2)-x0g))%((x.rows(1,ind_xlims(0)-1)-x.rows(0,ind_xlims(0)-2))/(x.rows(2,ind_xlims(0))-x.rows(1,ind_xlims(0)-1)));
	
	int NCg=3*Ng-1;
	mat B=zeros<mat>(NCg,NCg);
	mat Ps=zeros<mat>(NCg, Nx);
	mat Pg=zeros<mat>(4*Ng,4*Ng-1);
	
	B(0,0)=1;
	B(0,1)=1;
	B(1,0)=3;
	B(1,1)=2;
	B(1,4)=-RDg(0);
	B(2,0)=6;
	B(2,1)=2;
	B(2,3)=-2*pow(RDg(0),2);
	
	Ps(0,0)=-1;
	Ps(0,1)=1;
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(3,NCg)=1;
	
	int j;
	for (j=1; j<Ng-1; j++)
	{
		B(3*j,3*j-1)=1;
		B(3*j,3*j)=1;
		B(3*j,3*j+1)=1;
		B(3*j+1,3*j-1)=3;
		B(3*j+1,3*j)=2;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+4)=-RDg(j);
		B(3*j+2,3*j-1)=6;
		B(3*j+2,3*j)=2;
		B(3*j+2,3*j+3)=-2*pow(RDg(j),2);
		
		Ps(3*j,j)=-1;
		Ps(3*j,j+1)=1;
		
		Pg(4*j,3*j-1)=1;
		Pg(4*j+1,3*j)=1;
		Pg(4*j+2,3*j+1)=1;
		Pg(4*j+3,NCg+j)=1;
	}
	
	j=Ng-1;
	B(3*j,3*j-1)=1;
	B(3*j,3*j)=1;
	B(3*j,3*j+1)=1;
	B(3*j+1,3*j-1)=3;
	B(3*j+1,3*j)=2;
	B(3*j+1,3*j+1)=1;
	
	double fdAg=((x(j+1)-x0g)/(x(j)-x0g))*((x(j+1)-x(j))/(x(j+2)-x(j)));
	
	Ps(3*j,j)=-1;
	Ps(3*j,j+1)=1;
	Ps(3*j+1,j)=-fdAg;
	Ps(3*j+1,j+2)=fdAg;
	
	Pg(4*j,3*j-1)=1;
	Pg(4*j+1,3*j)=1;
	Pg(4*j+2,3*j+1)=1;
	Pg(4*j+3,NCg+j)=1;
	
	mat IB=eye(NCg,NCg);
	mat invB=solve(B,IB);
	
	mat IA=eye(Nx,Nx);
	mat PA=IA.submat(0,0,Ng-1,Nx-1);
	
	mat Lg=join_vert(invB*Ps,PA);
	
	mat Mg=Pg*Lg;
	
	// solve in the interval [wl,wr]
	//vec Dc=1.0/(x.rows(ind_xlims(0)+1,ind_xlims(1))-x.rows(ind_xlims(0),ind_xlims(1)-1));
	
	vec RDc=(x.rows(ind_xlims(0)+1,ind_xlims(1)-1)-x.rows(ind_xlims(0),ind_xlims(1)-2))/(x.rows(ind_xlims(0)+2,ind_xlims(1))-x.rows(ind_xlims(0)+1,ind_xlims(1)-1));
	
	int NCc=3*Nc-1;
	
	B.zeros(NCc,NCc);
	Ps.zeros(NCc, Nx);
	Pg.zeros(4*Nc,4*Nc);
	
	B(0,0)=1;
	B(0,1)=1;
	B(1,0)=3;
	B(1,1)=2;
	B(1,4)=-RDc(0);
	B(2,0)=6;
	B(2,1)=2;
	B(2,3)=-2*pow(RDc(0),2);
	
	Ps(0,Ng-1)=(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Ps(0,Ng)=-1;
	Ps(0,Ng+1)=1-(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Ps(1,Ng-1)=(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Ps(1,Ng+1)=-(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(2,NCc)=-(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Pg(2,NCc+2)=(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Pg(3,NCc+1)=1;
	
	for (j=1; j<Nc-1; j++)
	{
		B(3*j,3*j-1)=1;
		B(3*j,3*j)=1;
		B(3*j,3*j+1)=1;
		B(3*j+1,3*j-1)=3;
		B(3*j+1,3*j)=2;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+4)=-RDc(j);
		B(3*j+2,3*j-1)=6;
		B(3*j+2,3*j)=2;
		B(3*j+2,3*j+3)=-2*pow(RDc(j),2);
		
		Ps(3*j,Ng+j)=-1;
		Ps(3*j,Ng+j+1)=1;
		
		Pg(4*j,3*j-1)=1;
		Pg(4*j+1,3*j)=1;
		Pg(4*j+2,3*j+1)=1;
		Pg(4*j+3,NCc+j+1)=1;
	}
	
	j=Nc-1;
	B(3*j,3*j-1)=1;
	B(3*j,3*j)=1;
	B(3*j,3*j+1)=1;
	B(3*j+1,3*j-1)=3;
	B(3*j+1,3*j)=2;
	B(3*j+1,3*j+1)=1;
	
	Ps(3*j,Ng+j)=-1;
	Ps(3*j,Ng+j+1)=1;
	Ps(3*j+1,Ng+j)=-(x(ind_xlims(1))-x(ind_xlims(1)-1))/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	Ps(3*j+1,Ng+j+2)=(x(ind_xlims(1))-x(ind_xlims(1)-1))/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	
	Pg(4*j,3*j-1)=1;
	Pg(4*j+1,3*j)=1;
	Pg(4*j+2,3*j+1)=1;
	Pg(4*j+3,NCc+j+1)=1;
	
	IB.eye(NCc,NCc);
	invB=solve(B,IB);
	
	PA=IA.submat(Ng-1,0,ind_xlims(1)-1,Nx-1);
	
	Lg=join_vert(invB*Ps,PA);
	
	mat Mc=Pg*Lg;
	
	int NCd=3*Nd-1;
	
	//vec RDd=((x.rows(ind_xlims(1)+2,Nx-1)-x.rows(ind_xlims(1)+1,Nx-2))/(x.rows(ind_xlims(1)+1,Nx-2)-x.rows(ind_xlims(1),Nx-3)))%((x.rows(ind_xlims(1),Nx-3)-x0d)/(x.rows(ind_xlims(1)+2,Nx-1)-x0d));
	
	double fdAd=((x(ind_xlims(1)+1)-x(ind_xlims(1)))/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1)))*((x(ind_xlims(1))-x0d)/(x(ind_xlims(1)+1)-x0d));
	
	vec RDd=((x.rows(ind_xlims(1)+2,Nx-1)-x0d)/(x.rows(ind_xlims(1),Nx-3)-x0d))%((x.rows(ind_xlims(1),Nx-3)-x.rows(ind_xlims(1)+1,Nx-2))/(x.rows(ind_xlims(1)+1,Nx-2)-x.rows(ind_xlims(1)+2,Nx-1)));
	
	B.zeros(NCd,NCd);
	Ps.zeros(NCd, Nx);
	Pg.zeros(4*Nd,4*Nd);
	
	B(0,0)=1;
	B(0,1)=1;
	B(1,0)=3;
	B(1,1)=2;
	B(1,4)=-RDd(0);
	B(2,0)=6;
	B(2,1)=2;
	B(2,3)=-2*pow(RDd(0),2);
	
	Ps(0,ind_xlims(1)-1)=fdAd;
	Ps(0,ind_xlims(1))=-1;
	Ps(0,ind_xlims(1)+1)=1-fdAd;
	Ps(1,ind_xlims(1)-1)=fdAd;
	Ps(1,ind_xlims(1)+1)=-fdAd;
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(2,NCd)=-fdAd;
	Pg(2,NCd+2)=fdAd;
	Pg(3,NCd+1)=1;
	
	for (j=1; j<Nd-1; j++)
	{
		B(3*j,3*j-1)=1;
		B(3*j,3*j)=1;
		B(3*j,3*j+1)=1;
		B(3*j+1,3*j-1)=3;
		B(3*j+1,3*j)=2;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+4)=-RDd(j);
		B(3*j+2,3*j-1)=6;
		B(3*j+2,3*j)=2;
		B(3*j+2,3*j+3)=-2*pow(RDd(j),2);
		
		Ps(3*j,ind_xlims(1)+j)=-1;
		Ps(3*j,ind_xlims(1)+j+1)=1;
		
		Pg(4*j,3*j-1)=1;
		Pg(4*j+1,3*j)=1;
		Pg(4*j+2,3*j+1)=1;
		Pg(4*j+3,NCd+j+1)=1;
	}
	
	j=Nd-1;
	B(3*j,3*j-1)=1;
	B(3*j,3*j)=1;
	B(3*j,3*j+1)=1;
	B(3*j+1,3*j-1)=3;
	B(3*j+1,3*j)=2;
	B(3*j+1,3*j+1)=1;
	
	Ps(3*j,ind_xlims(1)+j)=-1;
	Ps(3*j,ind_xlims(1)+j+1)=1;
	
	Pg(4*j,3*j-1)=1;
	Pg(4*j+1,3*j)=1;
	Pg(4*j+2,3*j+1)=1;
	Pg(4*j+3,NCd+j+1)=1;
	
	IB.eye(NCd,NCd);
	invB=solve(B,IB);
	
	PA=IA.submat(ind_xlims(1)-1,0,Nx-2,Nx-1);
	
	Lg=join_vert(invB*Ps,PA);
	
	mat Md=Pg*Lg;
	
	/*
	Pg.zeros(4*Nd,4*Nd-1);
	 
	B(0,0)=3;
	B(0,1)=2;
	B(0,2)=1;
	B(1,0)=1;
	B(1,1)=1;
	B(1,2)=1;
	
	Ps(0,ind_xlims(1)-1)=-fdAd;
	Ps(0,ind_xlims(1)+1)=fdAd;
	Ps(1,ind_xlims(1))=1;
	Ps(1,ind_xlims(1)+1)=-1;
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(2,2)=1;
	Pg(3,NCd)=1;
	
	for (j=1; j<Nd-1; j++)
	{
		B(3*j-1,3*j-1)=-RDd(j-1);
		B(3*j-1,3*j)=3;
		B(3*j-1,3*j+1)=2;
		B(3*j-1,3*j+2)=1;
		
		B(3*j,3*j-2)=-2*pow(RDd(j-1),2);
		B(3*j,3*j)=6;
		B(3*j,3*j+1)=2;
		
		B(3*j+1,3*j)=1;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+2)=1;
		
		Ps(3*j+1,ind_xlims(1)+j)=1;
		Ps(3*j+1,ind_xlims(1)+j+1)=-1;
		
		Pg(4*j,3*j)=1;
		Pg(4*j+1,3*j+1)=1;
		Pg(4*j+2,3*j+2)=1;
		Pg(4*j+3,NCd+j)=1;
	}
	
	j=Nd-1;
	B(3*j-1,3*j-1)=-RDd(j-1);
	B(3*j-1,3*j)=3;
	B(3*j-1,3*j+1)=2;
	
	B(3*j,3*j-2)=-2*pow(RDd(j-1),2);
	B(3*j,3*j)=6;
	B(3*j,3*j+1)=2;
	
	B(3*j+1,3*j)=1;
	B(3*j+1,3*j+1)=1;
	
	Ps(3*j+1,ind_xlims(1)+j)=1;
	Ps(3*j+1,ind_xlims(1)+j+1)=-1;
	
	Pg(4*j,3*j)=1;
	Pg(4*j+1,3*j+1)=1;
	Pg(4*j+3,NCd+j)=1;
	 
	IB.eye(NCd,NCd);
	invB=solve(B,IB);
	
	PA=IA.submat(ind_xlims(1)+1,0,Nx-1,Nx-1);
	
	Lg=join_vert(invB*Ps,PA);
	
	mat Md=Pg*Lg;
	*/
	 
	M=join_vert(Mg,Mc);
	M=join_vert(M,Md);
	
	return true;
}

void OmegaMaxEnt_data::convert_matrix_to_band_format(mat M, mat &Mbf, int KL, int KU)
{
	int N=M.n_rows;
	int ind_d;
	
	for (ind_d=KU; ind_d>=0; ind_d--)
	{
		Mbf.submat(KL+KU-ind_d,ind_d,KL+KU-ind_d,N-1)=trans(M.diag(ind_d));
	}
	for (ind_d=-1; ind_d>=-KL; ind_d--)
	{
		Mbf.submat(KL+KU-ind_d,0,KL+KU-ind_d,N+ind_d-1)=trans(M.diag(ind_d));
	}
}

bool OmegaMaxEnt_data::spline_matrix_grid_transf_G_part(vec x, uvec ind_xlims, vec xs, mat &M)
{
	int Nx=x.n_rows;
	
	int Ng=ind_xlims(0);
	int Nc=ind_xlims(1)-ind_xlims(0);
	int Nd=Nx-ind_xlims(1)-1;
	
	double x0g=xs(0);
	double x0d=xs(1);
	
	//	 solve the spline in the interval -inf,wl]
	//vec Dg=-((x.rows(1,ind_xlims(0))-x0g)%(x.rows(0,ind_xlims(0)-1)-x0g))/(x.rows(1,ind_xlims(0))-x.rows(0,ind_xlims(0)-1));
	
	vec RDg=((x.rows(2,ind_xlims(0))-x0g)/(x.rows(0,ind_xlims(0)-2)-x0g))%((x.rows(1,ind_xlims(0)-1)-x.rows(0,ind_xlims(0)-2))/(x.rows(2,ind_xlims(0))-x.rows(1,ind_xlims(0)-1)));
	
	int NCg=3*Ng-1;
	mat B=zeros<mat>(NCg,NCg);
	mat Ps=zeros<mat>(NCg, Nx);
	mat Pg=zeros<mat>(4*Ng,4*Ng-1);
	
	B(0,0)=1;
	B(0,1)=1;
	B(1,0)=3;
	B(1,1)=2;
	B(1,4)=-RDg(0);
	B(2,0)=6;
	B(2,1)=2;
	B(2,3)=-2*pow(RDg(0),2);
	
	Ps(0,0)=-1;
	Ps(0,1)=1;
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(3,NCg)=1;
	
	int j;
	for (j=1; j<Ng-1; j++)
	{
		B(3*j,3*j-1)=1;
		B(3*j,3*j)=1;
		B(3*j,3*j+1)=1;
		B(3*j+1,3*j-1)=3;
		B(3*j+1,3*j)=2;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+4)=-RDg(j);
		B(3*j+2,3*j-1)=6;
		B(3*j+2,3*j)=2;
		B(3*j+2,3*j+3)=-2*pow(RDg(j),2);
		
		Ps(3*j,j)=-1;
		Ps(3*j,j+1)=1;
		
		Pg(4*j,3*j-1)=1;
		Pg(4*j+1,3*j)=1;
		Pg(4*j+2,3*j+1)=1;
		Pg(4*j+3,NCg+j)=1;
	}
	
	j=Ng-1;
	B(3*j,3*j-1)=1;
	B(3*j,3*j)=1;
	B(3*j,3*j+1)=1;
	B(3*j+1,3*j-1)=3;
	B(3*j+1,3*j)=2;
	B(3*j+1,3*j+1)=1;
	
	double fdAg=((x(j+1)-x0g)/(x(j)-x0g))*((x(j+1)-x(j))/(x(j+2)-x(j)));
	
	Ps(3*j,j)=-1;
	Ps(3*j,j+1)=1;
	Ps(3*j+1,j)=-fdAg;
	Ps(3*j+1,j+2)=fdAg;
	
	Pg(4*j,3*j-1)=1;
	Pg(4*j+1,3*j)=1;
	Pg(4*j+2,3*j+1)=1;
	Pg(4*j+3,NCg+j)=1;
	
	mat IB=eye(NCg,NCg);
	
/*
	int N=NCg;
	int INFO;
	int *IPIV=new int[N];
	int KD=3;
	
	mat invB=IB;
	int Nbf=3*KD+1;
	mat Bbf(Nbf,N);
	convert_matrix_to_band_format(B, Bbf, KD, KD);

	dgbsv_(&N, &KD, &KD, &N, Bbf.memptr(), &Nbf, IPIV, invB.memptr(), &N, &INFO );
	
	delete [] IPIV;
*/
	
//	mat invB2=solve(B,IB);
//	cout<<"max(|invB-invB2|): "<<max(max(abs(invB-invB2)))<<endl;
	mat invB=solve(B,IB);
	
	mat IA=eye(Nx,Nx);
	mat PA=IA.submat(0,0,Ng-1,Nx-1);
	
	mat Lg=join_vert(invB*Ps,PA);
	
	mat Mg=Pg*Lg;
	
	// solve in the interval [wl,wr]
	//vec Dc=1.0/(x.rows(ind_xlims(0)+1,ind_xlims(1))-x.rows(ind_xlims(0),ind_xlims(1)-1));
	
	vec RDc=(x.rows(ind_xlims(0)+1,ind_xlims(1)-1)-x.rows(ind_xlims(0),ind_xlims(1)-2))/(x.rows(ind_xlims(0)+2,ind_xlims(1))-x.rows(ind_xlims(0)+1,ind_xlims(1)-1));
	
	int NCc=3*Nc-1;
	
	B.zeros(NCc,NCc);
	Ps.zeros(NCc, Nx);
	Pg.zeros(4*Nc,4*Nc);
	
	B(0,0)=1;
	B(0,1)=1;
	B(1,0)=3;
	B(1,1)=2;
	B(1,4)=-RDc(0);
	B(2,0)=6;
	B(2,1)=2;
	B(2,3)=-2*pow(RDc(0),2);
	
	Ps(0,Ng-1)=(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Ps(0,Ng)=-1;
	Ps(0,Ng+1)=1-(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Ps(1,Ng-1)=(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Ps(1,Ng+1)=-(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(2,NCc)=-(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Pg(2,NCc+2)=(x(Ng+1)-x(Ng))/(x(Ng+1)-x(Ng-1));
	Pg(3,NCc+1)=1;
	
	for (j=1; j<Nc-1; j++)
	{
		B(3*j,3*j-1)=1;
		B(3*j,3*j)=1;
		B(3*j,3*j+1)=1;
		B(3*j+1,3*j-1)=3;
		B(3*j+1,3*j)=2;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+4)=-RDc(j);
		B(3*j+2,3*j-1)=6;
		B(3*j+2,3*j)=2;
		B(3*j+2,3*j+3)=-2*pow(RDc(j),2);
		
		Ps(3*j,Ng+j)=-1;
		Ps(3*j,Ng+j+1)=1;
		
		Pg(4*j,3*j-1)=1;
		Pg(4*j+1,3*j)=1;
		Pg(4*j+2,3*j+1)=1;
		Pg(4*j+3,NCc+j+1)=1;
	}
	
	j=Nc-1;
	B(3*j,3*j-1)=1;
	B(3*j,3*j)=1;
	B(3*j,3*j+1)=1;
	B(3*j+1,3*j-1)=3;
	B(3*j+1,3*j)=2;
	B(3*j+1,3*j+1)=1;
	
	Ps(3*j,Ng+j)=-1;
	Ps(3*j,Ng+j+1)=1;
	Ps(3*j+1,Ng+j)=-(x(ind_xlims(1))-x(ind_xlims(1)-1))/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	Ps(3*j+1,Ng+j+2)=(x(ind_xlims(1))-x(ind_xlims(1)-1))/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	
	Pg(4*j,3*j-1)=1;
	Pg(4*j+1,3*j)=1;
	Pg(4*j+2,3*j+1)=1;
	Pg(4*j+3,NCc+j+1)=1;
	
	IB.eye(NCc,NCc);
/*
	N=NCc;
	IPIV=new int[N];
	invB=IB;
	
	Bbf.zeros(Nbf,N);
	convert_matrix_to_band_format(B, Bbf, KD, KD);
	
	dgbsv_(&N, &KD, &KD, &N, Bbf.memptr(), &Nbf, IPIV, invB.memptr(), &N, &INFO );
	
	delete [] IPIV;
*/
//	mat invB2=solve(B,IB);
//	cout<<"max(|invB-invB2|): "<<max(max(abs(invB-invB2)))<<endl;
	invB=solve(B,IB);
	
	PA=IA.submat(Ng-1,0,ind_xlims(1)-1,Nx-1);
	
	Lg=join_vert(invB*Ps,PA);
	
	mat Mc=Pg*Lg;
	
	int NCd=3*Nd-1;
	
	//vec Dd=((x.rows(ind_xlims(1),Nx-2)-x0d)%(x.rows(ind_xlims(1)+1,Nx-1)-x0d))/(x.rows(ind_xlims(1)+1,Nx-1)-x.rows(ind_xlims(1),Nx-2));
	
	vec RDd=((x.rows(ind_xlims(1)+2,Nx-1)-x.rows(ind_xlims(1)+1,Nx-2))/(x.rows(ind_xlims(1)+1,Nx-2)-x.rows(ind_xlims(1),Nx-3)))%((x.rows(ind_xlims(1),Nx-3)-x0d)/(x.rows(ind_xlims(1)+2,Nx-1)-x0d));
	
	double fdAd=((x(ind_xlims(1))-x0d)/(x(ind_xlims(1)+1)-x0d))*((x(ind_xlims(1))-x(ind_xlims(1)+1))/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1)));
	
	B.zeros(NCd,NCd);
	Ps.zeros(NCd, Nx);
	Pg.zeros(4*Nd,4*Nd-1);
	
	B(0,0)=3;
	B(0,1)=2;
	B(0,2)=1;
	B(1,0)=1;
	B(1,1)=1;
	B(1,2)=1;
	
	Ps(0,ind_xlims(1)-1)=-fdAd;
	Ps(0,ind_xlims(1)+1)=fdAd;
	Ps(1,ind_xlims(1))=1;
	Ps(1,ind_xlims(1)+1)=-1;
	
	Pg(0,0)=1;
	Pg(1,1)=1;
	Pg(2,2)=1;
	Pg(3,NCd)=1;
	
	for (j=1; j<Nd-1; j++)
	{
		B(3*j-1,3*j-1)=-RDd(j-1);
		B(3*j-1,3*j)=3;
		B(3*j-1,3*j+1)=2;
		B(3*j-1,3*j+2)=1;
		
		B(3*j,3*j-2)=-2*pow(RDd(j-1),2);
		B(3*j,3*j)=6;
		B(3*j,3*j+1)=2;
		
		B(3*j+1,3*j)=1;
		B(3*j+1,3*j+1)=1;
		B(3*j+1,3*j+2)=1;
		
		Ps(3*j+1,ind_xlims(1)+j)=1;
		Ps(3*j+1,ind_xlims(1)+j+1)=-1;
		
		Pg(4*j,3*j)=1;
		Pg(4*j+1,3*j+1)=1;
		Pg(4*j+2,3*j+2)=1;
		Pg(4*j+3,NCd+j)=1;
	}
	
	j=Nd-1;
	B(3*j-1,3*j-1)=-RDd(j-1);
	B(3*j-1,3*j)=3;
	B(3*j-1,3*j+1)=2;
	
	B(3*j,3*j-2)=-2*pow(RDd(j-1),2);
	B(3*j,3*j)=6;
	B(3*j,3*j+1)=2;
	
	B(3*j+1,3*j)=1;
	B(3*j+1,3*j+1)=1;
	
	Ps(3*j+1,ind_xlims(1)+j)=1;
	Ps(3*j+1,ind_xlims(1)+j+1)=-1;
	
	Pg(4*j,3*j)=1;
	Pg(4*j+1,3*j+1)=1;
	Pg(4*j+3,NCd+j)=1;
	
	IB.eye(NCd,NCd);
/*
	N=NCd;
	IPIV=new int[N];
	invB=IB;
	
	Bbf.zeros(Nbf,N);
	convert_matrix_to_band_format(B, Bbf, KD, KD);
	
	dgbsv_(&N, &KD, &KD, &N, Bbf.memptr(), &Nbf, IPIV, invB.memptr(), &N, &INFO );
	
	delete [] IPIV;
*/
//	mat invB2=solve(B,IB);
//	cout<<"max(|invB-invB2|): "<<max(max(abs(invB-invB2)))<<endl;
	invB=solve(B,IB);
	
	PA=IA.submat(ind_xlims(1)+1,0,Nx-1,Nx-1);
	
	Lg=join_vert(invB*Ps,PA);
	
	mat Md=Pg*Lg;
	
	M=join_vert(Mg,Mc);
	M=join_vert(M,Md);
	
	return true;
}

bool OmegaMaxEnt_data::spline_matrix_G_part(vec x, uvec ind_xlims, vec xs, mat &Mspl)
{
	int Nx=x.n_rows;
	
	int Ng=ind_xlims(0)+1;
	int Nc=ind_xlims(1)-ind_xlims(0);
	int Nd=Nx-ind_xlims(1);
	
	double x0g=xs(0);
	double x0d=xs(1);
	
	
	//	 solve the spline in the interval -inf,wl]
	int NSg=3*Ng-1;
	mat Aspl=zeros<mat>(NSg,NSg);
	
	// Dg=zeros(NSg,1);
	// Dg(1)=S(1);
	
	double u=1.0/(x(0)-x0g);
	
	Aspl(0,0)=pow(u,3);
	Aspl(0,1)=pow(u,2);
	Aspl(1,0)=3*pow(u,2);
	Aspl(1,1)=2*u;
	Aspl(1,4)=-1;
	Aspl(2,0)=6*u;
	Aspl(2,1)=2;
	Aspl(2,3)=-2;
	
	int j;
	for (j=2; j<Ng; j++)
	{
		u=1/(x(j-1)-x0g)-1/(x(j-2)-x0g);
	 
	 //    Dg(3*j-2)=S(j)-S(j-1);
	 
		Aspl(3*j-3,3*j-4)=pow(u,3);
		Aspl(3*j-3,3*j-3)=pow(u,2);
		Aspl(3*j-3,3*j-2)=u;
	 
		Aspl(3*j-2,3*j-4)=3*pow(u,2);
		Aspl(3*j-2,3*j-3)=2*u;
		Aspl(3*j-2,3*j-2)=1;
		Aspl(3*j-2,3*j+1)=-1;
	 
		Aspl(3*j-1,3*j-4)=6*u;
		Aspl(3*j-1,3*j-3)=2;
		Aspl(3*j-1,3*j)=-2;
	}
	
	j=ind_xlims(0)+1;
	//Dg(3*j-2)=S(j)-S(j-1);
	u=1/(x(j-1)-x0g)-1/(x(j-2)-x0g);
	double ug=1/(x(j-1)-x0g);
	
	Aspl(3*j-3,3*j-4)=pow(u,3);
	Aspl(3*j-3,3*j-3)=pow(u,2);
	Aspl(3*j-3,3*j-2)=u;
	
	Aspl(3*j-2,3*j-4)=-3*pow(ug,2)*pow(u,2);
	Aspl(3*j-2,3*j-3)=-2*pow(ug,2)*u;
	Aspl(3*j-2,3*j-2)=-pow(ug,2);
	
	//Dg(3*j-1)=(S(j+1)-S(j-1))/(x(j+1)-x(j-1));
	//Cg=Ag\Dg;
	
	mat D(NSg,NSg);
	D.eye();
	
	mat Cg=solve(Aspl,D);
	
	mat T(NSg,Nx);
	T.zeros();
	T(0,0)=1;
	for (j=2; j<=Ng; j++)
	{
		T(3*j-3,j-2)=-1;
		T(3*j-3,j-1)=1;
	}
	T(3*Ng-2,Ng-2)=-1/(x(Ng)-x(Ng-2));
	T(3*Ng-2,Ng)=1/(x(Ng)-x(Ng-2));
	
	Cg=Cg*T;
	
	//solve the spline in the interval [wl,wr]
	int NSc=3*Nc-1;
	Aspl.zeros(NSc,NSc);
	
	//Dc=zeros(NSc,1);
	//Dc(1)=S(Ng+1)-(x(Ng+1)-x(Ng))*(S(Ng+1)-S(Ng-1))/(x(Ng+1)-x(Ng-1))-S(Ng);
	
	double Dx=x(Ng)-x(Ng-1);
	
	Aspl(0,0)=pow(Dx,3);
	Aspl(0,1)=pow(Dx,2);
	
	//Dc(2)=-(S(Ng+1)-S(Ng-1))/(x(Ng+1)-x(Ng-1));
	
	Aspl(1,0)=3*pow(Dx,2);
	Aspl(1,1)=2*Dx;
	Aspl(1,4)=-1;
	
	Aspl(2,0)=6*Dx;
	Aspl(2,1)=2;
	Aspl(2,3)=-2;
	
	for (j=2; j<Nc; j++)
	{
		Dx=x(Ng+j-1)-x(Ng+j-2);
	 
	 //    Dc(3*j-2)=S(Ng+j)-S(Ng+j-1);
	 
		Aspl(3*j-3,3*j-4)=pow(Dx,3);
		Aspl(3*j-3,3*j-3)=pow(Dx,2);
		Aspl(3*j-3,3*j-2)=Dx;
	 
		Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
		Aspl(3*j-2,3*j-3)=2*Dx;
		Aspl(3*j-2,3*j-2)=1;
		Aspl(3*j-2,3*j+1)=-1;
	 
		Aspl(3*j-1,3*j-4)=6*Dx;
		Aspl(3*j-1,3*j-3)=2;
		Aspl(3*j-1,3*j)=-2;
	}
	
	j=Nc;
	Dx=x(Ng+j-1)-x(Ng+j-2);
	//Dc(3*j-2)=S(Ng+j)-S(Ng+j-1);
	Aspl(3*j-3,3*j-4)=pow(Dx,3);
	Aspl(3*j-3,3*j-3)=pow(Dx,2);
	Aspl(3*j-3,3*j-2)=Dx;
	
	//Dc(3*j-1)=(S(Ng+j+1)-S(Ng+j-1))/(x(Ng+j+1)-x(Ng+j-1));
	Aspl(3*j-2,3*j-4)=3*pow(Dx,2);
	Aspl(3*j-2,3*j-3)=2*Dx;
	Aspl(3*j-2,3*j-2)=1;
	
	//Cc=A\Dc;
	
	D.eye(NSc,NSc);
	mat Cc=solve(Aspl,D);
	
	T.zeros(NSc,Nx);
	T(0,Ng-2)=(x(Ng)-x(Ng-1))/(x(Ng)-x(Ng-2));
	T(0,Ng-1)=-1;
	T(0,Ng)=1-(x(Ng)-x(Ng-1))/(x(Ng)-x(Ng-2));
	T(1,Ng-2)=1/(x(Ng)-x(Ng-2));
	T(1,Ng)=-1/(x(Ng)-x(Ng-2));
	for (j=2; j<=Nc; j++)
	{
		T(3*j-3,Ng+j-2)=-1;
		T(3*j-3,Ng+j-1)=1;
	}
	T(3*Nc-2,ind_xlims(1)-1)=-1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	T(3*Nc-2,ind_xlims(1)+1)=1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	
	Cc=Cc*T;
	
	mat Cc2(NSc+1,Nx);
	Cc2.zeros();
	
	Cc2.rows(0,1)=Cc.rows(0,1);
	Cc2.rows(3,NSc)=Cc.rows(2,NSc-1);
	Cc2(2,Ng-2)=-1/(x(Ng)-x(Ng-2));
	Cc2(2,Ng)=1/(x(Ng)-x(Ng-2));
	
	//solve the spline in the interval [wr,inf
	int NSd=3*Nd-1;
	Aspl.zeros(NSd,NSd);
	//Dd=zeros(NSd,1);
	
	u=1/(x(ind_xlims(1))-x0d)-1/(x(ind_xlims(1)+1)-x0d);
	double ud=1/(x(ind_xlims(1))-x0d);
	
	//Dd(1)=(S(ind_xlims(2)+1)-S(ind_xlims(2)-1))/(x(ind_xlims(2)+1)-x(ind_xlims(2)-1));
	Aspl(0,0)=-3*pow(ud,2)*pow(u,2);
	Aspl(0,1)=-2*pow(ud,2)*u;
	Aspl(0,2)=-pow(ud,2);
	
	//Dd(2)=S(ind_xlims(2))-S(ind_xlims(2)+1);
	Aspl(1,0)=pow(u,3);
	Aspl(1,1)=pow(u,2);
	Aspl(1,2)=u;
	
	for (j=1; j<Nd-1; j++)
	{
		u=1/(x(ind_xlims(1)+j)-x0d)-1/(x(ind_xlims(1)+j+1)-x0d);
		
		Aspl(3*j-1,3*j-1)=-1;
		Aspl(3*j-1,3*j)=3*pow(u,2);
		Aspl(3*j-1,3*j+1)=2*u;
		Aspl(3*j-1,3*j+2)=1;
		
		Aspl(3*j,3*j-2)=-2;
		Aspl(3*j,3*j)=6*u;
		Aspl(3*j,3*j+1)=2;
	 
	 //    Dd(3*j+2)=S(ind_xlims(2)+j)-S(ind_xlims(2)+j+1);
		Aspl(3*j+1,3*j)=pow(u,3);
		Aspl(3*j+1,3*j+1)=pow(u,2);
		Aspl(3*j+1,3*j+2)=u;
	}
	
	j=Nd-1;
	u=1/(x(ind_xlims(1)+j)-x0d);
	
	Aspl(3*j-1,3*j-1)=-1;
	Aspl(3*j-1,3*j)=3*pow(u,2);
	Aspl(3*j-1,3*j+1)=2*u;
	
	Aspl(3*j,3*j-2)=-2;
	Aspl(3*j,3*j)=6*u;
	Aspl(3*j,3*j+1)=2;
	
	//Dd(3*j+2)=S(ind_xlims(2)+j);
	Aspl(3*j+1,3*j)=pow(u,3);
	Aspl(3*j+1,3*j+1)=pow(u,2);
	
	//Cd=A\Dd;
	
	D.eye(NSd,NSd);
	mat Cd=solve(Aspl,D);
	
	T.zeros(NSd,Nx);
	T(0,ind_xlims(1)-1)=-1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	T(0,ind_xlims(1)+1)=1/(x(ind_xlims(1)+1)-x(ind_xlims(1)-1));
	for (j=1; j<Nd; j++)
	{
		T(3*j-2,ind_xlims(1)+j-1)=1;
		T(3*j-2,ind_xlims(1)+j)=-1;
	}
	j=Nd;
	T(3*j-2,ind_xlims(1)+j-1)=1;
	
	Cd=Cd*T;
	
	int NS_tot=NSg+NSc+NSd+1;
	
	Mspl.zeros(NS_tot,Nx);
	Mspl.rows(0,NSg-1)=Cg;
	Mspl.rows(NSg,NSg+NSc)=Cc2;
	Mspl.rows(NSg+NSc+1,NS_tot-1)=Cd;
	
	return true;
}

bool OmegaMaxEnt_data::set_default_model()
{
	if (def_model_file.size())
	{
		cout<<"default model provided\n";
		vec w_def=def_data.col(0);
		vec def_m=def_data.col(1);
		if (def_m.min()<0)
		{
			cout<<"set_default_model() error: provided default model has a negative value.\n";
			return false;
		}
		
		uint Nw_def=w_def.n_rows;
		double wmn=w(0), wmx=w(Nw-1);
		double wmn_def=w_def(0), wmx_def=w_def(Nw_def-1);
		int j1=0, j2=1, j3=2;
		double w0l_def=(2*w_def(j1)*w_def(j3)-w_def(j1)*w_def(j2)-w_def(j2)*w_def(j3))/(w_def(j1)-2*w_def(j2)+w_def(j3));
		j1=Nw_def-3;
		j2=Nw_def-2;
		j3=Nw_def-1;
		double w0r_def=(2*w_def(j1)*w_def(j3)-w_def(j1)*w_def(j2)-w_def(j2)*w_def(j3))/(w_def(j1)-2*w_def(j2)+w_def(j3));
		if (w0l_def<wmn_def || w0r_def>wmx_def)
		{
			//cout<<"user defined default model.\n";
			if (w_def(0)<wl && w_def(Nw_def-1)>wr)
			{
				int R_wmn_def_SW=4;
				double wmn_def2=w(0), wmx_def2=w(Nw-1);
				//wmn_def2=wl-R_wmn_def_SW*SW;
				//wmx_def2=wr+R_wmn_def_SW*SW;
				int j=0;
				while (w_def(j)<wmn_def2 && j<Nw_def-1)	j++;
				int jl=j-1;
				if (jl<0) jl=0;
				j=Nw_def-1;
				while (w_def(j)>wmx_def2 && j>0) j--;
				int jr=j+1;
				if (jr>Nw_def-1) jr=Nw_def-1;
				
				if (def_m(jl+1)>def_m(jl) && def_m(jr-1)>def_m(jr))
				{
					//w_def=w_def.rows(jl,jr);
					//def_m=def_m.rows(jl,jr);
					//Nw_def=jr-jl+1;
					
					double l21=log(def_m(1)/def_m(0));
					double l31=log(def_m(2)/def_m(0));
					double w1=w_def(0);
					double w2=w_def(1);
					double w3=w_def(2);
					double wcg=(l21*(pow(w1,2)-pow(w3,2))-l31*(pow(w1,2)-pow(w2,2)))/(2*l31*(w2-w1)-2*l21*(w3-w1));
					double C1g=(pow(w1-wcg,2)-pow(w2-wcg,2))/l21;
					double C2g=exp(-pow(w1-wcg,2)/C1g)/def_m(0);
					
					l21=log(def_m(Nw_def-2)/def_m(Nw_def-3));
					l31=log(def_m(Nw_def-1)/def_m(Nw_def-3));
					w1=w_def(Nw_def-3);
					w2=w_def(Nw_def-2);
					w3=w_def(Nw_def-1);
					double wcd=(l21*(pow(w1,2)-pow(w3,2))-l31*(pow(w1,2)-pow(w2,2)))/(2*l31*(w2-w1)-2*l21*(w3-w1));
					double C1d=(pow(w1-wcd,2)-pow(w2-wcd,2))/l21;
					double C2d=exp(-pow(w1-wcd,2)/C1d)/def_m(Nw_def-3);
					
					vec gaussians_params(6);
					gaussians_params(0)=wcg;
					gaussians_params(1)=C1g;
					gaussians_params(2)=C2g;
					gaussians_params(3)=wcd;
					gaussians_params(4)=C1d;
					gaussians_params(5)=C2d;
					
					double dfg=-2*(w_def(0)-wcg)*def_m(0)/C1g;
					double dfd=-2*(w_def(Nw_def-1)-wcd)*def_m(Nw_def-1)/C1d;
					
					uint Nc=4*(Nw_def-1);
					vec coeffs_spline_def(Nc+1);
					coeffs_spline_def(0)=dfg;
					coeffs_spline_def(1)=dfd;
					coeffs_spline_def(Nc)=def_m(Nw_def-1);
					
					if (dfg>=0 && dfd<=0 && C1g>0 && C1d>0 && C2g>0 && C2d>0)
					{
						spline_coeffs(w_def.memptr(),def_m.memptr(),Nw_def,coeffs_spline_def.memptr());
						if (!default_model_val_G(w, w_def, coeffs_spline_def, gaussians_params, default_model))
						{
							return false;
						}
					}
					else
					{
						cout<<"set_default_model(): problem with the user defined default model.\n";
						if (dfg<=0)
							cout<<"The derivative does not have the correct sign at the left boundary\n";
						if (dfd>=0)
							cout<<"The derivative does not have the correct sign at the right boundary\n";
						if ( (C1g<0 || C1d<0 || C2g<0 || C2d<0) && (w_def(1)>w(1) || w_def(Nw_def)<w(Nw)) )
							cout<<"Incorrect parameters found. Unable to extend the model to whole frequency range.\n";
						
						return false;
					}
					
				}
				else if (def_m(jl+1)==def_m(jl) && def_m(jr-1)==def_m(jr))
				{
					default_model.zeros(Nw);
					int l=1;
					for (j=1; j<Nw-1; j++)
					{
						while (l<Nw_def-1 && w(j)>=w_def(l)) l++;
						
						if (w(j)>=w_def(l-1) && w(j)<w_def(l))
							default_model(j)=(w(j)-w_def(l-1))*(def_m(l)-def_m(l-1))/(w_def(l)-w_def(l-1))+def_m(l-1);
						else if (w(j)==w_def(l))
							default_model(j)=def_m(l);
					}
					//				rowvec dw_tmp=trans(w.rows(2,Nw-1)-w.rows(0,Nw-3))/2.0;
					//				mat int_def_m=dw_tmp*(default_model.rows(1,Nw-2))/(2*PI);
					//				if (!boson)
					//					default_model=M0*default_model/int_def_m(0);
					//				else
					//					default_model=M1n*default_model/int_def_m(0);
				}
				else
				{
					cout<<"warning: the provided default model is not valid. The function must be decreasing \n";
					return false;
				}
			}
			else
			{
				cout<<"warning: the grid of the user defined default model must extend beyond the frequency range of the spectral function\n";
				return false;
			}
			/*
			if (w_def(0)<wl && w_def(Nw_def-1)>wr)
			{
				int j=0;
				while (abs(w_def(j+1)-wl)<abs(w_def(j)-wl) && j<Nw_def-2)	j++;
				int jl=j;
				while (abs(w_def(j+1)-wr)<abs(w_def(j)-wr))	 j++;
				int jr=j;
				
				if (def_m(jl+1)>def_m(jl) && def_m(jr-1)>def_m(jr))
				{
					w_def=w_def.rows(jl,jr);
					def_m=def_m.rows(jl,jr);
					Nw_def=jr-jl+1;
					
					double l21=log(def_m(1)/def_m(0));
					double l31=log(def_m(2)/def_m(0));
					double w1=w_def(0);
					double w2=w_def(1);
					double w3=w_def(2);
					double wcg=(l21*(pow(w1,2)-pow(w3,2))-l31*(pow(w1,2)-pow(w2,2)))/(2*l31*(w2-w1)-2*l21*(w3-w1));
					double C1g=(pow(w1-wcg,2)-pow(w2-wcg,2))/l21;
					double C2g=exp(-pow(w1-wcg,2)/C1g)/def_m(0);
					
					l21=log(def_m(Nw_def-2)/def_m(Nw_def-3));
					l31=log(def_m(Nw_def-1)/def_m(Nw_def-3));
					w1=w_def(Nw_def-3);
					w2=w_def(Nw_def-2);
					w3=w_def(Nw_def-1);
					double wcd=(l21*(pow(w1,2)-pow(w3,2))-l31*(pow(w1,2)-pow(w2,2)))/(2*l31*(w2-w1)-2*l21*(w3-w1));
					double C1d=(pow(w1-wcd,2)-pow(w2-wcd,2))/l21;
					double C2d=exp(-pow(w1-wcd,2)/C1d)/def_m(Nw_def-3);
					
					vec gaussians_params(6);
					gaussians_params(0)=wcg;
					gaussians_params(1)=C1g;
					gaussians_params(2)=C2g;
					gaussians_params(3)=wcd;
					gaussians_params(4)=C1d;
					gaussians_params(5)=C2d;
					
					double dfg=-2*(w_def(0)-wcg)*def_m(0)/C1g;
					double dfd=-2*(w_def(Nw_def-1)-wcd)*def_m(Nw_def-1)/C1d;
					//vec Ddefw(2);
					//Ddefw(0)=dfg;
					//Ddefw(1)=dfd;
					
					//			string spline_type("normal");
					uint Nc=4*(Nw_def-1);
					vec coeffs_spline_def(Nc+1);
					coeffs_spline_def(0)=dfg;
					coeffs_spline_def(1)=dfd;
					coeffs_spline_def(Nc)=def_m(Nw_def-1);
					
					//			coeffs_spline_def=spline_coeffs(w_def,def_m,Nw_def,Ddefw);
					//			spline_params=Ddefw;
					//			wl_def=wl;
					//			wr_def=wr;
					if (dfg>=0 && dfd<=0 && C1g>0 && C1d>0 && C2g>0 && C2d>0)
					{
						spline_coeffs(w_def.memptr(),def_m.memptr(),Nw_def,coeffs_spline_def.memptr());
						if (!default_model_val_G(w, w_def, coeffs_spline_def, gaussians_params, default_model))
						{
							return false;
						}
						//				default_model=default_model_val_maxent_G(w,w_def,def_m,coeffs_spline_def,spline_type,Ddefw,gaussians_params);
						//				int_def_m=KM(1,:)*(default_model);
						//				default_model=default_model/int_def_m;
					}
					else
					{
						cout<<"set_default_model(): problem with the user defined default model.\n";
						if (dfg<=0)
							cout<<"The derivative does not have the correct sign at the left boundary\n";
						if (dfd>=0)
							cout<<"The derivative does not have the correct sign at the right boundary\n";
						if ( (C1g<0 || C1d<0 || C2g<0 || C2d<0) && (w_def(1)>w(1) || w_def(Nw_def)<w(Nw)) )
							cout<<"Incorrect parameters found. Unable to extend the model to whole frequency range.\n";
						
						return false;
					}
				}
				else if (def_m(jl+1)==def_m(jl) && def_m(jr-1)==def_m(jr))
				{
					default_model.zeros(Nw);
					int l=1;
					for (j=1; j<Nw-1; j++)
					{
						while (l<Nw_def-1 && w(j)>=w_def(l)) l++;
						
						if (w(j)>=w_def(l-1) && w(j)<w_def(l))
							default_model(j)=(w(j)-w_def(l-1))*(def_m(l)-def_m(l-1))/(w_def(l)-w_def(l-1))+def_m(l-1);
						else if (w(j)==w_def(l))
							default_model(j)=def_m(l);
					}
					//				rowvec dw_tmp=trans(w.rows(2,Nw-1)-w.rows(0,Nw-3))/2.0;
					//				mat int_def_m=dw_tmp*(default_model.rows(1,Nw-2))/(2*PI);
					//				if (!boson)
					//					default_model=M0*default_model/int_def_m(0);
					//				else
					//					default_model=M1n*default_model/int_def_m(0);
				}
				else
				{
					cout<<"warning: the provided default model is not valid\n";
					return false;
				}
			}
			else
			{
				cout<<"warning: the grid of the user defined default model must extend beyond the frequency range of the spectral function\n";
				return false;
			}
			 */
		}
		else
		{
			cout<<"provided default model was generated by this code.\n";
			
			if (displ_adv_prep_figs)
			{
				vec w_def2=(w_def.rows(0,Nw_def-2)+w_def.rows(1,Nw_def-1))/2;
				vec Dw_def=1.0/(w_def.rows(1,Nw_def-1)-w_def.rows(0,Nw_def-2));
				graph_2D g1;
				char xl[]="$\\\\omega$";
				char yl[]="$1/(\\\\omega_{i+1}-\\\\omega_i)$";
				char ttl[]="provided default model grid density";
				char attr[]="'o-',markeredgecolor='b', markerfacecolor='none'";
				g1.add_title(ttl);
				plot(g1,w_def2,Dw_def,xl,yl,attr);
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
			
			uvec ind_xlims=zeros<uvec>(2);
			vec xs=zeros<vec>(2);
			xs(0)=w0l_def;
			xs(1)=w0r_def;
			int j=0;
			double dw1=w_def(j+1)-w_def(j);
			double dw2=w_def(j+2)-w_def(j+1);
			double rdw=dw2/dw1;
			while (abs(rdw-1.0)>tol_rdw)
			{
				j=j+1;
				dw1=dw2;
				dw2=w_def(j+2)-w_def(j+1);
				rdw=dw2/dw1;
			}
			ind_xlims(0)=j+1;
			j=Nw_def-3;
			dw1=w_def(j+1)-w_def(j);
			dw2=w_def(j+2)-w_def(j+1);
			rdw=dw2/dw1;
			while (abs(rdw-1.0)>tol_rdw)
			{
				j=j-1;
				dw2=dw1;
				dw1=w_def(j+1)-w_def(j);
				rdw=dw2/dw1;
			}
			ind_xlims(1)=j+1;
			
			vec coeffs;
			spline_G_part(w_def, ind_xlims, xs, def_m, coeffs);
			//spline_G_omega_u(w_def, ind_xlims, xs, def_m, coeffs);
			spline_val_G_part(w, w_def, ind_xlims, xs, coeffs, default_model);
			
			/*
			//test the spline
			rowvec x0={-2, 0, 3};
			rowvec s0={0.8, 0.3, 1};
			rowvec wgt={1,1,1};
			
			vec test_A1;
			sum_gaussians(w_def, x0, s0, wgt, test_A1);
			vec coeffs2;
			spline_G_part(w_def, ind_xlims, xs, test_A1, coeffs2);
			
			double Nw2=1000;
			vec w2=linspace<vec>(wl-SW,wr+SW,Nw2);
			
			vec test_A2;
			sum_gaussians(w2, x0, s0, wgt, test_A2);
			vec test_A3;
			spline_val_G_part(w2, w_def, ind_xlims, xs, coeffs2, test_A3);
			
			graph_2D g1, g2;
			char xl[]="$\\\\omega$";
			char yl[]="A";
			char attr1[]="'o-', color='r', markeredgecolor='r', markerfacecolor='none'";
			char attr2[]="'s-', color='b', markeredgecolor='b', markerfacecolor='none'";
			char attr3[]="'+-', color='m', markeredgecolor='m', markerfacecolor='none'";
			double xlims[2], ylims[2];
			xlims[0]=w2(0);
			xlims[1]=w2(w2.n_rows-1);
			ylims[0]=0;
			ylims[1]=1.1*test_A2.max();
			
			g1.add_data(w_def.memptr(),test_A1.memptr(),w_def.n_rows);
			g1.add_attribute(attr1);
			g1.add_data(w2.memptr(),test_A2.memptr(),w2.n_rows);
			g1.add_attribute(attr2);
			g1.add_data(w2.memptr(),test_A3.memptr(),w2.n_rows);
			g1.add_attribute(attr3);
			g1.set_axes_labels(xl,yl);
			g1.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			
			char yl2[]="Delta A";
			vec Delta_A=test_A2-test_A3;
			ylims[0]=1.1*Delta_A.min();
			ylims[1]=1.1*Delta_A.max();
			g2.add_data(w2.memptr(),Delta_A.memptr(),w2.n_rows);
			g2.add_attribute(attr1);
			g2.set_axes_labels(xl,yl2);
			g2.set_axes_lims(xlims,ylims);
			g2.curve_plot();
			
			graph_2D::show_figures();
			*/
		}
	}
	else
	{
		if (!default_model_center_in.size())
		{
			if (!boson)
			{
				if (M1_set)
					default_model_center=M1/M0;
				else
					default_model_center=SC;
			}
			else
				default_model_center=M0/M1n;
		}
		if (!default_model_shape_in.size())
			default_model_shape=2.0;
		if (!default_model_width_in.size())
		{
			if (std_omega)
				default_model_width=pow(2.0,1.0/default_model_shape)*std_omega;
			else
				default_model_width=pow(2.0,1.0/default_model_shape)*SW/2;
		}
		general_normal(w, default_model_center, default_model_width, default_model_shape, default_model);
		
		/*
		 double width_DM;
		 if (std_omega)
			width_DM=std_omega;
		 else
			width_DM=SW/2;
		rowvec x0={SC};
		rowvec s0={width_DM};
		rowvec wgt={1.0};
		sum_gaussians(w,x0,s0,wgt,default_model);
		 */
//		if (!boson)
//			default_model=M0*default_model;
//		else
//			default_model=M1n*default_model;
	}
	
	default_model=abs(default_model);
	
	if (displ_prep_figs)
	{
		graph_2D g1;
		char xl[]="$\\\\omega$";
		char yl[]="default model";
		g1.add_title(yl);
		plot(g1,w,default_model,xl,NULL);
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	return true;
}

bool OmegaMaxEnt_data::general_normal(vec x, double x0, double s0, double p, vec &F)
{
	if (p<=0 || s0<=0)
	{
		if (p<0)
			cout<<"general_normal(): power must be greater than 0\n";
		else
			cout<<"general_normal(): width must be greater than 0\n";
		return false;
	}
	
	F=2*PI*p*exp(-pow(abs((x-x0)/s0),p))/(2*s0*tgamma(1.0/p));
	
	return true;
}

double OmegaMaxEnt_data::general_normal_val(double x, void *par[])
{
	double *x0=reinterpret_cast<double*>(par[0]);
	double *s0=reinterpret_cast<double*>(par[1]);
	double *p=reinterpret_cast<double*>(par[2]);
	
	if (p[0]<=0 || s0[0]<=0)
	{
		if (p[0]<0)
			cout<<"general_normal(): power must be greater than 0\n";
		else
			cout<<"general_normal(): width must be greater than 0\n";
		return false;
	}
	
	return 2*PI*p[0]*exp(-pow(abs((x-x0[0])/s0[0]),p[0]))/(2*s0[0]*tgamma(1.0/p[0]));
}

bool OmegaMaxEnt_data::sum_gaussians(vec x, rowvec x0, rowvec s0, rowvec wgt, vec &F)
{
	int Nx=x.n_rows;
	int Np=x0.n_cols;
	
	if (s0.n_cols<Np || wgt.n_cols<Np)
	{
		cout<<"sum_gaussians(): third or fourth parameter has too few elements\n";
		return false;
	}
	
	F.zeros(Nx);

	double wgtTot=0;

	for (int j=0; j<Np; j++)
	{
		F=F+wgt(j)*exp(-pow(x-x0(j),2)/(2*pow(s0(j),2)))/s0(j);
		wgtTot=wgtTot+wgt(j);
	}

	F=sqrt(2.0*PI)*F/wgtTot;
	
	return true;
}

bool OmegaMaxEnt_data::default_model_val_G(vec x, vec x0, vec coeffs, vec gaussians_params, vec &dm)
{
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;

	double wcg=gaussians_params(0);
	double C1g=gaussians_params(1);
	double C2g=gaussians_params(2);
	double wcd=gaussians_params(3);
	double C1d=gaussians_params(4);
	double C2d=gaussians_params(5);
	
	vec diff_x=x.rows(1,Nx-1)-x.rows(0,Nx-2);
	if (diff_x.min()<=0)
	{
		cout<<"default_model_val_G(): values in position vector are not strictly increasing\n";
		return false;
	}
	
	dm.zeros(Nx);
	
	int j=0;
	while (x(j)<x0(0) && j<Nx-1) j++;
	int jl=j;
	while (x(j)<x0(Nx0-1) && j<Nx-1) j++;
	int jr=j-1;
	int Nm=jr-jl+1, Nl=jl;
	int Nr=Nx-Nm-Nl;
	
	if (Nl)
	{
		vec vl=exp(-pow(x.rows(0,jl-1)-wcg,2)/C1g)/C2g;
		dm.rows(0,jl-1)=vl;
	}
	if (Nm)
	{
		vec vm;
		spline_val(x.rows(jl,jr), x0, coeffs, vm);
		dm.rows(jl,jr)=vm;
	}
	if (Nr)
	{
		vec vr=exp(-pow(x.rows(jr+1,Nx-1)-wcd,2)/C1d)/C2d;
		dm.rows(jr+1,Nx-1)=vr;
	}
	
	return true;
}

bool OmegaMaxEnt_data::spline_val(vec x, vec x0, vec coeffs, vec &s)
{
	double tol_dx=1e-6;
	
	int Nx=x.n_rows;
	int Nx0=x0.n_rows;
	int Ns=Nx0-1;
	int Nc=4*Ns;
	
	if (coeffs.n_rows<Nc)
	{
		cout<<"spline_val(): either the coefficients vector is incomplete or the position vector is too long\n";
		return false;
	}

	s.zeros(Nx);
	
	double a, b, c, d, Dx;

	int j, l;
	for (j=0; j<Nx; j++)
	{
		if (x(j)>=x0(0) && x(j)<x0(Nx0-1))
		{
			l=0;
			while ( x(j)>=x0(l+1) && l<Ns-1) l++;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			
			Dx=x(j)-x0(l);
			s(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
		}
		else if (x(j)==x0(Nx0-1))
		{
			if (coeffs.n_rows>Nc)
			{
				s(j)=coeffs[Nc];
			}
			else
			{
				l=Ns-1;
				a=coeffs(4*l);
				b=coeffs(4*l+1);
				c=coeffs(4*l+2);
				d=coeffs(4*l+3);
				
				Dx=x(j)-x0(l);
				s(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
			}
		}
		else if (fabs(x(j)-x0(0))<tol_dx*(x0(1)-x0(0)))
		{
			l=0;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			
			Dx=x(j)-x0(l);
			s(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
		}
		else if (fabs(x(j)-x0(Nx0-1))<tol_dx*(x0(Nx0-1)-x0(Nx0-2)))
		{
			l=Ns-1;
			a=coeffs(4*l);
			b=coeffs(4*l+1);
			c=coeffs(4*l+2);
			d=coeffs(4*l+3);
			
			Dx=x(j)-x0(l);
			s(j)=a*pow(Dx,3)+b*pow(Dx,2)+c*Dx+d;
		}
/*
		else
		{
			cout<<"spline_val() warning: x="<<x(j)<<" outside the boundaries of the spline\n";
		//	cout<<"spline_val(): the provided position vector has a value outside the boundaries of the spline.\n";
		//	return false;
		}
*/
	}
	
	return true;
}


void OmegaMaxEnt_data::spline_matrix(double *x0, int N0, mat &MS)
{
	int j;
	
	int NS=N0-1;
	int N=3*NS-1;
	int KL=3;
	int KU=2;
	int NA=2*KL+KU+1;
	int SA=NA*N;
	
	mat Ps(N,N0+2);
	
	double *A=new double[SA];
	for (j=0; j<SA; j++) A[j]=0;
	int *P=new int[N];
	
//	for (j=0; j<N; j++) coeffs_tmp[j]=0;
//	double dV1=coeffs[0], dVN0=coeffs[1];
	
	double x;
	
	x=x0[1]-x0[0];
	
	A[KL+KU]=x*x*x;
	A[KL+KU+1]=-6.0*x;
	A[KL+KU+2]=-3.0*x*x;
	
	A[NA+KL+KU-1]=x*x;
	A[NA+KL+KU]=-2.0;
	A[NA+KL+KU+1]=-2.0*x;
	
//	coeffs_tmp[0]=(V[1]-V[0]-dV1*x);
//	coeffs_tmp[2]=(double)dV1;
	
	Ps(0,0)=-x;
	Ps(2,0)=1.0;
	Ps(0,1)=-1.0;
	Ps(0,2)=1.0;
	
	for (j=1; j<NS-1; j++)
	{
		x=x0[j+1]-x0[j];
		
		A[(3*j-1)*NA+KL+KU+1]=x*x*x;
		A[(3*j-1)*NA+KL+KU+2]=-6.0*x;
		A[(3*j-1)*NA+KL+KU+3]=-3.0*x*x;
		
		A[3*j*NA+KL]=2.0;
		A[3*j*NA+KL+KU]=x*x;
		A[3*j*NA+KL+KU+1]=-2.0;
		A[3*j*NA+KL+KU+2]=-2.0*x;
		
		A[(3*j+1)*NA+KL]=1.0;
		A[(3*j+1)*NA+KL+1]=x;
		A[(3*j+1)*NA+KL+KU+1]=-1.0;
		
		Ps(3*j,j+1)=-1.0;
		Ps(3*j,j+2)=1.0;
		
//		coeffs_tmp[3*j]=(V[j+1]-V[j]);
	}
	
	j=NS-1;
	x=x0[j+1]-x0[j];
	
	A[(3*j-1)*NA+KL+KU+1]=x*x*x;
	A[(3*j-1)*NA+KL+KU+2]=3.0*x*x;
	
	A[3*j*NA+KL]=2.0;
	A[3*j*NA+KL+KU]=x*x;
	A[3*j*NA+KL+KU+1]=2.0*x;
	
	A[(3*j+1)*NA+KL]=1.0;
	A[(3*j+1)*NA+KL+1]=x;
	A[(3*j+1)*NA+KL+KU]=1.0;
	
	Ps(3*j,j+1)=-1.0;
	Ps(3*j,j+2)=1.0;
	Ps(3*j+1,j+3)=1.0;
	
//	coeffs_tmp[3*j]=(double)(V[j+1]-V[j]);
//	coeffs_tmp[3*j+1]=(double)dVN0;

	mat IB=eye(N,N);
	
	int NRHS=1;
	int INFO=0;
	dgbsv_(&N, &KL, &KU, &N, A, &NA, P, IB.memptr(), &N, &INFO );
	
//	coeffs[0]=coeffs_tmp[0];
//	coeffs[1]=coeffs_tmp[1];
//	coeffs[2]=dV1;
//	coeffs[3]=V[0];
//	for (j=1; j<NS; j++)
//	{
//		coeffs[4*j]=coeffs_tmp[3*j-1];
//		coeffs[4*j+1]=coeffs_tmp[3*j];
//		coeffs[4*j+2]=coeffs_tmp[3*j+1];
//		coeffs[4*j+3]=V[j];
//	}
	
	if (INFO)	cout<<"spline_coeffs(): INFO:  "<<INFO<<'\n';
	
	delete [] A;
	delete [] P;
}


//!Compute the coefficients of the cubic spline for V(x) known at N0 positions x0, size of coeffs must be 4*(N0-1).
//!upon entry, coeffs[0] and coeffs[1] must contain the derivatives of V(x) at x0[0] and x[N0-1],
//!upon exit, the form of coeffs is {a_0, b_0, c_0, d_0, ... a_(N0-2), b_(N0-2), c_(N0-2), d_(N0-2)}.
//!the spline values are given by S_i(x)=a_i(x-x0[i])^3+b_i(x-x0[i])^2+c_i(x-x0[i])+d_i
void OmegaMaxEnt_data::spline_coeffs(double *x0, double *V, int N0, double *coeffs)
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

bool OmegaMaxEnt_data::truncate_G_omega_n()
{
	double wn_max=wn(ind_cutoff_wn);
//	cout<<"wn_max: "<<wn_max<<endl;
	wn=wn.rows(0,ind_cutoff_wn);
	n=n.rows(0,ind_cutoff_wn);
	Nn=ind_cutoff_wn+1;
	if (Nn>Nn_max) // || Nn>=Nw)
	{
		cout<<"Number of Matsubara frequencies larger than the maximum allowed...\n";
		uvec diff_n=n.rows(1,Nn-1)-n.rows(0,Nn-2);
		if (diff_n.max()==1)
		{
			cout<<"Using a non-uniform Matsubara grid.\n";
			int i;
			int p=ceil(log2(Nn));
			int N0=pow(2,p);
			if (N0+1>Nn_all)
			{
				p=p-1;
				N0=pow(2,p);
			}
			int r=1;
			int N1=N0/pow(2,r);
			int N2=N1/2;
			Nn=N1+r*N2+1;
			n.zeros(Nn);
			n.rows(0,N1-1)=linspace<uvec>(0,N1-1,N1);
			uvec j=linspace<uvec>(0,Nn-N1-1,Nn-N1);
			uvec lj=j/N2;
			for (i=0; i<Nn-N1; i++) n(i+N1)=(j(i) % N2)*pow(2,lj(i)+1) + N1*pow(2,lj(i));
			if (!boson)
				wn=PI*tem*conv_to<vec>::from(2*n+1);
			else
				wn=2*PI*tem*conv_to<vec>::from(n);
			i=Nn-1;
			while (wn(i)>wn_max && i>0) i--;
			Nn=i+1;
			while (Nn>Nn_max) //|| Nn>=Nw)
			{
				r=r+1;
				N1=N0/pow(2,r);
				N2=N1/2;
				Nn=N1+r*N2+1;
				n.zeros(Nn);
				n.rows(0,N1-1)=linspace<uvec>(0,N1-1,N1);
				uvec j=linspace<uvec>(0,Nn-N1-1,Nn-N1);
				uvec lj=j/N2;
				for (i=0; i<Nn-N1; i++) n(i+N1)=(j(i) % N2)*pow(2,lj(i)+1) + N1*pow(2,lj(i));
				if (!boson)
					wn=PI*tem*conv_to<vec>::from(2*n+1);
				else
					wn=2*PI*tem*conv_to<vec>::from(n);
				i=Nn-1;
				while (wn(i)>wn_max && i>0) i--;
				Nn=i+1;
			}
			wn=wn.rows(0,Nn-1);
			n=n.rows(0,Nn-1);
			G=G.rows(n);
			Gr=real(G);
			Gi=imag(G);
			uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
			uvec odd_ind=even_ind+1;
			
			Gchi2.zeros(2*Nn);
			Gchi2.rows(even_ind)=Gr;
			Gchi2.rows(odd_ind)=Gi;
			
			uvec ind_R=2*n;
			uvec ind_I=2*n+1;
			
			mat COVtmp(2*Nn,2*Nn);
			COVtmp(even_ind,even_ind)=COV(ind_R,ind_R);
			COVtmp(odd_ind,odd_ind)=COV(ind_I,ind_I);
			COVtmp(even_ind,odd_ind)=COV(ind_R,ind_I);
			COVtmp(odd_ind,even_ind)=COV(ind_I,ind_R);
			COV=COVtmp;
			if (cov_diag)
			{
				errGr=errGr.rows(n);
				errGi=errGi.rows(n);
				errG.zeros(2*Nn);
				errG.rows(even_ind)=errGr;
				errG.rows(odd_ind)=errGi;
			}
			
			if (displ_prep_figs)
			{
				graph_2D g1;
				char xl[]="frequency number";
				char yl[]="frequency index";
				char attr[]="'o'";
				j=linspace<uvec>(1,Nn,Nn);
				vec jv=conv_to<vec>::from(j);
				vec nv=conv_to<vec>::from(n);
				plot(g1,jv,nv,xl,yl,attr);
				if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
				graph_2D::show_figures();
			}
		}
		else
		{
			cout<<"Your Matsubara grid is not uniform. Either make it more sparse to reduce further the number of frequencies, or increase \"Nn_max\" in the file \"OmegaMaxEnt_other_params.dat\".\n";
			return false;
		}
	}
	else
	{
		G=G.rows(0,Nn-1);
		Gr=real(G);
		Gi=imag(G);
		Gchi2=Gchi2.rows(0,2*Nn-1);
		COV=COV.submat(0,0,2*Nn-1,2*Nn-1);
		if (cov_diag)
		{
			errG=errG.rows(0,2*Nn-1);
			errGr=errGr.rows(0,Nn-1);
			errGi=errGi.rows(0,Nn-1);
		}
	}
	
	return true;
}

bool OmegaMaxEnt_data::set_omega_grid()
{
	if (!SC_set || !SW_set)
	{
		cout<<"set_omega_grid() error: SW or SC undefined\n";
		return false;
	}
	
	int i,j;
	
	if (!uniform_grid && !gaussian_grid_density)
	{
		if (!wc_exists)
		{
			cout<<"set_omega_grid() error: central part of grid undefined\n";
			return false;
		}
		double wmin=SC-f_w_range*SW/2.0;
		if (wl<0)
		{
			if (wmin>R_wmax_wr_min*wl) wmin=R_wmax_wr_min*wl;
		}
		w0l=wl+sqrt(dwl*(wl-wmin));
		int Nul=ceil((w0l-wl)/dwl);
		w0l=wl+Nul*dwl;
		double dul=dwl/((wl-dwl-w0l)*(wl-w0l));
		wmin=-1.0/dul+w0l;
		ivec ul_int=linspace<ivec>(1,Nul,Nul);
		vec ul(Nul);
		for (j=0; j<Nul; j++)
			ul(j)=-dul*ul_int(j);
		
		double wmax=SC+f_w_range*SW/2.0;
		if (wr>0)
		{
			if (wmax<R_wmax_wr_min*wr) wmax=R_wmax_wr_min*wr;
		}
		w0r=wr-sqrt(dwr*(wmax-wr));
		int Nur=ceil((wr-w0r)/dwr);
		w0r=wr-Nur*dwr;
		double dur=dwr/((wr-w0r)*(wr+dwr-w0r));
		wmax=1.0/dur+w0r;
		ivec ur_int=linspace<ivec>(Nur,1,Nur);
		vec ur(Nur);
		for (j=0; j<Nur; j++)
			ur(j)=dur*ur_int(j);
		
		vec wl_vec=1.0/ul+w0l;
		vec wr_vec=1.0/ur+w0r;
		Nw=Nul+Nur+Nwc;
		w.zeros(Nw);
		w.rows(0,Nul-1)=wl_vec;
		w.rows(Nul,Nul+Nwc-1)=wc;
		w.rows(Nul+Nwc,Nw-1)=wr_vec;
		w_exists=true;
		ws.zeros(2);
		ws(0)=w0l;
		ws(1)=w0r;
		Nw_lims.zeros(2);
		Nw_lims(0)=Nul;
		Nw_lims(1)=Nul+Nwc-1;
		
		Du_constant=true;
	}
	else if (uniform_grid)
	{
		if (!wc_exists)
		{
			cout<<"set_omega_grid() error: central part of grid undefined\n";
			return false;
		}
		w=wc;
		Nw=Nwc;
		w_exists=true;
		
		Du_constant=false;
	}
	else
	{
		if (!SW_set || !SC_set)
		{
			cout<<"set_omega_grid() error: SW or SC undefined\n";
			return false;
		}
		
		cout<<"gaussian grid density\n";
		
		fctPtr ptr=static_cast<fctPtr> (&OmegaMaxEnt_data::diff_gaussian);
		double init[3];
		double root[1];
		double par[3];
		double lims[2];
		double min_dg[1];
		double tol;
		double tol0=1e-12;
		
		double dw;
		if (step_omega_in.size())
		{
			dw=step_omega;
	//		if (peak_exists && dw>dw_peak)
	//		{
	//			cout<<"set_omega_grid() warning: step is larger than the estimated width of the peak at low energy\n";
	//		}
		}
		else
		{
			dw=SW/(f_SW_std_omega*Rmin_SW_dw);
			if (peak_exists)
			{
				if (dw_peak<dw)
					dw=2*dw_peak/R_peak_width_dw;
			}
		}
//		cout<<"dw: "<<dw<<endl;
		
		double width_gd;
		if (std_omega)
			width_gd=std_omega;
		else
			width_gd=SW/2;
		
		par[1]=dw;
		par[2]=pow(width_gd,2);

		double dwmax=50;
		double wmax=100;
//		double wmax=f_w_range*width_gd;
//		cout<<"wmax: "<<wmax<<endl;
		
		int max_grid_size=ceil(wmax/dw);
//		cout<<"max_grid_size: "<<max_grid_size<<endl;
		vec wp(max_grid_size,fill::zeros);
		j=0;
		par[0]=dw/2;
		
/*
		double x, y, y1;
		do
		{
			j++;
			lims[0]=par[0];
			lims[1]=wmax;
			if (diff_gaussian(lims[1], par)>0)
			{
				tol=tol0*diff_gaussian(lims[0], par);
				if (find_min_golden(ptr, lims, par, min_dg, tol))
				{
					if (diff_gaussian(min_dg[0], par)<0)
						lims[1]=min_dg[0];
					else
					{
						cout<<"par[0]: "<<par[0]<<endl;
						cout<<"min_dg[0]: "<<min_dg[0]<<endl;
						cout<<"diff_gaussian(min_dg[0], par): "<<diff_gaussian(min_dg[0], par)<<endl;
						break;
					}
				}
				else
				{
					cout<<"set_omega_grid() error from find_min_golden()\n";
					break;
				}
			}
//			init[0]=2*par[0]-wp(j-1);
//			init[1]=lims[0];
//			init[2]=3*par[0]-2*wp(j-1);
//			if (find_zero(ptr, init, par, root, lims, tol))
			tol=tol0;
			if (find_zero_bisect(ptr, par, root, lims, tol))
			{
				wp(j)=root[0];
//				cout<<wp(j)<<endl;
			}
			else
			{
				break;
			}
			par[0]=2*wp(j)-par[0];
			
		} while (j<max_grid_size && wp(j)<wmax);
	*/
		
		double invdwi, dwi, wi=dw/2, vargd=width_gd*width_gd;
		do
		{
			j++;
			invdwi=(1/dw-1/dwmax)*exp(-wi*wi/(2*vargd))+1/dwmax;
			dwi=1/invdwi;
//			dwi=dw*exp(wi*wi/(2*vargd));
			wp(j)=wp(j-1)+dwi;
			wi=2*wp(j)-wi;
		} while (j<max_grid_size && wp(j)<wmax);
		j--;
//		cout<<"wp\n"<<wp.rows(0,j)<<endl;
		wp=wp.rows(0,j);
		wp=wp+SC;
		vec wn=flipud(2*wp(0)-wp.rows(1,j));
		w=join_vert(wn,wp);
//		cout<<"w\n"<<w<<endl;
		Nw=2*j+1;
		wl=SC-f_SW_std_omega*width_gd/2;
		wr=SC+f_SW_std_omega*width_gd/2;
		i=j-1;
		while (i>0 && w(i)>wl) i--;
		wl=w(i+1);
//		cout<<"wl: "<<wl<<endl;
		Nw_lims.zeros(2);
		Nw_lims(0)=i+1;
		i=j+1;
		while (i<Nw-1 && w(i)<wr) i++;
		wr=w(i-1);
//		cout<<"wr: "<<wr<<endl;
		Nw_lims(1)=i;
		w0r=SC;
		w0l=SC;
		ws.zeros(2);
		ws(0)=w0l;
		ws(1)=w0r;
		w_origin=SC;
		w_origin_set=true;
		
		wc=w.rows(Nw_lims(0),Nw_lims(1));
		Nwc=wc.n_rows;
		wc_exists=true;
		
		w_exists=true;
		main_spectral_region_set=true;
		Du_constant=false;
		
	}
	
	return true;
}

double OmegaMaxEnt_data::diff_gaussian(double x, double par[])
{
	double xs=par[0];
	double c1=par[1];
	double c2=par[2];
//	double x0=par[3];

	if (x<=xs) x=xs+EPSILON;
	return log(c1/2)+x*x/(2*c2)-log(x-xs);
//	return (c1/2)*exp(x*x/(2*c2))-x+xs;
}


bool OmegaMaxEnt_data::set_wc()
{
	cout<<"definition of real frequency grid...\n";
	
	bool use_nu_grid=false;
	
	cout<<setiosflags(ios::left);
	
	double dw_min;


	if (non_uniform_grid && step_omega_in.size())
	{
		dw_min=step_omega;
		
		use_nu_grid=true;
	}
	else if (non_uniform_grid && peak_exists)
	{
		dw_min=2*dw_peak/R_peak_width_dw;
		
		use_nu_grid=true;
	}
	
	if (!use_nu_grid)
	{
		cout<<"using uniform grid in main spectral range\n";
		
		double dw;
		
		if (SW_set && SC_set && !main_spectral_region_set)
		{
			wl=SC-SW/2;
			wr=SC+SW/2;
			main_spectral_region_set=true;
		}
		else if (std_omega && SC_set && !main_spectral_region_set)
		{
			SW=f_SW_std_omega*std_omega;
			SW_set=true;
			wl=SC-SW/2;
			wr=SC+SW/2;
			main_spectral_region_set=true;
		}
		else if (main_spectral_region_set)
		{
			SW=wr-wl;
			SW_set=true;
		}
		else if (SW_set)
		{
			SC=0;
			SC_set=true;
			wr=SW/2;
			wl=-SW/2;
			main_spectral_region_set=true;
		}
		else
		{
			cout<<"The central part of the grid cannot be defined. Not enough information available.\n";
			return false;
		}
		
		if (!w_origin_set)
		{
			if (SC_set)
				w_origin=SC;
			else
				w_origin=0;
			w_origin_set=true;
		}
		
		if (step_omega_in.size())
		{
			dw=step_omega;
		//	if (peak_exists && dw>dw_peak)
		//	{
		//		cout<<"Warning: step is larger than the estimated width of the peak at low energy. You can use the parameters of section FREQUENCY GRID/ PARAMETERS to make the grid better adapted to the spectrum.\n";
		//	}
		}
		else
		{
			dw=SW/(f_SW_std_omega*Rmin_SW_dw);
			if (peak_exists)
			{
				if (dw_peak<dw)
					dw=dw_peak;
			}
		}
		
		if (w_origin<wl+dw || w_origin>wr-dw) w_origin=SC;
		
		uint Nwr=round((wr-w_origin)/dw);
		wr=w_origin+Nwr*dw;
		vec wcr=linspace<vec>(w_origin,wr,Nwr+1);
		uint Nwl=round((w_origin-wl)/dw);
		wl=w_origin-Nwl*dw;
		vec wcl=linspace<vec>(w_origin-dw,wl,Nwl);
		wcl=flipud(wcl);
		
		Nwc=Nwl+Nwr+1;
		
		wc.zeros(Nwc);
		wc.rows(0,Nwl-1)=wcl;
		wc.rows(Nwl,Nwc-1)=wcr;
		
		//cout<<"wl: "<<wl<<endl;
		//cout<<"wr: "<<wr<<endl;
		
		//wl=wc(0);
		//wr=wc(Nwc-1);
		dwl=dw;
		dwr=dw;
		wc_exists=true;
		main_spectral_region_set=true;
		
		return true;
	}
	else
	{
		cout<<"using non-uniform grid in main spectral range\n";
		
		if (!SW_set && !jfit && !std_omega)
		{
			cout<<"The central part of the grid cannot be defined. Not enough information available.\n";
			return false;
		}
		
		double SC_tmp=0;
		if (SC_set)
		{
			SC_tmp=SC;
		}
		
		if (!w_origin_in.size())
		{
			w_origin=0;
			w_origin_set=true;
		}
	
		double wr_tmp, wl_tmp;
		
		if (SW_in.size())
		{
			wr_tmp=SC_tmp+SW/2;
			wl_tmp=SC_tmp-SW/2;
		}
		else if (jfit)
		{
			wr_tmp=wn(jfit)/R_wncutoff_wr;
			wl_tmp=-wr_tmp;
		}
		else if (SW_set)
		{
			wr_tmp=SC_tmp+R_SW_wr*SW/2;
			wl_tmp=SC_tmp-R_SW_wr*SW/2;
		}
		else
		{
			wr_tmp=SC_tmp+R_SW_wr*f_SW_std_omega*std_omega/2;
			wl_tmp=SC_tmp-R_SW_wr*f_SW_std_omega*std_omega/2;
		}
		
		if (SW_set)
		{
			if (wr_tmp<SC_tmp+R_SW_wr*SW/2) wr_tmp=SC_tmp+R_SW_wr*SW/2;
			if (wl_tmp>SC_tmp-R_SW_wr*SW/2) wl_tmp=SC_tmp-R_SW_wr*SW/2;
			double wmax_tmp=SC_tmp+f_w_range*SW/2.0;
			if (wmax_tmp>0)
			{
				if (wr_tmp>wmax_tmp/R_wmax_wr_min) wr_tmp=wmax_tmp/R_wmax_wr_min;
			}
			double wmin_tmp=SC_tmp-f_w_range*SW/2.0;
			if (wmin_tmp<0)
			{
				if (wl_tmp<wmin_tmp/R_wmax_wr_min) wl_tmp=wmin_tmp/R_wmax_wr_min;
			}
		}
		
	//	cout<<"wr_tmp: "<<wr_tmp<<endl;
	//	cout<<"wl_tmp: "<<wl_tmp<<endl;
		
		rowvec grid_params_r, grid_params_l;
		
		grid_params_l.zeros(3);
		grid_params_r.zeros(3);
		
		grid_params_l(0)=w_origin;
		grid_params_r(0)=w_origin;
		
		grid_params_l(1)=dw_min;
		grid_params_r(1)=dw_min;
		
//		grid_params_l(1)=2*dw_peak/R_peak_width_dw;
//		grid_params_r(1)=2*dw_peak/R_peak_width_dw;
//		w_origin=0;

		rowvec params_tmp;
		int j=2;
		grid_params_l(2)=w_origin-R_Dw_dw*grid_params_l(1);
		grid_params_r(2)=w_origin+R_Dw_dw*grid_params_r(1);
		
	//	grid_params_l(2)=w_origin-R_Dw_dw*grid_params_l(1)/2;
	//	grid_params_r(2)=w_origin+R_Dw_dw*grid_params_r(1)/2;
		
	/*
		if (peak_exists)
		{
			if (grid_params_l(2)>w_origin-4*dw_peak) grid_params_l(2)=w_origin-4*dw_peak;
			if (grid_params_r(2)<w_origin+4*dw_peak) grid_params_r(2)=w_origin+4*dw_peak;
		}
	*/
		
		if (SW_set)
		{
			if (grid_params_r(1)>SW/(f_SW_std_omega*Rmin_SW_dw))
			{
				grid_params_l(1)=SW/(f_SW_std_omega*Rmin_SW_dw);
				grid_params_r(1)=SW/(f_SW_std_omega*Rmin_SW_dw);
			}
		}
		
//		cout<<"wr_tmp: "<<wr_tmp<<endl;
//		cout<<"wl_tmp: "<<wl_tmp<<endl;
		
		while (grid_params_r(j)<wr_tmp)
		{
			params_tmp.zeros(j+3);
			params_tmp.cols(0,j)=grid_params_r;
			grid_params_r=params_tmp;
			j++;
			grid_params_r(j)=2*grid_params_r(j-2);
			//			cout<<grid_params_r(j)<<endl;
			j++;
			grid_params_r(j)=grid_params_r(j-2)+R_Dw_dw*grid_params_r(j-1);
			//			cout<<grid_params_r(j)<<endl;
		}
		j=2;
		while (grid_params_l(j)>wl_tmp)
		{
			params_tmp.zeros(j+3);
			params_tmp.cols(0,j)=grid_params_l;
			grid_params_l=params_tmp;
			j++;
			grid_params_l(j)=2*grid_params_l(j-2);
			//			cout<<grid_params_l(j)<<endl;
			j++;
			grid_params_l(j)=grid_params_l(j-2)-R_Dw_dw*grid_params_l(j-1);
			//			cout<<grid_params_l(j)<<endl;
		}
		
//		cout<<"grid_params_l\n"<<grid_params_l<<endl;
//		cout<<"grid_params_r\n"<<grid_params_r<<endl;
		
		grid_params_l=fliplr(grid_params_l);
		omega_grid_params=join_rows(grid_params_l.cols(0,grid_params_l.n_cols-2),grid_params_r.cols(2,grid_params_r.n_cols-1));
		
//		cout<<"omega_grid_params:\n"<<omega_grid_params<<endl;

//		cout<<"SC, SW: "<<setw(20)<<SC<<SW<<endl;
		
		return set_grid_from_params();
	}
}

/*
bool OmegaMaxEnt_data::set_wc()
{
	double dw;
	
	if (SW_set && SC_set && !main_spectral_region_set)
	{
		wl=SC-SW/2;
		wr=SC+SW/2;
		main_spectral_region_set=true;
	}
	else if (std_omega && SC_set && !main_spectral_region_set)
	{
		SW=f_SW_std_omega*std_omega;
		SW_set=true;
		wl=SC-SW/2;
		wr=SC+SW/2;
		main_spectral_region_set=true;
	}
	else if (!main_spectral_region_set)
	{
		cout<<"The central part of the grid cannot be defined. Not enough information available.\n";
		return false;
	}
	if (!w_origin_in.size())	w_origin=SC;
	if (step_omega_in.size())
	{
		dw=step_omega;
		if (peak_exists)
		{
			if (dw>dw_peak)
			{
				cout<<"warning: step is larger than the estimated width of the peak at low energy\n";
			}
		}
	}
	else
	{
		//dw=SW/(2.0*Nw_min);
		dw=SW/Rmin_SW_dw;
		if (peak_exists)
		{
			if (dw_peak<dw)
				dw=dw_peak;
		}
	}

	if (w_origin<wl+dw || w_origin>wr-dw) w_origin=SC;
		
	uint Nwr=round((wr-w_origin)/dw);
	wr=w_origin+Nwr*dw;
	vec wcr=linspace<vec>(w_origin,wr,Nwr+1);
	uint Nwl=round((w_origin-wl)/dw);
	wl=w_origin-Nwl*dw;
	vec wcl=linspace<vec>(w_origin-dw,wl,Nwl);
	wcl=flipud(wcl);
	
	Nwc=Nwl+Nwr+1;
	
	wc.zeros(Nwc);
	wc.rows(0,Nwl-1)=wcl;
	wc.rows(Nwl,Nwc-1)=wcr;
	
	//cout<<"wl: "<<wl<<endl;
	//cout<<"wr: "<<wr<<endl;
	
	//wl=wc(0);
	//wr=wc(Nwc-1);
	dwl=dw;
	dwr=dw;
	wc_exists=true;
	main_spectral_region_set=true;
	
	return true;
}
*/

/*
bool OmegaMaxEnt_data::non_uniform_frequency_grid(rowvec w_steps, rowvec wlims, double w0, vec R, vec &grid)
{
	bool ordered=true, grid_set=true;
	
	uint Ndw=w_steps.n_cols;
	uint Nlims=wlims.n_cols;
	
	if (Nlims<=Ndw)
		Ndw=Nlims-1;
	else if (Nlims>Ndw+1)
		Nlims=Ndw+1;
	
	double RW=R(0);
	double RWD=R(1);
	
	int j;
	for (j=1; j<Nlims; j++)
	{
		if (wlims(j)<wlims(j-1)) ordered=false;
	}
	
	if (ordered )
	{
		double D, wc_j, dwtmp;
		int max_grid_size=ceil((wlims(Nlims-1)-wlims(0))/w_steps.min());
		int j0, l, Nm, Np;
		if (w0>wlims(0) && w0<wlims(Nlims-1))
		{
			j=0;
			while (w0>=wlims(j+1))	j=j+1;
			j0=j;
			double wstart=w0;
//			double wstart=(wlims(j)+wlims(j+1))/2.0;
			vec wp(max_grid_size);
			wp(0)=wstart;
			l=0;
			for (j=j0; j<Ndw-1; j++)
			{
				D=(wlims(j+2)-wlims(j))/(RW*RWD);
				wc_j=wlims(j+1)+(wlims(j+2)-2*wlims(j+1)+wlims(j))/(2*RW);
				while (wp(l)<(wlims(j+2)+wlims(j+1))/2.0)
				{
					dwtmp=w_steps(j+1)+(w_steps(j)-w_steps(j+1))/(exp((wp(l)-wc_j)/D)+1.0);
					wp(l+1)=wp(l)+dwtmp;
					l=l+1;
				}
			}
			while (wp(l)<wlims(Nlims-1))
			{
				wp(l+1)=wp(l)+w_steps(Ndw-1);
				l=l+1;
			}
			Np=l+1;
			vec wm(max_grid_size);
			wm(0)=wstart;
			l=0;
			for (j=j0-1; j>=0; j--)
			{
				D=(wlims(j+2)-wlims(j))/(RW*RWD);
				wc_j=wlims(j+1)+(wlims(j+2)-2*wlims(j+1)+wlims(j))/(2*RW);
				while (wm(l)>(wlims(j)+wlims(j+1))/2)
				{
					dwtmp=w_steps(j+1)+(w_steps(j)-w_steps(j+1))/(exp((wm(l)-wc_j)/D)+1.0);
					wm(l+1)=wm(l)-dwtmp;
					l=l+1;
				}
			}
			while (wm(l)>wlims(0))
			{
				wm(l+1)=wm(l)-w_steps(0);
				l=l+1;
			}
			Nm=l+1;
			grid.zeros(Nm+Np-1);
			grid.rows(0,Nm-1)=flipud(wm.rows(0,Nm-1));
			grid.rows(Nm,Nm+Np-2)=wp.rows(1,Np-1);
		}
		else if (w0==wlims(0))
		{
			vec wp(max_grid_size);
			wp(0)=w0;
			j0=0;
			l=0;
			for (j=j0; j<Ndw-1; j++)
			{
				D=(wlims(j+2)-wlims(j))/(RW*RWD);
				wc_j=wlims(j+1)+(wlims(j+2)-2*wlims(j+1)+wlims(j))/(2*RW);
				while (wp(l)<(wlims(j+2)+wlims(j+1))/2.0)
				{
					dwtmp=w_steps(j+1)+(w_steps(j)-w_steps(j+1))/(exp((wp(l)-wc_j)/D)+1.0);
					wp(l+1)=wp(l)+dwtmp;
					l=l+1;
				}
			}
			while (wp(l)<wlims(Nlims-1))
			{
				wp(l+1)=wp(l)+w_steps(Ndw-1);
				l=l+1;
			}
			Np=l+1;
			grid=wp.rows(0,Np-1);
		}
		else if (w0==wlims(Nlims-1))
		{
			j0=Nlims-2;
			vec wm(max_grid_size);
			wm(0)=w0;
			l=0;
			for (j=j0-1; j<=0; j--)
			{
				D=(wlims(j+2)-wlims(j))/(RW*RWD);
				wc_j=wlims(j+1)+(wlims(j+2)-2*wlims(j+1)+wlims(j))/(2*RW);
				while (wm(l)>(wlims(j)+wlims(j+1))/2.0)
				{
					dwtmp=w_steps(j+1)+(w_steps(j)-w_steps(j+1))/(exp((wm(l)-wc_j)/D)+1.0);
					wm(l+1)=wm(l)-dwtmp;
					l=l+1;
				}
			}
			while (wm(l)>wlims(0))
			{
				wm(l+1)=wm(l)-w_steps(0);
				l=l+1;
			}
			Nm=l+1;
			grid=flipud(wm.rows(0,Nm-1));
		}
		else
		{
			cout<<"grid origin is outside grid boundaries\n";
			grid_set=false;
		}
	}
	else
	{
		cout<<"non_uniform_frequency_grid(): interval boundaries are not strictly increasing values\n";
		grid_set=false;
	}
	
	return grid_set;
}
*/

bool OmegaMaxEnt_data::non_uniform_frequency_grid(rowvec w_steps, rowvec wlims, double w0, vec R, vec &grid)
{
	bool ordered=true, grid_set=true;
	
//	cout<<"defining non-uniform grid\n";
	
	int dj=3;
	
	uint Ndw=w_steps.n_cols;
	uint Nlims=wlims.n_cols;
	
	if (Nlims<=Ndw)
		Ndw=Nlims-1;
	else if (Nlims>Ndw+1)
		Nlims=Ndw+1;
	
	double RW=R(0);
	double RWD=R(1);
	
//	cout<<"RW: "<<RW<<endl;
//	cout<<"RWD: "<<RWD<<endl;
	
	int j;
	for (j=1; j<Nlims; j++)
	{
		if (wlims(j)<wlims(j-1)) ordered=false;
	}
	
	if (ordered )
	{
		int max_grid_size=ceil((wlims(Nlims-1)-wlims(0))/w_steps.min());
		int j0, k, l, Nm, Np;
		double dwtmp;
		double *D=new double[Ndw];
		double *wc_j=new double[Ndw];
		int kmax, kmin;
		
		if (w0>wlims(0) && w0<wlims(Nlims-1))
		{
			j=0;
			while (w0>=wlims(j+1))	j++;
			j0=j;
			vec wp(max_grid_size);
			wp(0)=w0;
			l=0;
			
			for (j=0; j<Ndw-1; j++)
			{
				D[j]=(wlims(j+2)-wlims(j))/(RW*RWD);
				wc_j[j]=wlims(j+1)+(wlims(j+2)-2*wlims(j+1)+wlims(j))/(2*RW);
			}
			
			for (j=j0; j<Ndw; j++)
			{
				while (wp(l)<wlims(j+1))
				{
					kmax=j+dj;
					while (kmax>=Ndw) kmax--;
					kmin=j-dj+1;
					while (kmin<0) kmin++;
					dwtmp=w_steps(kmax);
					for (k=kmax-1; k>=kmin; k--)
					{
						dwtmp+=(w_steps(k)-w_steps(k+1))/(exp((wp(l)-wc_j[k])/D[k])+1.0);
					}
					wp(l+1)=wp(l)+dwtmp;
					l=l+1;
				}
			}
			Np=l+1;
			
			vec wm(max_grid_size);
			wm(0)=w0;
			l=0;
			for (j=j0; j>=0; j--)
			{
				while (wm(l)>wlims(j))
				{
					kmax=j+dj;
					while (kmax>=Ndw) kmax--;
					kmin=j-dj+1;
					while (kmin<0) kmin++;
					dwtmp=w_steps(kmax);
					for (k=kmax-1; k>=kmin; k--)
					{
						dwtmp+=(w_steps(k)-w_steps(k+1))/(exp((wm(l)-wc_j[k])/D[k])+1.0);
					}
					wm(l+1)=wm(l)-dwtmp;
					l=l+1;
				}
			}
			Nm=l+1;
			grid.zeros(Nm+Np-1);
			grid.rows(0,Nm-1)=flipud(wm.rows(0,Nm-1));
			grid.rows(Nm,Nm+Np-2)=wp.rows(1,Np-1);
		}
		else if (w0==wlims(0))
		{
			vec wp(max_grid_size);
			wp(0)=w0;
			j0=0;
			l=0;
			
			for (j=0; j<Ndw-1; j++)
			{
				D[j]=(wlims(j+2)-wlims(j))/(RW*RWD);
				wc_j[j]=wlims(j+1)+(wlims(j+2)-2*wlims(j+1)+wlims(j))/(2*RW);
			}
			
			for (j=j0; j<Ndw; j++)
			{
				while (wp(l)<wlims(j+1))
				{
					kmax=j+dj;
					while (kmax>=Ndw) kmax--;
					kmin=j-dj+1;
					while (kmin<0) kmin++;
					dwtmp=w_steps(kmax);
					for (k=kmax-1; k>=kmin; k--)
					{
						dwtmp+=(w_steps(k)-w_steps(k+1))/(exp((wp(l)-wc_j[k])/D[k])+1.0);
					}
					wp(l+1)=wp(l)+dwtmp;
					l=l+1;
				}
			}
			Np=l+1;
			grid=wp.rows(0,Np-1);
		}
		else if (w0==wlims(Nlims-1))
		{
			j0=Nlims-1;
			vec wm(max_grid_size);
			wm(0)=w0;
			l=0;
			for (j=j0; j>=0; j--)
			{
				while (wm(l)>wlims(j))
				{
					kmax=j+dj;
					while (kmax>=Ndw) kmax--;
					kmin=j-dj+1;
					while (kmin<0) kmin++;
					dwtmp=w_steps(kmax);
					for (k=kmax-1; k>=kmin; k--)
					{
						dwtmp+=(w_steps(k)-w_steps(k+1))/(exp((wm(l)-wc_j[k])/D[k])+1.0);
					}
					wm(l+1)=wm(l)-dwtmp;
					l=l+1;
				}
			}
			Nm=l+1;
			grid=flipud(wm.rows(0,Nm-1));
		}
		else
		{
			cout<<"grid origin is outside grid boundaries\n";
			grid_set=false;
		}
	}
	else
	{
		cout<<"non_uniform_frequency_grid(): interval boundaries are not strictly increasing values\n";
		grid_set=false;
	}
	
	return grid_set;
}

bool OmegaMaxEnt_data::set_grid_from_params()
{
	int j;
	bool grid_set=false;
	cout<<"grid parameters provided\n";
	uint Npar_grid=omega_grid_params.n_cols;
	if ( Npar_grid % 2 )
	{
		uint Nint=(Npar_grid-1)/2;
		uint Nlims=Nint+1;
		uvec odd_ind=linspace<uvec>(1,Npar_grid-2,Nint);
		rowvec dw_par=omega_grid_params.cols(odd_ind);
		bool Rdw_too_large=false, Rdw_too_small=false;
		for (j=0; j<Nint-1; j++)
		{
			if (dw_par(j+1)/dw_par(j)>Rdw_max) Rdw_too_large=true;
			if (dw_par(j+1)/dw_par(j)<(1.0/Rdw_max)) Rdw_too_small=true;
		}
		if (Rdw_too_large || Rdw_too_small)
		{
			cout<<"The ratio of frequency steps in adjacent intervals of \"grid parameters\" is too large. Spurious oscillations may appear in the spectral function. The maximum ratio recommended is "<<Rdw_max<<endl;
		}
		uvec even_ind=linspace<uvec>(0,Npar_grid-1,Nlims);
		rowvec wlims_par=omega_grid_params.cols(even_ind);
		
//		cout<<"Dw: "<<dw_par<<endl;
//		cout<<"wlims: "<<wlims_par<<endl;
		bool ordered=true;
		for (j=1; j<Nlims; j++)
		{
			if (wlims_par(j)<wlims_par(j-1)) ordered=false;
		}
		if (ordered)
		{
			bool grid_par_ok=true;
			bool grid_step_pos=true;
			for (j=1; j<Nlims; j++)
			{
				if (dw_par(j-1)>=(wlims_par(j)-wlims_par(j-1))/Rmin_Dw_dw) grid_par_ok=false;
				if (dw_par(j-1)<0)
				{
					grid_par_ok=false;
					grid_step_pos=false;
				}
			}
			if (grid_par_ok)
			{
				double w0tmp=(wlims_par(0)+wlims_par(Nlims-1))/2.0;
				j=0;
				while (w0tmp>=wlims_par(j+1))
				{
					j=j+1;
				}
				double w0_par=(wlims_par(j)+wlims_par(j+1))/2.0;
//				if (w_origin_in.size())
				if (w_origin_set)
				{
					w0_par=w_origin;
				}
				else
				{
					w_origin=w0_par;
					w_origin_set=true;
				}
				vec R(2);
				R(0)=RW_grid;
				R(1)=RWD_grid;
				if (non_uniform_frequency_grid(dw_par, wlims_par, w0_par, R, wc))
				{
					wc_exists=true;
					Nwc=wc.n_rows;
					wl=wc(0);
					wr=wc(Nwc-1);
					main_spectral_region_set=true;
					dwl=wc(1)-wc(0);
					dwr=wc(Nwc-1)-wc(Nwc-2);
					if (!SW_set)
					{
						SW=wr-wl;
						SW_set=true;
					}
					if (!SC_set)
					{
						SC=(wl+wr)/2;
						SC_set=true;
					}
					grid_set=true;
				}
				else
				{
					cout<<"parameterized grid definition failed\n";
				}
			}
			else
			{
				if (grid_step_pos)
					cout<<"Error: the frequency step in \"grid parameters\" is too large in at least one interval. The step must be at least "<<Rmin_Dw_dw<<" times smaller than the interval size.\n";
				else
					cout<<"Error: negative step value found in \"grid parameters\"\n";
			}
		}
		else
		{
			cout<<"Error: interval boundaries in \"grid parameters\" are not strictly increasing values\n";
		}
	}
	else
	{
		cout<<"Error: \"grid parameters\" must have an odd number of elements\n";
	}
	
	return grid_set;
}

bool OmegaMaxEnt_data::set_grid_omega_from_file()
{
	bool grid_set=false;
	
	int j, jl, jr;
	cout<<"real frequency grid file provided\n";
	
	Nw=grid_w_data.n_rows;
	vec grid_w=grid_w_data.col(0);
	if (Nw<Nw_min)
	{
		cout<<"warning: size of privided real frequency grid must be larger than "<<Nw_min<<endl;
	}
	double wmin=grid_w(0), wmax=grid_w(Nw-1);
	int j1=0, j2=1, j3=2;
	w0l=(2*grid_w(j1)*grid_w(j3)-grid_w(j1)*grid_w(j2)-grid_w(j2)*grid_w(j3))/(grid_w(j1)-2*grid_w(j2)+grid_w(j3));
	j1=Nw-3;
	j2=Nw-2;
	j3=Nw-1;
	w0r=(2*grid_w(j1)*grid_w(j3)-grid_w(j1)*grid_w(j2)-grid_w(j2)*grid_w(j3))/(grid_w(j1)-2*grid_w(j2)+grid_w(j3));
	if (w0l<wmin || w0r>wmax)
	{
		cout<<"The real frequency grid in file "<<grid_omega_file<<" was not generated by this code,\n";
		if (!main_spectral_region_set)
		{
			wl=grid_w(0)+EPSILON;
			wr=grid_w(Nw-1)-EPSILON;
		}
		
		cout<<"only the part in the main spectral region will be used.\n";
		
		if (grid_w(0)<wl && grid_w(Nw-1)>wr)
		{
			j=0;
			while (grid_w(j)<wl && j<Nw-1)
			{
				j=j+1;
			}
			jl=j;
			if ((grid_w(j)-wl)>(wl-grid_w(j-1)))
			{
				jl=j-1;
			}
			wl=grid_w(jl);
			dwl=grid_w(jl+1)-grid_w(jl);
			while (grid_w(j)<wr && j<Nw-1)
			{
				j=j+1;
			}
			jr=j;
			if ((grid_w(j)-wr)>(wr-grid_w(j-1)))
			{
				jr=j-1;
			}
			wr=grid_w(jr);
			main_spectral_region_set=true;
			dwr=grid_w(jr)-grid_w(jr-1);
			wc=grid_w.rows(jl,jr);
			wc_exists=true;
			Nwc=wc.n_rows;
			
//			if (!SW_set)
//			{
				SW=wr-wl;
				SW_set=true;
//			}
			if (!SC_set)
			{
				SC=(wl+wr)/2.0;
				SC_set=true;
			}
			grid_set=true;
//			if (!w_origin_in.size())
			if (!w_origin_set)
			{
				w_origin=SC;
				w_origin_set=true;
			}
			j=0;
			while (abs(w_origin-wc(j+1))<abs(w_origin-wc(j)) && j<Nwc-2) j=j+1;
			w_origin=wc(j);
		}
		else
		{
			cout<<"The provided real frequency grid must extend beyond the spectral function frequency range.\n";
			cout<<"The grid cannot be used.\n";
		}
	}
	else
	{
		Nw_lims.zeros(2);
		ws.zeros(2);
		ws(0)=w0l;
		ws(1)=w0r;
		w=grid_w;
		int j=0;
		double dw1=w(j+1)-w(j);
		double dw2=w(j+2)-w(j+1);
		double rdw=dw2/dw1;
		while (abs(rdw-1.0)>tol_rdw)
		{
			j=j+1;
			dw1=dw2;
			dw2=w(j+2)-w(j+1);
			rdw=dw2/dw1;
		}
		Nw_lims(0)=j+1;
//		int Nu_l=j+1;
		dwl=dw2;
		wl=w(j+1);
		j=Nw-3;
		dw1=w(j+1)-w(j);
		dw2=w(j+2)-w(j+1);
		rdw=dw2/dw1;
		while (abs(rdw-1.0)>tol_rdw)
		{
			j=j-1;
			dw2=dw1;
			dw1=w(j+1)-w(j);
			rdw=dw2/dw1;
		}
		Nw_lims(1)=j+1;
//		int Nu_r=Nw-j-2;
		dwr=dw1;
		wr=w(j+1);
		wc=w.rows(Nw_lims(0),Nw_lims(1));
		Nwc=wc.n_rows;
		main_spectral_region_set=true;
//		if (!SW_set)
//		{
			SW=wr-wl;
			SW_set=true;
//		}
		if (!SC_set)
		{
			SC=(wl+wr)/2.0;
			SC_set=true;
		}
		w_exists=true;
		wc_exists=true;
		grid_set=true;
//		if (!w_origin_in.size()) w_origin=SC;
		if (!w_origin_set)
		{
			w_origin=SC;
			w_origin_set=true;
		}
		j=0;
		while (abs(w_origin-wc(j+1))<abs(w_origin-wc(j)) && j<Nwc-2) j=j+1;
		w_origin=wc(j);
		
		Du_constant=true;
	}
	
	return grid_set;
}

bool OmegaMaxEnt_data::set_initial_spectrum()
{
	Nw_lims.zeros(2);
	ws.zeros(2);
	
	bool A0_loaded=false;

	cout<<"initial spectral function provided\n";
	
	w=Aw_data.col(0);
	Nw=w.n_rows;
	A0=Aw_data.col(1);
	A0=A0.rows(1,Nw-2);
	double wmin=w(0), wmax=w(Nw-1);
	int j1=0, j2=1, j3=2;
	w0l=(2*w(j1)*w(j3)-w(j1)*w(j2)-w(j2)*w(j3))/(w(j1)-2*w(j2)+w(j3));
	j1=Nw-3;
	j2=Nw-2;
	j3=Nw-1;
	w0r=(2*w(j1)*w(j3)-w(j1)*w(j2)-w(j2)*w(j3))/(w(j1)-2*w(j2)+w(j3));
	if (w0l<wmin || w0r>wmax)
	{
		cout<<"the real frequency grid in file "<<init_spectr_func_file<<" was not generated by this code\n";
		cout<<"the spectrum in that file cannot be used as the initial one in the calculation\n";
	}
	else
	{
		int j=0;
		double dw1=w(j+1)-w(j);
		double dw2=w(j+2)-w(j+1);
		double rdw=dw2/dw1;
		while (abs(rdw-1.0)>tol_rdw)
		{
			j=j+1;
			dw1=dw2;
			dw2=w(j+2)-w(j+1);
			rdw=dw2/dw1;
		}
		Nw_lims(0)=j+1;
//		int Nu_l=j+1;
		dwl=dw2;
		wl=w(j+1);
		j=Nw-3;
		dw1=w(j+1)-w(j);
		dw2=w(j+2)-w(j+1);
		rdw=dw2/dw1;
		while (abs(rdw-1.0)>tol_rdw)
		{
			j=j-1;
			dw2=dw1;
			dw1=w(j+1)-w(j);
			rdw=dw2/dw1;
		}
		Nw_lims(1)=j+1;
//		int Nu_r=Nw-j-2;
		dwr=dw1;
		wr=w(j+1);
		wc=w.rows(Nw_lims(0),Nw_lims(1));
		A0_loaded=true;
		Nwc=wc.n_rows;
//		if (!SW_set)
//		{
			SW=wr-wl;
			SW_set=true;
//		}
		if (!SC_set)
		{
			SC=(wl+wr)/2.0;
			SC_set=true;
		}
		ws(0)=w0l;
		ws(1)=w0r;
		
		if (displ_prep_figs)
		{
			graph_2D g1;
			
			vec x=w.rows(1,Nw-2);
			vec y=A0;
			
			char xl[]="$\\\\omega$", yl[]="A0";
			plot(g1, x, y, xl, yl);
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		w_exists=true;
		wc_exists=true;
		main_spectral_region_set=true;
		
//		if (!w_origin_in.size()) w_origin=SC;
		if (!w_origin_set)
		{
			w_origin=SC;
			w_origin_set=true;
		}
		j=0;
		while (abs(w_origin-wc(j+1))<abs(w_origin-wc(j)) && j<Nwc-2) j=j+1;
		w_origin=wc(j);
		
		Du_constant=true;
	}

	return A0_loaded;
}

bool OmegaMaxEnt_data::test_low_energy_peak_fermions()
{
//	if (Nwn_test_metal>Nn-2)	Nwn_test_metal=Nn-2;
	
	int Nn_min_pk=12;
	
	peak_exists=false;
	
	if (Nn<Nn_min_pk) return false;

	cout<<"Looking for a peak in the spectral function at low energy...\n";
	
	int nmax=n(Nn-1);
	
	int NCnmin=2;
	int NCnmax=3;
	int NNCn=NCnmax-NCnmin+1;
	ivec NCn=linspace<ivec>(NCnmin,NCnmax,NNCn);
	
	int NCpmin=1;
	int NCpmax=15;
	if (NCpmax>(nmax-NCn.max()-2))
	{
		NCpmax=nmax-NCn.max()-2;
	}
	int NNCp=NCpmax-NCpmin+1;
	ivec NCp=linspace<ivec>(NCpmin,NCpmax,NNCp);
	
	int DNwn=2;
	
	int p0_min=1;
	int p0_max=1;
	int Np=p0_max-p0_min+1;
	ivec p0=linspace<ivec>(p0_min,p0_max,Np);
	
	imat NCmin(Np,2);
	
	vec norm_lf_min(Np,fill::zeros);
	vec std_peak_min(Np,fill::zeros);
	vec mean_omega_lf_min(Np,fill::zeros);
	vec mean_omega2_lf_min(Np,fill::zeros);
	vec chi2_lf_min(Np,fill::zeros);
	vec varM2_q(Np,fill::zeros);
	vec varM0_q(Np,fill::zeros);
	
	mat X, CG, P, invCG, AM, BM, AMP, BMP, MP, Mtmp, chi2tmp;
	mat norm_lf(NNCp,NNCn), std_peak(NNCp,NNCn), mean_omega_lf(NNCp,NNCn), mean_omega2_lf(NNCp,NNCn), chi2_lf(NNCp,NNCn), M2_lf(NNCp,NNCn);
	int j, l, m, p, Nfit;
	uword q;
	vec Gtmp, Glftmp, diffG;
	rowvec maxX;
	double vartmp;
	
	//		char UPLO='U';
	//		int NA, NRHS=1, INFO;
	
	//		mat test_M, AM2;
	//		vec BM2;
	
	
	for (q=0; q<Np; q++)
	{
		norm_lf.zeros();
		std_peak.zeros();
		mean_omega_lf.zeros();
		mean_omega2_lf.zeros();
		chi2_lf.zeros();
		M2_lf.zeros();
		
		
		for (l=0; l<NNCp; l++)
		{
			for (m=0; m<NNCn; m++)
			{
				Nfit=NCp(l)+NCn(m)+1+DNwn;
				
				X.zeros(2*Nfit,2*(NCp(l)+NCn(m)+1));
				
				for (j=NCp(l); j>=-NCn(m); j--)
				{
					for (p=p0(q); p<=Nfit+p0(q)-1; p++)
					{
						X(2*(p-p0(q)),2*NCp(l)-2*j+1)=pow(-1,j)*pow(wn(p-1),2*j);
						X(2*(p-p0(q))+1,2*NCp(l)-2*j)=pow(-1,j)*pow(wn(p-1),2*j+1);
					}
				}
				
				Gtmp=Gchi2.rows(2*p0(q)-2,2*(Nfit+p0(q))-3);
				CG=COV.submat(2*p0(q)-2,2*p0(q)-2,2*(Nfit+p0(q))-3,2*(Nfit+p0(q))-3);
				
				invCG=inv(CG);
				//	invCG=inv_sympd(CG);
				AM=(X.t())*invCG*X;
				BM=(X.t())*invCG*Gtmp;
				//					BM2=BM;
				//					AM2=AM;
				
				Mtmp=solve(AM,BM);
				
				//					cout<<l<<" "<<m<<endl;
				
				//					NA=AM.n_rows;
				//					dposv_(&UPLO, &NA, &NRHS, AM.memptr(), &NA, BM.memptr(), &NA, &INFO);
				//					Mtmp=BM;
				
				//					dposv_(&UPLO, &NA, &NRHS, AM2.memptr(), &NA, BM2.memptr(), &NA, &INFO);
				//					test_M.zeros(NA,2);
				//					test_M.col(0)=Mtmp;
				//					test_M.col(1)=BM2;
				//					cout<<test_M<<endl;
				
				//					maxX=max(abs(X),0);
				//					P=diagmat(1.0/maxX);
				//					AMP=(P*(AM*P)+(P*AM)*P)/2;
				//					BMP=P*BM;
				//					dposv_(&UPLO, &NA, &NRHS, AMP.memptr(), &NA, BMP.memptr(), &NA, &INFO);
				//					MP=BMP;
				//					MP=solve(AMP,BMP);
				//					Mtmp=P*MP;
				
				Glftmp=X*Mtmp;
				
				diffG=Gtmp-Glftmp;
				
				chi2tmp=((diffG.t())*invCG*diffG)/(2.0*Nfit);
				
				chi2_lf(l,m)=chi2tmp(0,0);
				
				norm_lf(l,m)=Mtmp(2*NCp(l)+2);
				mean_omega_lf(l,m)=Mtmp(2*NCp(l)+3)/norm_lf(l,m);
				mean_omega2_lf(l,m)=Mtmp(2*NCp(l)+4)/norm_lf(l,m);
				vartmp=mean_omega2_lf(l,m)-pow(mean_omega_lf(l,m),2);
				if (vartmp>0)
					std_peak(l,m)=sqrt(vartmp);
				M2_lf(l,m)=Mtmp(2*NCp(l)+4);
				
				//					cout<<setw(30)<<"norm_lf: "<<setw(20)<<norm_lf(l,m)<<"	"<<BM2(2*NCp(l)+2)<<endl;
				//					cout<<setw(30)<<"mean_omega_lf: "<<setw(20)<<mean_omega_lf(l,m)<<"	"<<BM2(2*NCp(l)+3)/BM2(2*NCp(l)+2)<<endl;
				//					cout<<setw(30)<<"mean_omega2_lf: "<<setw(20)<<mean_omega2_lf(l,m)<<"	"<<BM2(2*NCp(l)+4)/BM2(2*NCp(l)+2)<<endl;
				
			}
		}
		
		for (l=0; l<NNCp; l++)
		{
			for (m=0; m<NNCn; m++)
			{
				if (std_peak(l,m)==0)
				{
					chi2_lf(l,m)=max(max(chi2_lf));
				}
			}
		}
		
		//			cout<<"std_peak:\n"<<std_peak<<endl;
		//			cout<<"norm_lf:\n"<<norm_lf<<endl;
		//			cout<<"mean_omega_lf:\n"<<mean_omega_lf<<endl;
		
		uint NvM=2;
		uint NvarM=NNCp-2*NvM;
		mat varM0(NvarM,NNCn,fill::zeros);
		mat varM2(NvarM,NNCn,fill::zeros);
		for (j=NvM; j<NNCp-NvM; j++)
		{
			varM0.row(j-NvM)=var(norm_lf.rows(j-NvM,j+NvM));
			varM2.row(j-NvM)=var(M2_lf.rows(j-NvM,j+NvM));
		}
		
		uword indpM0, indnM0;
		
		double varM0_min=varM0.min(indpM0,indnM0);
		double varM2_min=varM2.min();
		
		l=indpM0+NvM;
		m=indnM0;
		
		NCmin(q,0)=l;
		NCmin(q,1)=m;
		
		norm_lf_min(q)=norm_lf(l,m);
		mean_omega_lf_min(q)=mean_omega_lf(l,m);
		mean_omega2_lf_min(q)=mean_omega2_lf(l,m);
		std_peak_min(q)=std_peak(l,m);
		chi2_lf_min(q)=chi2_lf(l,m);
		varM0_q(q)=varM0_min;
		varM2_q(q)=varM2_min;
		
		if (displ_adv_prep_figs)
		{
			graph_2D g1, g2, g3, g4;
			
			vec x(NNCp), y;
			
			for (l=0; l<NNCp; l++)
				x(l)=NCp(l);
			
			char xl[]="NCp";
			char yl[]="norm_lf";
			char yl2[]="std_peak_lf";
			char yl3[]="mean_omega_lf";
			char yl4[]="chi2_lf";
			
			double xlims[2], ylims[2], xlims2[2], ylim2[2], xlims3[2], ylims3[2], xlims4[2], ylims4[2], ymin, ymax;
			
			xlims[0]=x.min();
			xlims[1]=x.max();
			
			char lgd_entry[100];
			
			for (m=0; m<NNCn; m++)
			{
				sprintf(lgd_entry,"NCn=%d",(int)NCn(m));
				
				y=norm_lf.col(m);
				ymax=y.max();
				ymin=y.min();
			//	ylims[0]=ymin-0.1*(ymax-ymin);
			//	ylims[1]=ymax+0.1*(ymax-ymin);
				g1.add_data(x.memptr(),y.memptr(),x.n_rows);
				g1.add_to_legend(lgd_entry);
				
				y=std_peak.col(m);
				ymax=y.max();
				ymin=y.min();
			//	ylims2[0]=ymin-0.1*(ymax-ymin);
			//	ylims2[1]=ymax+0.1*(ymax-ymin);
				g2.add_data(x.memptr(),y.memptr(),x.n_rows);
				g2.add_to_legend(lgd_entry);
				
				y=mean_omega_lf.col(m);
				ymax=y.max();
				ymin=y.min();
			//	ylims3[0]=ymin-0.1*(ymax-ymin);
			//	ylims3[1]=ymax+0.1*(ymax-ymin);
				g3.add_data(x.memptr(),y.memptr(),x.n_rows);
				g3.add_to_legend(lgd_entry);
				
				y=chi2_lf.col(m);
				ymax=y.max();
				ymin=y.min();
			//	ylims4[0]=ymin-0.1*(ymax-ymin);
			//	ylims4[1]=ymax+0.1*(ymax-ymin);
				g4.add_data(x.memptr(),y.memptr(),x.n_rows);
				g4.add_to_legend(lgd_entry);
			}
			g1.set_axes_lims(xlims,NULL);
			g1.set_axes_labels(xl,yl);
			g1.curve_plot();
			
			g2.set_axes_lims(xlims,NULL);
			g2.set_axes_labels(xl,yl2);
			g2.curve_plot();
			
			g3.set_axes_lims(xlims,NULL);
			g3.set_axes_labels(xl,yl3);
			g3.curve_plot();
			
			g4.set_axes_lims(xlims,NULL);
			g4.set_axes_labels(xl,yl4);
			g4.curve_plot();
			
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
			
		}
	}
	
	double varM0_min=varM0_q.min(q);
	
	double peak_weight=norm_lf_min(q);
	double peak_center=mean_omega_lf_min(q);
	double peak_width=std_peak_min(q);
	//		double chi2_min=chi2_lf_min(q);
	double m2_lf_min=mean_omega2_lf_min(q);
	
//	cout<<"M0: "<<M0<<endl;
//	cout<<"peak_weight: "<<peak_weight<<endl;
//	cout<<"peak_width: "<<peak_width<<endl;
//	cout<<"sqrt(varM0_min): "<<sqrt(varM0_min)<<endl;
	
	if (peak_width>100*EPSILON && peak_weight>peak_weight_min*M0 && sqrt(varM0_min)/peak_weight<std_norm_peak_max)
	{
		peak_exists=true;
		
		l=NCmin(q,0);
		m=NCmin(q,1);
		
		Nfit=NCp(l)+NCn(m)+1+DNwn;
		
		X.zeros(2*Nfit,2*(NCp(l)+NCn(m)+1));
		
		for (j=NCp(l); j>=-NCn(m); j--)
		{
			for (p=p0(q); p<=Nfit+p0(q)-1; p++)
			{
				X(2*(p-p0(q)),2*NCp(l)-2*j+1)=pow(-1,j)*pow(wn(p-1),2*j);
				X(2*(p-p0(q))+1,2*NCp(l)-2*j)=pow(-1,j)*pow(wn(p-1),2*j+1);
			}
		}
		
		Gtmp=Gchi2.rows(2*p0(q)-2,2*(Nfit+p0(q))-3);
		CG=COV.submat(2*p0(q)-2,2*p0(q)-2,2*(Nfit+p0(q))-3,2*(Nfit+p0(q))-3);
		
		invCG=inv(CG);
		//	invCG=inv_sympd(CG);
		AM=(X.t())*invCG*X;
		AM=0.5*(AM.t()+AM);
		BM=(X.t())*invCG*Gtmp;
		
		Mtmp=solve(AM,BM);
		
		//			NA=AM.n_rows;
		//			dposv_(&UPLO, &NA, &NRHS, AM.memptr(), &NA, BM.memptr(), &NA, &INFO);
		//			Mtmp=BM;
		
		//			rowvec maxX=max(abs(X),0);
		//			mat P=diagmat(1.0/maxX);
		//			AMP=(P*(AM*P)+(P*AM)*P)/2.0;
		//			BMP=P*BM;
		//			dposv_(&UPLO, &NA, &NRHS, AMP.memptr(), &NA, BMP.memptr(), &NA, &INFO);
		//			MP=BMP;
		//			MP=solve(AMP,BMP);
		//			Mtmp=P*MP;
		
		Glftmp=X*Mtmp;
		
		mat COVpeak=inv(AM);
		
		//			invAMP=inv(AMP);
		//			COVpeak=((P*invAMP)*P+P*(invAMP*P))/2;
		
		double err_norm_peak=sqrt(COVpeak(2*NCp(l)+2,2*NCp(l)+2));
		double err_M1_peak=sqrt(COVpeak(2*NCp(l)+3,2*NCp(l)+3));
		double err_M2_peak=sqrt(COVpeak(2*NCp(l)+4,2*NCp(l)+4));
		double err_peak_position=err_M1_peak/peak_weight-err_norm_peak*peak_center/peak_weight;
		double err_std_peak=(-err_norm_peak*m2_lf_min/peak_weight+2*err_norm_peak*peak_center*peak_center/peak_weight+err_M2_peak/peak_weight-2*err_M1_peak*peak_center/peak_weight)/(2*peak_width);
		
		cout<<"Peak detected\n";
		cout<<"peak width: "<<peak_width<<endl;
		//cout<<"error on width: "<<err_std_peak<<endl;
		cout<<"peak weight: "<<peak_weight<<endl;
		//cout<<"error on weight: "<<err_norm_peak<<endl;
		//cout<<"peak position: "<<peak_center<<endl;
		//cout<<"error on position: "<<err_peak_position<<endl;
		
		if (displ_adv_prep_figs)
		{
			graph_2D g1, g2;
			
			vec x=wn.rows(p0(q)-1,Nfit+p0(q)-2), y;
			
			char xl[]="$\\\\omega_n$";
			char yl[]="Gr";
			char yl2[]="Gi";
			char lgd1[]="data";
			char lgd2[]="fit";
			char attr1[]="'o', markeredgecolor='r', markerfacecolor='none'";
			char attr2[]="'s', markeredgecolor='b', markerfacecolor='none'";
			
			double xlims[2], ylims[2], ymin, ymax;
			
			xlims[0]=x.min();
			xlims[1]=x.max();
			
			y=Gr.rows(p0(q)-1,Nfit+p0(q)-2);
			ymax=y.max();
			ymin=y.min();
			ylims[0]=ymin-0.1*(ymax-ymin);
			ylims[1]=ymax+0.1*(ymax-ymin);
			g1.add_data(x.memptr(),y.memptr(),Nfit);
			g1.add_to_legend(lgd1);
			g1.add_attribute(attr1);
			uvec even_ind=linspace<uvec>(0,2*Nfit-2,Nfit);
			y=Glftmp.rows(even_ind);
			g1.add_data(x.memptr(),y.memptr(),Nfit);
			g1.add_to_legend(lgd2);
			g1.add_attribute(attr2);
			g1.set_axes_labels(xl,yl);
			g1.set_axes_lims(xlims,ylims);
			g1.curve_plot();
			
			y=Gi.rows(p0(q)-1,Nfit+p0(q)-2);
			ymax=y.max();
			ymin=y.min();
			ylims[0]=ymin-0.1*(ymax-ymin);
			ylims[1]=ymax+0.1*(ymax-ymin);
			g2.add_data(x.memptr(),y.memptr(),Nfit);
			g2.add_to_legend(lgd1);
			g2.add_attribute(attr1);
			uvec odd_ind=linspace<uvec>(1,2*Nfit-1,Nfit);
			y=Glftmp.rows(odd_ind);
			g2.add_data(x.memptr(),y.memptr(),Nfit);
			g2.add_to_legend(lgd2);
			g2.add_attribute(attr2);
			g2.set_axes_labels(xl,yl2);
			g2.set_axes_lims(xlims,ylims);
			g2.curve_plot();
			
			if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
			graph_2D::show_figures();
		}
		
		dw_peak=peak_width/2.0;
	}
	else
	{
		cout<<"no peak found\n";
	}

	return peak_exists;
}

bool OmegaMaxEnt_data::compute_moments_omega_n_2()
{
	int j, NC=3;
	bool sol_found;
	
	double Ginf;
	
	cout<<"COMPUTING MOMENTS and frequency-independant part of data\n";
	
	vec C0b(Nn-3), C1b(Nn-3), C2b(Nn-3), C3b(Nn-3), C4b(Nn-3);
	double wn1, wn2, wn3, reG1, reG2, reG3, imG1, imG2, denom2, denom;
	for (j=1; j<Nn-2; j++)
	{
		wn1=wn(j);
		wn2=wn(j+1);
		wn3=wn(j+2);
		reG1=Gr(j);
		reG2=Gr(j+1);
		reG3=Gr(j+2);
		imG1=Gi(j);
		imG2=Gi(j+1);
		denom2=wn1*wn1 - wn2*wn2;
		C1b(j-1)=-(imG1*pow(wn1,3) - imG2*pow(wn2,3))/denom2;
		C3b(j-1)=((wn1*wn1)*(wn2*wn2)*(-imG1*wn1 + imG2*wn2))/denom2;
		denom=denom2*(wn1*wn1 - wn3*wn3)*(wn2*wn2-wn3*wn3);
		C0b(j-1)=(reG3*pow(wn3,4)*(wn1*wn1 - wn2*wn2) + reG1*pow(wn1,4)*(wn2*wn2 - wn3*wn3) + reG2*pow(wn2,4)*(wn3*wn3-wn1*wn1))/denom;
		C2b(j-1)=-(reG3*(-pow(wn1,4) + pow(wn2,4))*pow(wn3,4) + reG2*pow(wn2,4)*(pow(wn1,4) - pow(wn3,4)) +  reG1*pow(wn1,4)*(-pow(wn2,4) + pow(wn3,4)))/denom;
		C4b(j-1)=(wn1*wn1*wn2*wn2*wn3*wn3*(reG3*(wn1*wn1 - wn2*wn2)*wn3*wn3 + reG1*wn1*wn1*(wn2*wn2 - wn3*wn3) + reG2*wn2*wn2*(-wn1*wn1 + wn3*wn3)))/denom;
	}
	
	if (displ_adv_prep_figs)
	{
		graph_2D g0, g1, g2;
		
		vec x=wn.rows(1,Nn-3);
		char xl[]="fit frequency $\\\\omega_{n}$";
		char yl0[]="$G_{inf}$";
		char yl[]="$M_0$";
		char yl2[]="$M_1$";
		
		plot(g0, x, C0b, xl, yl0);
		plot(g1, x, C1b, xl, yl);
		plot(g2, x, C2b, xl, yl2);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	

	int p, Nfitmin, jfitmin, jfitmax, NNfit, Nfit;
	jfitmin=2;
	Nfitmin=2*NC+4;
	jfitmax=Nn-Nfitmin;
	NNfit=jfitmax-jfitmin+1;
	
	int jfitmax_fin=jfitmax;
	
	vec Ginfv(NNfit), M0v(NNfit), M1v(NNfit), M2v(NNfit), M3v(NNfit);
	mat invCG, A, X, CG;
	vec Mtmp;
	
	//	char UPLO='U';
	//	int NA=2*NC, NRHS=1, INFO;
	
	if (jfitmax<jfitmin) return false;
	
	//	mat LC, invLC;
	for (jfit=jfitmin; jfit<=jfitmax; jfit++)
	{
		Nfit=Nn_fit_max;
		if ((Nn-jfit+1)<Nn_fit_max)
			Nfit=Nn-jfit+1;
		
		X.zeros(2*Nfit,2*NC+1);
		
		j=0;
		
		for (p=jfit; p<=jfit+Nfit-1; p++)
		{
			X(2*(p-jfit),0)=1;
		}
		
		for (j=1; j<=NC; j++)
		{
			for (p=jfit; p<=jfit+Nfit-1; p++)
			{
				X(2*(p-jfit),2*j)=pow(-1,j)/pow(wn(p-1),2*j);
				X(2*(p-jfit)+1,2*j-1)=pow(-1,j)/pow(wn(p-1),2*j-1);
			}
		}
		
		CG=COV.submat(2*jfit-2,2*jfit-2,2*(jfit+Nfit-1)-1,2*(jfit+Nfit-1)-1);
		
		//		if (jfit==jfitmin) cout<<CG.submat(0,0,10,10);
		
		//		LC=chol(CG);
		//		invLC=inv(LC);
		//		invCG=invLC*invLC.t();
		
		invCG=inv(CG);
		//	invCG=inv_sympd(CG);
		A=trans(X)*invCG*X;
		A=0.5*(A+A.t());
		Mtmp=trans(X)*invCG*Gchi2.rows(2*jfit-2,2*(jfit+Nfit-1)-1);
		
		//		dposv_(&UPLO, &NA, &NRHS, A.memptr(), &NA, Mtmp.memptr(), &NA, &INFO);
		Mtmp=solve(A,Mtmp);
		//		sol_found=solve(Mtmp,A,Mtmp);
		
		//		if (jfitmax_fin==jfitmax && (!sol_found || rcond(A)<EPSILON))
		//		{
		//			jfitmax_fin=jfit;
		//			//break;
		//		}
		
		Ginfv(jfit-jfitmin)=Mtmp(0);
		M0v(jfit-jfitmin)=Mtmp(1);
		M1v(jfit-jfitmin)=Mtmp(2);
		M2v(jfit-jfitmin)=Mtmp(3);
		M3v(jfit-jfitmin)=Mtmp(4);
	}

/*
	if (displ_adv_prep_figs)
	{
		graph_2D g0, g1, g2, g3, g4;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="$\\\\omega_n$";
		char yl0[]="$G_{inf}$";
		char yl[]="$M_0$";
		char yl2[]="$M_1$";
		char yl3[]="$M_2$";
		char yl4[]="$M_3$";
		
		plot(g0, x, Ginfv, xl, yl0);
		plot(g1, x, M0v, xl, yl);
		plot(g2, x, M1v, xl, yl2);
		plot(g3, x, M2v, xl, yl3);
		plot(g4, x, M3v, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
*/
	
	int Nv=Nn/16;
	if (Nv<2) Nv=2;
	
	//	NNfit=jfitmax_fin-jfitmin;
	
	vec varGinf(NNfit-2*Nv), varM0(NNfit-2*Nv), varM1(NNfit-2*Nv), varM2(NNfit-2*Nv), varM3(NNfit-2*Nv);
	
	for (j=Nv; j<NNfit-Nv; j++)
	{
		varGinf(j-Nv)=var(Ginfv.rows(j-Nv,j+Nv));
		varM0(j-Nv)=var(M0v.rows(j-Nv,j+Nv));
		varM1(j-Nv)=var(M1v.rows(j-Nv,j+Nv));
		varM2(j-Nv)=var(M2v.rows(j-Nv,j+Nv));
		varM3(j-Nv)=var(M3v.rows(j-Nv,j+Nv));
	}
	
	if (displ_adv_prep_figs)
	{
		graph_2D g0, g1, g2, g3, g4;
		
		vec x=wn.rows(Nv,NNfit-Nv-1);
		
		char xl[]="$\\\\omega_n$";
		char yl0[]="varGinf";
		char yl[]="varM0";
		char yl2[]="varM1";
		char yl3[]="varM2";
		char yl4[]="varM3";
		
		plot(g0, x, varGinf, xl, yl0);
		plot(g1, x, varM0, xl, yl);
		plot(g2, x, varM1, xl, yl2);
		plot(g3, x, varM2, xl, yl3);
		plot(g4, x, varM3, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	uword j0, j1, j2, j3, j4;
	varM0.min(j0);
	varM1.min(j1);
	varM2.min(j2);
	varM3.min(j3);
	varGinf.min(j4);
	
	j0=j0+Nv;
	j1=j1+Nv;
	j2=j2+Nv;
	j3=j3+Nv;
	j4=j4+Nv;
	
	Mfit.zeros(4);
	Mfit(0)=mean(M0v.rows(j0-Nv,j0+Nv));
	Mfit(1)=mean(M1v.rows(j1-Nv,j1+Nv));
	Mfit(2)=mean(M2v.rows(j2-Nv,j2+Nv));
	Mfit(3)=mean(M3v.rows(j3-Nv,j3+Nv));
	
	Ginf=mean(Ginfv.rows(j3-Nv,j3+Nv));
	
//	cout<<"frequency range used to determine the moments: "<<wn(j0+jfitmin-Nv-1)<<" to "<<wn(j0+jfitmin+Nv-1)<<" (indices "<<j0+jfitmin-Nv-1<<" to "<<j0+jfitmin+Nv-1<<")"<<endl;
	
	int jfit0;
	j=Nv;
	if (!boson)
	{
		while ( j<j0 && (abs(mean(M0v.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(M0v.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) )
		{
			j=j+1;
		}
	}
	else
	{
		while ( j<j0 && (abs(mean(M1v.rows(j-Nv,j+Nv))-Mfit(1))/Mfit(1)>tol_mean_C1 || stddev(M1v.rows(j-Nv,j+Nv))/Mfit(1)>tol_std_C1) )
		{
			j=j+1;
		}
	}
	jfit0=j;
/*
	int jfit0;
	j=Nv;
	if (!boson)
	{
		while ((abs(mean(C1b.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(C1b.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) && j<Nn-Nv-4)
		{
			j=j+1;
		}
	}
	else
	{
		while ((abs(mean(C2b.rows(j-Nv,j+Nv))-Mfit(1))/Mfit(1)>tol_mean_C1 || stddev(C2b.rows(j-Nv,j+Nv))/Mfit(1)>tol_std_C1) && j<Nn-Nv-4)
		{
			j=j+1;
		}
	}
	jfit0=j;
*/
	
	if (!boson)
		jfit=j0+jfitmin-1;
	else
		jfit=j1+jfitmin-1;
	Nfit=Nn_fit_fin;
	if (Nfit>Nn-jfit+1)
		Nfit=Nn-jfit+1;
	
	X.zeros(2*Nfit,2*NC);
	
	for (j=1; j<=NC; j++)
	{
		for (p=jfit; p<=jfit+Nfit-1; p++)
		{
			X(2*(p-jfit),2*j-1)=pow(-1,j)/pow(wn(p-1),2*j);
			X(2*(p-jfit)+1,2*j-2)=pow(-1,j)/pow(wn(p-1),2*j-1);
		}
	}
	
	CG=COV.submat(2*jfit-2,2*jfit-2,2*(jfit+Nfit-1)-1,2*(jfit+Nfit-1)-1);
	
	//	LC=chol(CG);
	//	invLC=inv(LC);
	//	invCG=invLC*invLC.t();
	
	invCG=inv(CG);
	//	invCG=inv_sympd(CG);
	A=trans(X)*invCG*X;
	A=0.5*(A+A.t());
	mat COVMtmp=inv(A);
	COVMfit=COVMtmp.submat(0,0,3,3);
	
//	jfit=jfit0;
	jfit=jfit0+jfitmin-1;
	
	cout<<"frequency range of asymptotic behavior: "<<wn(jfit-1)<<" to "<<wn(Nn-1)<<" (indices "<<jfit-1<<" to "<<Nn-1<<")"<<endl;
	
	cout<<"frequency-independant part of G: "<<Ginf<<endl;
	cout<<"norm extracted from high frequencies of G: "<<Mfit(0)<<endl;
	cout<<"1st moment extracted from high frequencies: "<<Mfit(1)<<endl;
	cout<<"2nd moment extracted from high frequencies: "<<Mfit(2)<<endl;
	cout<<"3rd moment extracted from high frequencies: "<<Mfit(3)<<endl;
	
	if (displ_prep_figs)
	{
		graph_2D g0, g1, g2; //, g3, g4;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="$\\\\omega_n$";
		char yl0[]="$G_{inf}$";
		char yl[]="$M_0$";
		char yl2[]="$M_1$";
		//	char yl3[]="$M_2$";
		//	char yl4[]="$M_3$";
		
		plot(g0, x, Ginfv, xl, yl0);
		plot(g1, x, M0v, xl, yl);
		plot(g2, x, M1v, xl, yl2);
		//	plot(g3, x, M2v, xl, yl3);
		//	plot(g4, x, M3v, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	if (displ_prep_figs)
	{
		graph_2D g3, g4;
		//		graph_2D g0, g1, g2, g3, g4;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="$\\\\omega_n$";
		//	char yl0[]="$G_{inf}$";
		//	char yl[]="$M_0$";
		//	char yl2[]="$M_1$";
		char yl3[]="$M_2$";
		char yl4[]="$M_3$";
		
		//	plot(g0, x, Ginfv, xl, yl0);
		//	plot(g1, x, M0v, xl, yl);
		//	plot(g2, x, M1v, xl, yl2);
		plot(g3, x, M2v, xl, yl3);
		plot(g4, x, M3v, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	if (G_omega_inf_in.size())
	{
		if (G_omega_inf)
		{
			if (abs(Ginf/G_omega_inf)>tol_G_inf)
			{
				cout<<"warning: residual frequency-independant part of G might be too large\n";
			}
		}
	}
	else
	{
		G_omega_inf=Ginf;
		Gr=Gr-Ginf;
		G.set_real(Gr);
		if (col_Gi>0)
		{
			uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
			Gchi2.rows(even_ind)=Gr;
		}
		else
			Gchi2=Gr;
	}
	
	double std_omega_tmp;
	double var_omega;
	if (!boson)
	{
		if (M0)
		{
			if (abs(Mfit(0)-M0)/Mfit(0)>tol_norm)
			{
				if (M0_in.size())
					cout<<"warning: norm of spectral function is different from provided one.\n";
				else
				{
					cout<<"warning: spectral function is not normalized.\n";
				}
			}
		}
		
		var_omega=Mfit(2)/Mfit(0)-pow(Mfit(1)/Mfit(0),2);
		
		if (var_omega>0)
			std_omega_tmp=sqrt(var_omega);
		else
		{
			cout<<"Negative variance found during computation of moments.\n";
			return false;
		}
		
		if (!moments_provided)
		{
			M=Mfit;
			NM=M.n_rows;
			COVM.zeros(NM,NM);
			COVM(0,0)=pow(errM0,2);
			COVM.submat(1,1,NM-1,NM-1)=COVMfit.submat(1,1,NM-1,NM-1);
			//		COVM=COVMfit;
			if (!M0_in.size())
			{
				if (abs(Mfit(0)-M0)/Mfit(0)>tol_norm)
					M0=Mfit(0);
			}
			M(0)=M0;
			M1=M(1);
			M2=M(2);
			M3=M(3);
			covm_diag=false;
			M1_set=true;
			M2_set=true;
		}
		else
		{
			if (abs(M1-Mfit(1))/std_omega_tmp>tol_M1)
				cout<<"warning: first moment different from provided one\n";
			if (M2_in.size())
			{
				if (abs(M2-Mfit(2))/Mfit(2)>tol_M2)
					cout<<"warning: second moment different from provided one\n";
				
				if (M3_in.size())
				{
					if (abs(M3-Mfit(3))/pow(std_omega_tmp,3)>tol_M3)
					{
						cout<<"warning: third moment different from provided one\n";
					}
				}
				else
				{
					M=Mfit;
					M(0)=M0;
					M(1)=M1;
					M(2)=M2;
					M3=M(3);
					COVMtmp=COVM.submat(0,0,2,2);
					COVM.zeros(4,4);
					COVM.submat(0,0,2,2)=COVMtmp;
					COVM(3,3)=COVMfit(3,3);
				}
			}
			else if (M3_in.size())
			{
				M=Mfit;
				M(0)=M0;
				M(1)=M1;
				M2=M(2);
				M2_set=true;
				M(3)=M3;
				COVMtmp=COVM.submat(0,0,1,1);
				COVM.zeros(4,4);
				COVM.submat(0,0,1,1)=COVMtmp;
				if (errM3_in.size())
				{
					COVM(2,2)=COVMfit(2,2);
					COVM(3,3)=pow(errM3,2);
				}
				else
				{
					COVM.submat(2,2,3,3)=COVMfit.submat(2,2,3,3);
					covm_diag=false;
				}
			}
			else
			{
				M=Mfit;
				M(0)=M0;
				M(1)=M1;
				M2=M(2);
				M2_set=true;
				M3=M(3);
				COVMtmp=COVM.submat(0,0,1,1);
				COVM.zeros(4,4);
				COVM.submat(0,0,1,1)=COVMtmp;
				COVM.submat(2,2,3,3)=COVMfit.submat(2,2,3,3);
				covm_diag=false;
			}
		}
		
		NM=M.n_rows;
		M_ord=linspace<vec>(0,NM-1,NM);
		if (errM.n_rows<NM)
		{
			vec errM_tmp=errM;
			errM.zeros(NM);
			errM.rows(0,errM_tmp.n_rows-1)=errM_tmp;
			COVMtmp=COVM.submat(errM_tmp.n_rows,errM_tmp.n_rows,NM-1,NM-1);
			errM.rows(errM_tmp.n_rows,NM-1)=sqrt(COVMtmp.diag());
		}
		
		if (!std_omega)
		{
			var_omega=M2/M0-pow(M1/M0,2);
			std_omega=sqrt(var_omega);
		}
		if (!SC_set)
		{
			SC=M1/M0;
			SC_set=true;
		}
		if (!SW_set)
		{
			SW=f_SW_std_omega*std_omega;
			SW_set=true;
		}
		if (M(2)<0)
		{
			M=M.rows(0,1);
			COVM=COVM.submat(0,0,1,1);
			NM=2;
		}
	}
	else
	{
		if (M0_in.size())
		{
			if (M0)
			{
				if (abs((Mfit(0)-M0)/Mfit(0))>tol_norm)
					cout<<"warning: norm of spectral function is different from provided one.\n";
			}
		}
		
		var_omega=Mfit(1)/M1n-pow(Mfit(0)/M1n,2);
		if (var_omega>0)
			std_omega_tmp=sqrt(var_omega);
		else
		{
			cout<<"Negative variance found during computation of moments.\n";
			return false;
		}
		
		if (!moments_provided)
		{
			M=Mfit.rows(0,2);
			NM=3;
			COVM=COVMfit.submat(0,0,NM-1,NM-1);
			if (M0_in.size())
			{
				M(0)=M0;
				COVM.row(0)=zeros<rowvec>(NM);
				COVM.col(0)=zeros<vec>(NM);
				COVM(0,0)=pow(errM0,2);
			}
			else if (M0)
			{
				if (abs((Mfit(0)-M0)/Mfit(0))<tol_norm)
				{
					M(0)=M0;
					COVM.row(0)=zeros<rowvec>(NM);
					COVM.col(0)=zeros<vec>(NM);
					COVM(0,0)=pow(errM0,2);
				}
				else
					M0=M(0);
			}
			else
			{
				M(0)=M0;
				COVM.row(0)=zeros<rowvec>(NM);
				COVM.col(0)=zeros<vec>(NM);
				COVM(0,0)=pow(errM0,2);
			}
			M1=M(1);
			M2=M(2);
			covm_diag=false;
			M1_set=true;
			M2_set=true;
		}
		else
		{
			if (abs((M1-Mfit(1))/Mfit(1))>tol_M1)
				cout<<"warning: first moment different from provided one\n";
			if (M2_in.size())
			{
				if (M2)
				{
					if (abs((M2-Mfit(2))/Mfit(2))>tol_M2)
						cout<<"warning: second moment different from provided one\n";
				}
			}
			else
			{
				M=Mfit.rows(0,2);
				NM=3;
				M(0)=M0;
				M(1)=M1;
				M2=M(2);
				M2_set=true;
				vec errMtmp=errM;
				errM.zeros(NM);
				errM.rows(0,1)=errMtmp;
				errM(2)=sqrt(COVMfit(2,2));
				COVM=diagmat(square(errM));
			}
		}
		M_ord=linspace<vec>(0,NM-1,NM);
		
		if (!std_omega)
		{
			var_omega=M1/M1n-pow(M0/M1n,2);
			if (var_omega>0)
				std_omega=sqrt(var_omega);
			else
			{
				cout<<"Negative variance found during computation of moments.\n";
				return false;
			}
		}
		if (!SC_set)
		{
			SC=M0/M1n;
			SC_set=true;
		}
		if (!SW_set)
		{
			SW=f_SW_std_omega*std_omega;
			SW_set=true;
		}
	}
	
	if (Nn-jfit<Nn_as_min)
	{
		jfit=0;
		if (!moments_provided && maxM>0) maxM=0;
	}
	
/*
	if (displ_prep_figs)
	{
		graph_2D g0, g1, g2; //, g3, g4;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="$\\\\omega_n$";
		char yl0[]="$G_{inf}$";
		char yl[]="$M_0$";
		char yl2[]="$M_1$";
	//	char yl3[]="$M_2$";
	//	char yl4[]="$M_3$";
		
		plot(g0, x, Ginfv, xl, yl0);
		plot(g1, x, M0v, xl, yl);
		plot(g2, x, M1v, xl, yl2);
	//	plot(g3, x, M2v, xl, yl3);
	//	plot(g4, x, M3v, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	if (displ_adv_prep_figs)
	{
		graph_2D g3, g4;
//		graph_2D g0, g1, g2, g3, g4;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="$\\\\omega_n$";
	//	char yl0[]="$G_{inf}$";
	//	char yl[]="$M_0$";
	//	char yl2[]="$M_1$";
		char yl3[]="$M_2$";
		char yl4[]="$M_3$";
		
	//	plot(g0, x, Ginfv, xl, yl0);
	//	plot(g1, x, M0v, xl, yl);
	//	plot(g2, x, M1v, xl, yl2);
		plot(g3, x, M2v, xl, yl3);
		plot(g4, x, M3v, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	*/
	return true;
}

bool OmegaMaxEnt_data::compute_moments_omega_n()
{
	int j, NC=3;
	bool sol_found;
	
	cout<<"COMPUTING MOMENTS\n";
	
	vec C1b(Nn-2), C2b(Nn-2), C3b(Nn-2), C4b(Nn-2);
	double wn1, wn2, reG1, reG2, imG1, imG2, denom2;
	for (j=1; j<Nn-1; j++)
	{
		wn1=wn(j);
		wn2=wn(j+1);
		reG1=Gr(j);
		reG2=Gr(j+1);
		imG1=Gi(j);
		imG2=Gi(j+1);
		denom2=wn1*wn1 - wn2*wn2;
		C1b(j-1)=-(imG1*pow(wn1,3) - imG2*pow(wn2,3))/denom2;
		C2b(j-1)=-(reG1*pow(wn1,4) - reG2*pow(wn2,4))/denom2;
		C3b(j-1)=((wn1*wn1)*(wn2*wn2)*(-imG1*wn1 + imG2*wn2))/denom2;
		C4b(j-1)=(-reG1*pow(wn1,4)*pow(wn2,2) + reG2*pow(wn1,2)*pow(wn2,4))/denom2;
	}

	if (displ_adv_prep_figs)
	{
		graph_2D g1, g2;
		
		vec x=wn.rows(1,Nn-2);
		char xl[]="fit frequency $\\\\omega_{n}$";
		char yl[]="$M_0$";
		char yl2[]="$M_1$";
		
		plot(g1, x, C1b, xl, yl);
		plot(g2, x, C2b, xl, yl2);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}

	int p, Nfitmin, jfitmin, jfitmax, NNfit, Nfit;
	jfitmin=2;
	Nfitmin=2*NC+4;
	jfitmax=Nn-Nfitmin;
	NNfit=jfitmax-jfitmin+1;
	
	int jfitmax_fin=jfitmax;
	
	vec M0v(NNfit), M1v(NNfit), M2v(NNfit), M3v(NNfit);
	mat invCG, A, X, CG;
	vec Mtmp;
	
//	char UPLO='U';
//	int NA=2*NC, NRHS=1, INFO;
	
	if (jfitmax<jfitmin) return false;
	
//	mat LC, invLC;
	for (jfit=jfitmin; jfit<=jfitmax; jfit++)
	{
		Nfit=Nn_fit_max;
		if ((Nn-jfit+1)<Nn_fit_max)
			Nfit=Nn-jfit+1;
		
		X.zeros(2*Nfit,2*NC);
		
		for (j=1; j<=NC; j++)
		{
			for (p=jfit; p<=jfit+Nfit-1; p++)
			{
				X(2*(p-jfit),2*j-1)=pow(-1,j)/pow(wn(p-1),2*j);
				X(2*(p-jfit)+1,2*j-2)=pow(-1,j)/pow(wn(p-1),2*j-1);
			}
		}
		
		CG=COV.submat(2*jfit-2,2*jfit-2,2*(jfit+Nfit-1)-1,2*(jfit+Nfit-1)-1);

//		if (jfit==jfitmin) cout<<CG.submat(0,0,10,10);
	
//		LC=chol(CG);
//		invLC=inv(LC);
//		invCG=invLC*invLC.t();
		
		invCG=inv(CG);
		//	invCG=inv_sympd(CG);
		A=trans(X)*invCG*X;
		A=0.5*(A+A.t());
		Mtmp=trans(X)*invCG*Gchi2.rows(2*jfit-2,2*(jfit+Nfit-1)-1);
		
		//		dposv_(&UPLO, &NA, &NRHS, A.memptr(), &NA, Mtmp.memptr(), &NA, &INFO);
		Mtmp=solve(A,Mtmp);
//		sol_found=solve(Mtmp,A,Mtmp);
		
//		if (jfitmax_fin==jfitmax && (!sol_found || rcond(A)<EPSILON))
//		{
//			jfitmax_fin=jfit;
//			//break;
//		}
		
		M0v(jfit-jfitmin)=Mtmp(0);
		M1v(jfit-jfitmin)=Mtmp(1);
		M2v(jfit-jfitmin)=Mtmp(2);
		M3v(jfit-jfitmin)=Mtmp(3);
	}
	
	
/*
	if (displ_adv_prep_figs)
	{
		graph_2D g1, g2, g3, g4;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="fit frequency $\\\\omega_n$";
		char yl[]="$M_0$";
		char yl2[]="$M_1$";
		char yl3[]="$M_2$";
		char yl4[]="$M_3$";
		
		plot(g1, x, M0v, xl, yl);
		plot(g2, x, M1v, xl, yl2);
		plot(g3, x, M2v, xl, yl3);
		plot(g4, x, M3v, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
*/
	int Nv=Nn/16;
	if (Nv<2) Nv=2;
	
//	NNfit=jfitmax_fin-jfitmin;
	
	vec varM0(NNfit-2*Nv), varM1(NNfit-2*Nv), varM2(NNfit-2*Nv), varM3(NNfit-2*Nv);
	
	for (j=Nv; j<NNfit-Nv; j++)
	{
		varM0(j-Nv)=var(M0v.rows(j-Nv,j+Nv));
		varM1(j-Nv)=var(M1v.rows(j-Nv,j+Nv));
		varM2(j-Nv)=var(M2v.rows(j-Nv,j+Nv));
		varM3(j-Nv)=var(M3v.rows(j-Nv,j+Nv));
	}
	
	uword j0, j1, j2, j3;
	varM0.min(j0);
	varM1.min(j1);
	varM2.min(j2);
	varM3.min(j3);
	
	j0=j0+Nv;
	j1=j1+Nv;
	j2=j2+Nv;
	j3=j3+Nv;
	
	Mfit.zeros(4);
	Mfit(0)=mean(M0v.rows(j0-Nv,j0+Nv));
	Mfit(1)=mean(M1v.rows(j1-Nv,j1+Nv));
	Mfit(2)=mean(M2v.rows(j2-Nv,j2+Nv));
	Mfit(3)=mean(M3v.rows(j3-Nv,j3+Nv));
	
//	cout<<"frequency range used to determine the moments: "<<wn(j0+jfitmin-Nv-1)<<" to "<<wn(j0+jfitmin+Nv-1)<<" (indices "<<j0+jfitmin-Nv-1<<" to "<<j0+jfitmin+Nv-1<<")"<<endl;
	
	int jfit0;
	j=Nv;
	if (!boson)
	{
		while ( j<j0 && (abs(mean(M0v.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(M0v.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) )
		{
			j=j+1;
		}
	}
	else
	{
		while ( j<j0 && (abs(mean(M1v.rows(j-Nv,j+Nv))-Mfit(1))/Mfit(1)>tol_mean_C1 || stddev(M1v.rows(j-Nv,j+Nv))/Mfit(1)>tol_std_C1) )
		{
			j=j+1;
		}
	}
	jfit0=j;
	
//	cout<<"jfit0: "<<jfit0<<endl;
	
/*
	int jfit0;
	j=Nv;
	if (!boson)
	{
		while ((abs(mean(C1b.rows(j-Nv,j+Nv))-Mfit(0))/Mfit(0)>tol_mean_C1 || stddev(C1b.rows(j-Nv,j+Nv))/Mfit(0)>tol_std_C1) && j<Nn-Nv-3)
		{
			j=j+1;
		}
	}
	else
	{
		while ((abs(mean(C2b.rows(j-Nv,j+Nv))-Mfit(1))/Mfit(1)>tol_mean_C1 || stddev(C2b.rows(j-Nv,j+Nv))/Mfit(1)>tol_std_C1) && j<Nn-Nv-3)
		{
			j=j+1;
		}
	}
	jfit0=j;
*/
	
	if (!boson)
		jfit=j0+jfitmin-1;
	else
		jfit=j1+jfitmin-1;
	Nfit=Nn_fit_fin;
	if (Nfit>Nn-jfit+1)
		Nfit=Nn-jfit+1;
	
	X.zeros(2*Nfit,2*NC);
	
	for (j=1; j<=NC; j++)
	{
		for (p=jfit; p<=jfit+Nfit-1; p++)
		{
			X(2*(p-jfit),2*j-1)=pow(-1,j)/pow(wn(p-1),2*j);
			X(2*(p-jfit)+1,2*j-2)=pow(-1,j)/pow(wn(p-1),2*j-1);
		}
	}
	
	CG=COV.submat(2*jfit-2,2*jfit-2,2*(jfit+Nfit-1)-1,2*(jfit+Nfit-1)-1);

//	LC=chol(CG);
//	invLC=inv(LC);
//	invCG=invLC*invLC.t();
	
	invCG=inv(CG);
	//	invCG=inv_sympd(CG);
	A=trans(X)*invCG*X;
	A=0.5*(A+A.t());
	mat COVMtmp=inv(A);
	COVMfit=COVMtmp.submat(0,0,3,3);
	
	/*
	 int jfitmin0=jfitmin;
	 jfitmin=j0+jfitmin0-1-Nv;
	 jfitmax=j0+jfitmin0-1+Nv;
	 NNfit=jfitmax-jfitmin+1;
	 M0v.zeros(NNfit);
	 M1v.zeros(NNfit);
	 M2v.zeros(NNfit);
	 M3v.zeros(NNfit);
	 
	 for (jfit=jfitmin; jfit<=jfitmax; jfit++)
	 {
		Nfit=Nn_fit_fin;
		if ((Nn-jfit+1)<Nn_fit_fin)
	 Nfit=Nn-jfit+1;
	 
		X.zeros(2*Nfit,2*NC);
		
		for (j=1; j<=NC; j++)
		{
	 for (p=jfit; p<=jfit+Nfit-1; p++)
	 {
	 X(2*(p-jfit),2*j-1)=pow(-1,j)/pow(wn(p-1),2*j);
	 X(2*(p-jfit)+1,2*j-2)=pow(-1,j)/pow(wn(p-1),2*j-1);
	 }
		}
		
		CG=COV.submat(2*jfit-2,2*jfit-2,2*(jfit+Nfit-1)-1,2*(jfit+Nfit-1)-1);
		
		invCG=inv(CG);
		A=trans(X)*invCG*X;
		A=0.5*(A+A.t());
		Mtmp=trans(X)*invCG*Gchi2.rows(2*jfit-2,2*(jfit+Nfit-1)-1);
		
		dposv_(&UPLO, &NA, &NRHS, A.memptr(), &NA, Mtmp.memptr(), &NA, &INFO);
		
		M0v(jfit-jfitmin)=Mtmp(0);
		M1v(jfit-jfitmin)=Mtmp(1);
		M2v(jfit-jfitmin)=Mtmp(2);
		M3v(jfit-jfitmin)=Mtmp(3);
		
	 }
	 
	 Mfit(0)=mean(M0v);
	 Mfit(1)=mean(M1v);
	 Mfit(2)=mean(M2v);
	 Mfit(3)=mean(M3v);
	 
	 cout<<"Mfit final:\n"<<Mfit(0)<<endl<<Mfit(1)<<endl<<Mfit(2)<<endl<<Mfit(3)<<endl;
	 */
	
//	jfit=jfit0;
	jfit=jfit0+jfitmin-1;
	
	cout<<"frequency range of asymptotic behavior: "<<wn(jfit-1)<<" to "<<wn(Nn-1)<<" (indices "<<jfit-1<<" to "<<Nn-1<<")"<<endl;
	
	cout<<"norm extracted from high frequencies of G: "<<Mfit(0)<<endl;
	cout<<"1st moment extracted from high frequencies: "<<Mfit(1)<<endl;
	cout<<"2nd moment extracted from high frequencies: "<<Mfit(2)<<endl;
	cout<<"3rd moment extracted from high frequencies: "<<Mfit(3)<<endl;
	
	double std_omega_tmp;
	double var_omega;
	if (!boson)
	{
		if (M0)
		{
			if (abs(Mfit(0)-M0)/Mfit(0)>tol_norm)
			{
				if (M0_in.size())
					cout<<"warning: norm of spectral function is different from provided one.\n";
				else
				{
					cout<<"warning: spectral function is not normalized.\n";
				}
			}
		}
		
		var_omega=Mfit(2)/Mfit(0)-pow(Mfit(1)/Mfit(0),2);
		
		if (var_omega>0)
			std_omega_tmp=sqrt(var_omega);
		else
		{
			cout<<"Negative variance found during computation of moments.\n";
			return false;
		}
		
		if (!moments_provided)
		{
			M=Mfit;
			NM=M.n_rows;
			COVM.zeros(NM,NM);
			COVM(0,0)=pow(errM0,2);
			COVM.submat(1,1,NM-1,NM-1)=COVMfit.submat(1,1,NM-1,NM-1);
			//		COVM=COVMfit;
			if (!M0_in.size())
			{
				if (abs(Mfit(0)-M0)/Mfit(0)>tol_norm)
					M0=Mfit(0);
			}
			M(0)=M0;
			M1=M(1);
			M2=M(2);
			M3=M(3);
			covm_diag=false;
			M1_set=true;
			M2_set=true;
		}
		else
		{
			if (abs(M1-Mfit(1))/std_omega_tmp>tol_M1)
				cout<<"warning: first moment different from provided one\n";
			if (M2_in.size())
			{
				if (abs(M2-Mfit(2))/Mfit(2)>tol_M2)
					cout<<"warning: second moment different from provided one\n";
				
				if (M3_in.size())
				{
					if (abs(M3-Mfit(3))/pow(std_omega_tmp,3)>tol_M3)
					{
						cout<<"warning: third moment different from provided one\n";
					}
				}
				else
				{
					M=Mfit;
					M(0)=M0;
					M(1)=M1;
					M(2)=M2;
					M3=M(3);
					COVMtmp=COVM.submat(0,0,2,2);
					COVM.zeros(4,4);
					COVM.submat(0,0,2,2)=COVMtmp;
					COVM(3,3)=COVMfit(3,3);
				}
			}
			else if (M3_in.size())
			{
				M=Mfit;
				M(0)=M0;
				M(1)=M1;
				M2=M(2);
				M2_set=true;
				M(3)=M3;
				COVMtmp=COVM.submat(0,0,1,1);
				COVM.zeros(4,4);
				COVM.submat(0,0,1,1)=COVMtmp;
				if (errM3_in.size())
				{
					COVM(2,2)=COVMfit(2,2);
					COVM(3,3)=pow(errM3,2);
				}
				else
				{
					COVM.submat(2,2,3,3)=COVMfit.submat(2,2,3,3);
					covm_diag=false;
				}
			}
			else
			{
				M=Mfit;
				M(0)=M0;
				M(1)=M1;
				M2=M(2);
				M2_set=true;
				M3=M(3);
				COVMtmp=COVM.submat(0,0,1,1);
				COVM.zeros(4,4);
				COVM.submat(0,0,1,1)=COVMtmp;
				COVM.submat(2,2,3,3)=COVMfit.submat(2,2,3,3);
				covm_diag=false;
			}
		}
		
		NM=M.n_rows;
		M_ord=linspace<vec>(0,NM-1,NM);
		if (errM.n_rows<NM)
		{
			vec errM_tmp=errM;
			errM.zeros(NM);
			errM.rows(0,errM_tmp.n_rows-1)=errM_tmp;
			COVMtmp=COVM.submat(errM_tmp.n_rows,errM_tmp.n_rows,NM-1,NM-1);
			errM.rows(errM_tmp.n_rows,NM-1)=sqrt(COVMtmp.diag());
		}
		
		if (!std_omega)
		{
			var_omega=M2/M0-pow(M1/M0,2);
			std_omega=sqrt(var_omega);
		}
		if (!SC_set)
		{
			SC=M1/M0;
			SC_set=true;
		}
		if (!SW_set)
		{
			SW=f_SW_std_omega*std_omega;
			SW_set=true;
		}
		if (M(2)<0)
		{
			M=M.rows(0,1);
			COVM=COVM.submat(0,0,1,1);
			NM=2;
		}
	}
	else
	{
		if (M0_in.size())
		{
			if (M0)
			{
				if (abs((Mfit(0)-M0)/Mfit(0))>tol_norm)
					cout<<"warning: norm of spectral function is different from provided one.\n";
			}
		}
		
		var_omega=Mfit(1)/M1n-pow(Mfit(0)/M1n,2);
		if (var_omega>0)
			std_omega_tmp=sqrt(var_omega);
		else
		{
			cout<<"Negative variance found during computation of moments.\n";
			return false;
		}
		
		if (!moments_provided)
		{
			M=Mfit.rows(0,2);
			NM=3;
			COVM=COVMfit.submat(0,0,NM-1,NM-1);
			if (M0_in.size())
			{
				M(0)=M0;
				COVM.row(0)=zeros<rowvec>(NM);
				COVM.col(0)=zeros<vec>(NM);
				COVM(0,0)=pow(errM0,2);
			}
			else if (M0)
			{
				if (abs((Mfit(0)-M0)/Mfit(0))<tol_norm)
				{
					M(0)=M0;
					COVM.row(0)=zeros<rowvec>(NM);
					COVM.col(0)=zeros<vec>(NM);
					COVM(0,0)=pow(errM0,2);
				}
				else
					M0=M(0);
			}
			else
			{
				M(0)=M0;
				COVM.row(0)=zeros<rowvec>(NM);
				COVM.col(0)=zeros<vec>(NM);
				COVM(0,0)=pow(errM0,2);
			}
			M1=M(1);
			M2=M(2);
			covm_diag=false;
			M1_set=true;
			M2_set=true;
		}
		else
		{
			if (abs((M1-Mfit(1))/Mfit(1))>tol_M1)
				cout<<"warning: first moment different from provided one\n";
			if (M2_in.size())
			{
				if (M2)
				{
					if (abs((M2-Mfit(2))/Mfit(2))>tol_M2)
						cout<<"warning: second moment different from provided one\n";
				}
			}
			else
			{
				M=Mfit.rows(0,2);
				NM=3;
				M(0)=M0;
				M(1)=M1;
				M2=M(2);
				M2_set=true;
				vec errMtmp=errM;
				errM.zeros(NM);
				errM.rows(0,1)=errMtmp;
				errM(2)=sqrt(COVMfit(2,2));
				COVM=diagmat(square(errM));
			}
		}
		M_ord=linspace<vec>(0,NM-1,NM);
		
		if (!std_omega)
		{
			var_omega=M1/M1n-pow(M0/M1n,2);
			if (var_omega>0)
				std_omega=sqrt(var_omega);
			else
			{
				cout<<"Negative variance found during computation of moments.\n";
				return false;
			}
		}
		if (!SC_set)
		{
			SC=M0/M1n;
			SC_set=true;
		}
		if (!SW_set)
		{
			SW=f_SW_std_omega*std_omega;
			SW_set=true;
		}
	}
	
	if (Nn-jfit<Nn_as_min)
	{
		jfit=0;
		if (!moments_provided && maxM>0) maxM=0;
	}

	if (displ_prep_figs)
	{
		graph_2D g1, g2;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="fit frequency $\\\\omega_n$";
		char yl[]="$M_0$";
		char yl2[]="$M_1$";
		
		plot(g1, x, M0v, xl, yl);
		plot(g2, x, M1v, xl, yl2);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
//	if (displ_adv_prep_figs)
	if (displ_prep_figs)
	{
		//	graph_2D g1, g2, g3, g4;
		graph_2D g3, g4;
		
		vec x=wn.rows(jfitmin-1,jfitmax-1);
		
		char xl[]="fit frequency $\\\\omega_n$";
		//	char yl[]="$M_0$";
		//	char yl2[]="$M_1$";
		char yl3[]="$M_2$";
		char yl4[]="$M_3$";
		
		//	plot(g1, x, M0v, xl, yl);
		//	plot(g2, x, M1v, xl, yl2);
		plot(g3, x, M2v, xl, yl3);
		plot(g4, x, M3v, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	if (displ_adv_prep_figs)
	{
		graph_2D g1, g2, g3, g4;
		
		vec x=wn.rows(Nv,NNfit-Nv-1);
		
		char xl[]="$\\\\omega_n$";
		char yl[]="varM0";
		char yl2[]="varM1";
		char yl3[]="varM2";
		char yl4[]="varM3";
		
		plot(g1, x, varM0, xl, yl);
		plot(g2, x, varM1, xl, yl2);
		plot(g3, x, varM2, xl, yl3);
		plot(g4, x, varM3, xl, yl4);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	return true;
}

bool OmegaMaxEnt_data::set_moments_fermions()
{
	NM_odd=0;
	
	moments_provided=false;
	covm_diag=true;
	
	M0=1.0;
	if (tau_GF)
		M0=M0t;
	if (M0_in.size())
		M0=stod(M0_in);
	errM0=err_norm*M0;
	
	M.zeros(1);
	M(0)=M0;
	errM.zeros(1);
	errM(0)=errM0;
	NM=1;
	M_ord=linspace<vec>(0,NM-1,NM);
	
	M_even.zeros(1);
	M_even(0)=M0;
	NM_even=1;
	
	std_omega=0;
	
	if (M1_in.size())
	{
		M1=stod(M1_in);
		M1_set=true;
//		NMinput=2;
		moments_provided=true;
		cout<<"moments provided\n";
		M.zeros(2);
		M(0)=M0;
		M(1)=M1;
		NM=2;
		M_ord=linspace<vec>(0,NM-1,NM);
		
		M_odd.zeros(1);
		M_odd(0)=M1;
		NM_odd=1;
		if (!SC_set)
		{
			SC=M1/M0;
			SC_set=true;
		}
		if (M2_in.size())
		{
			M2=stod(M2_in);
			M2_set=true;
			if (M2<=0)
			{
				cout<<"Error: second moment of spectral function must be greater than 0.\n";
				return false;
			}
//			NMinput=3;
			M.zeros(3);
			M(0)=M0;
			M(1)=M1;
			M(2)=M2;
			NM=3;
			M_ord=linspace<vec>(0,NM-1,NM);
			double var_omega=M2/M0-pow(M1/M0,2);
			if (var_omega>0)
			{
				std_omega=sqrt(var_omega);
				if (!SW_set)
				{
					SW=f_SW_std_omega*std_omega;
					SW_set=true;
				}
			}
			else
			{
				cout<<"Error: negative variance. First and/or second moment incorect.\n";
				return false;
			}
			errM2=default_error_M*M2;
			if (errM2_in.size())
				errM2=stod(errM2_in);
		}
		if (M3_in.size())
		{
			M3=stod(M3_in);
			if (M2_in.size())
			{
				M.zeros(4);
				M(0)=M0;
				M(1)=M1;
				M(2)=M2;
				M(3)=M3;
				NM=4;
				M_ord=linspace<vec>(0,NM-1,NM);
			}
			M_odd.zeros(2);
			M_odd(0)=M1;
			M_odd(1)=M3;
			NM_odd=2;
			if (errM3_in.size())
				errM3=stod(errM3_in);
			else if (M2_in.size())
				errM3=default_error_M*pow(std_omega,3);
			else if (SW_in.size())
				errM3=default_error_M*pow(SW,3);
			else
				errM3=default_error_M;
		}
		NM=M.n_rows;
		if (errM1_in.size())
			errM1=stod(errM1_in);
		else if (M2_in.size())
			errM1=default_error_M*std_omega;
		else if (SW_in.size())
			errM1=default_error_M*SW;
		else
			errM1=default_error_M;
		errM.zeros(NM);
		errM(0)=errM0;
		errM(1)=errM1;
		if (M2_in.size())
		{
			errM(2)=errM2;
			if (M3_in.size())
				errM(3)=errM3;
		}
	}
	COVM.zeros(NM,NM);
	COVM.diag()=square(errM);
	
	if ( NM<3 && !SW_in.size() && (!use_grid_params || omega_grid_params.n_cols<3) && !grid_omega_file.size() && !init_spectr_func_file.size() )
	{
		if (!tau_GF)
			cout<<"not enough information provided to define the real frequency grid. The program will try to extract moments from the high frequencies of the Green function\n";
		else
			cout<<"not enough information provided to define the real frequency grid. The program will try to extract moments from G(tau) around tau=0 and tau=beta.\n";
		eval_moments=true;
	}
	
	if (M2_in.size())
	{
		if (!M1_in.size())
		{
			M2=stod(M2_in);
			M2_set=true;
			if (M2<=0)
			{
				cout<<"Error: second moment of spectral function must be greater than 0.\n";
				return false;
			}
			errM2=default_error_M*M2;
			if (errM2_in.size())
				errM2=stod(errM2_in);
		}
		M_even.zeros(2);
		M_even(0)=M0;
		M_even(1)=M2;
		NM_even=2;
	}
	
	NMinput=NM_even+NM_odd;
	
	return true;
}

bool OmegaMaxEnt_data::set_covar_G_omega_n()
{
	int j;
	error_provided=false;
	
	COV.zeros(2*Nn,2*Nn);
	cov_diag=true;
	if (error_file.size())
	{
		if (error_data.n_rows<Nn)
		{
			cout<<"number of lines is too small in file "<<error_file<<endl;
			return false;
		}
		cout<<"error file provided\n";
		errGr=error_data.col(col_errGr-1);
		errGi=error_data.col(col_errGi-1);
		if (wn_sign_change)
		{
			errGr=errGr.rows(indG_0,indG_f);
			errGi=errGi.rows(indG_0,indG_f);
		}
		if (wn_inverted)
		{
			errGr=flipud(errGr);
			errGi=flipud(errGi);
		}
		errG.zeros(2*Nn);
		for (j=0; j<Nn; j++)
		{
			errG(2*j)=errGr(j);
			errG(2*j+1)=errGi(j);
		}
		COV.diag()=square(errG);
		error_provided=true;
	}
	else if ( covar_re_re_file.size() && covar_im_im_file.size() && covar_re_im_file.size() )
	{
		if (CRR.n_rows<Nn || CRR.n_cols<Nn || CII.n_rows<Nn || CII.n_cols<Nn || CRI.n_rows<Nn || CRI.n_cols<Nn)
		{
			cout<<"number of lines and/or columns in covariance file(s) is too small\n";
			return false;
		}
		cout<<"covariance matrix provided\n";
		cov_diag=false;
		uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
		uvec odd_ind=even_ind+1;
		COV(even_ind,even_ind)=CRR.submat(0,0,Nn-1,Nn-1);
		COV(odd_ind,odd_ind)=CII.submat(0,0,Nn-1,Nn-1);
		COV(even_ind,odd_ind)=CRI.submat(0,0,Nn-1,Nn-1);
		COV(odd_ind,even_ind)=trans(CRI.submat(0,0,Nn-1,Nn-1));
		error_provided=true;
	}
	else if (N_params_noise)
	{
		int k;
		
		cout<<"added noise relative error: "<<noise_params(ind_noise)<<endl;
		
		errGr=noise_params(ind_noise)*abs(Gr);
		errGi=noise_params(ind_noise)*abs(Gi);
		
		if (!Nsmooth_errG)
		{
			int Nsm_tmp=1;
			if (!boson)
			{
				double b=-log(wgt_min_sm)/Nsm_tmp;
				
				vec wgt_sm(2*Nsm_tmp+1);
				for (j=0; j<Nsm_tmp; j++) wgt_sm(j)=exp(-b*(Nsm_tmp-j));
				wgt_sm(Nsm_tmp)=1;
				for (j=Nsm_tmp+1; j<=2*Nsm_tmp; j++) wgt_sm(j)=wgt_sm(2*Nsm_tmp-j);
				wgt_sm=wgt_sm/sum(wgt_sm);
				//	cout<<"wgt_sm:\n";
				//	for (j=0; j<=2*Nsm_tmp; j++) cout<<setw(20)<<wgt_sm(j);
				//	cout<<endl;
				//	cout<<"sum(wgt_sm): "<<sum(wgt_sm)<<endl;
				
				vec errG_tmp=errGr;
				
				double wgt_tmp;
				for (j=0; j<Nsm_tmp; j++)
				{
					errGr(j)=0;
					wgt_tmp=0;
					for (k=Nsm_tmp-j; k<=2*Nsm_tmp; k++)
					{
						errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsm_tmp);
						wgt_tmp+=wgt_sm(k);
					}
					errGr(j)=errGr(j)/wgt_tmp;
				}
				for (j=Nsm_tmp; j<Nn-Nsm_tmp; j++)
				{
					errGr(j)=0;
					for (k=0; k<=2*Nsm_tmp; k++)
					{
						errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsm_tmp);
					}
				}
				for (j=Nn-Nsm_tmp; j<Nn; j++)
				{
					errGr(j)=0;
					wgt_tmp=0;
					for (k=0; k<=Nsm_tmp+Nn-1-j; k++)
					{
						errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsm_tmp);
						wgt_tmp+=wgt_sm(k);
					}
					errGr(j)=errGr(j)/wgt_tmp;
				}
			}
			else
			{
				double b=-log(wgt_min_sm)/Nsm_tmp;
				
				vec wgt_sm(2*Nsm_tmp+1);
				for (j=0; j<Nsm_tmp; j++) wgt_sm(j)=exp(-b*(Nsm_tmp-j));
				wgt_sm(Nsm_tmp)=1;
				for (j=Nsm_tmp+1; j<=2*Nsm_tmp; j++) wgt_sm(j)=wgt_sm(2*Nsm_tmp-j);
				wgt_sm=wgt_sm/sum(wgt_sm);
				//	cout<<"wgt_sm:\n";
				//	for (j=0; j<=2*Nsm_tmp; j++) cout<<setw(20)<<wgt_sm(j);
				//	cout<<endl;
				//	cout<<"sum(wgt_sm): "<<sum(wgt_sm)<<endl;
				
				vec errG_tmp=errGi;
				
				double wgt_tmp;
				for (j=0; j<Nsm_tmp; j++)
				{
					errGi(j)=0;
					wgt_tmp=0;
					for (k=Nsm_tmp-j; k<=2*Nsm_tmp; k++)
					{
						errGi(j)+=wgt_sm(k)*errG_tmp(j+k-Nsm_tmp);
						wgt_tmp+=wgt_sm(k);
					}
					errGi(j)=errGi(j)/wgt_tmp;
				}
				for (j=Nsm_tmp; j<Nn-Nsm_tmp; j++)
				{
					errGi(j)=0;
					for (k=0; k<=2*Nsm_tmp; k++)
					{
						errGi(j)+=wgt_sm(k)*errG_tmp(j+k-Nsm_tmp);
					}
				}
				for (j=Nn-Nsm_tmp; j<Nn; j++)
				{
					errGi(j)=0;
					wgt_tmp=0;
					for (k=0; k<=Nsm_tmp+Nn-1-j; k++)
					{
						errGi(j)+=wgt_sm(k)*errG_tmp(j+k-Nsm_tmp);
						wgt_tmp+=wgt_sm(k);
					}
					errGi(j)=errGi(j)/wgt_tmp;
				}
			}
		}
		else
		{
			double b=-log(wgt_min_sm)/Nsmooth_errG;
			
			vec wgt_sm(2*Nsmooth_errG+1);
			for (j=0; j<Nsmooth_errG; j++) wgt_sm(j)=exp(-b*(Nsmooth_errG-j));
			wgt_sm(Nsmooth_errG)=1;
			for (j=Nsmooth_errG+1; j<=2*Nsmooth_errG; j++) wgt_sm(j)=wgt_sm(2*Nsmooth_errG-j);
			wgt_sm=wgt_sm/sum(wgt_sm);
			//	cout<<"wgt_sm:\n";
			//	for (j=0; j<=2*Nsmooth_errG; j++) cout<<setw(20)<<wgt_sm(j);
			//	cout<<endl;
			//	cout<<"sum(wgt_sm): "<<sum(wgt_sm)<<endl;
			
			vec errG_tmp=errGr;
			
			double wgt_tmp;
			for (j=0; j<Nsmooth_errG; j++)
			{
				errGr(j)=0;
				wgt_tmp=0;
				for (k=Nsmooth_errG-j; k<=2*Nsmooth_errG; k++)
				{
					errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGr(j)=errGr(j)/wgt_tmp;
			}
			for (j=Nsmooth_errG; j<Nn-Nsmooth_errG; j++)
			{
				errGr(j)=0;
				for (k=0; k<=2*Nsmooth_errG; k++)
				{
					errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
				}
			}
			for (j=Nn-Nsmooth_errG; j<Nn; j++)
			{
				errGr(j)=0;
				wgt_tmp=0;
				for (k=0; k<=Nsmooth_errG+Nn-1-j; k++)
				{
					errGr(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGr(j)=errGr(j)/wgt_tmp;
			}
			
			errG_tmp=errGi;
			
			for (j=0; j<Nsmooth_errG; j++)
			{
				errGi(j)=0;
				wgt_tmp=0;
				for (k=Nsmooth_errG-j; k<=2*Nsmooth_errG; k++)
				{
					errGi(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGi(j)=errGi(j)/wgt_tmp;
			}
			for (j=Nsmooth_errG; j<Nn-Nsmooth_errG; j++)
			{
				errGi(j)=0;
				for (k=0; k<=2*Nsmooth_errG; k++)
				{
					errGi(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
				}
			}
			for (j=Nn-Nsmooth_errG; j<Nn; j++)
			{
				errGi(j)=0;
				wgt_tmp=0;
				for (k=0; k<=Nsmooth_errG+Nn-1-j; k++)
				{
					errGi(j)+=wgt_sm(k)*errG_tmp(j+k-Nsmooth_errG);
					wgt_tmp+=wgt_sm(k);
				}
				errGi(j)=errGi(j)/wgt_tmp;
			}
		}
		
		cout<<setiosflags(ios::left);
		errG.zeros(2*Nn);
		for (j=0; j<Nn; j++)
		{
			errG(2*j)=errGr(j);
			errG(2*j+1)=errGi(j);
		//	cout<<setw(20)<<errGr(j)<<errGi(j)<<endl;
		}
		COV.diag()=square(errG);
	}
	else
	{
		cout<<"no errors provided\nusing a constant error\n";
		if (!boson)
		{
			double Gi_max=max(abs(Gi));
			errGi=default_error_G*Gi_max*ones<vec>(Nn);
			errGr=errGi;
			errG=default_error_G*Gi_max*ones<vec>(2*Nn);
		}
		else
		{
			double Gr_max=max(abs(Gr));
			errGr=default_error_G*Gr_max*ones<vec>(Nn);
			errGi=errGr;
			errG=default_error_G*Gr_max*ones<vec>(2*Nn);
		}
		COV.diag()=square(errG);
	}
	
	if (displ_prep_figs && cov_diag)
	{
		graph_2D g1, g2;

		char ttlRe[]="error on $Re[G]$";
		char ttlIm[]="error on $Im[G]$";
		char xl[]="$\\\\omega_n$";
		char yl[]="$\\\\sigma_G^{Re}$";
		char yl2[]="$\\\\sigma_G^{Im}$";
		char attr1[]="'o-', color='b', markeredgecolor='b', markerfacecolor='b'";
		char attr2[]="'o-', color='r', markeredgecolor='r', markerfacecolor='r'";
		
		g1.add_title(ttlRe);
		plot(g1, wn, errGr, xl, yl, attr1);
		g2.add_title(ttlIm);
		plot(g2, wn, errGi, xl, yl2, attr2);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	COV=0.5*(COV+COV.t());
	
	return true;
}

bool OmegaMaxEnt_data::set_G_omega_n_fermions()
{
	bool allow_sign_change=true;
	bool apply_sign_change=false;
	
	int j;
	
	signG=1;
	
	wn=green_data.col(0);
	Nn=green_data.n_rows;
	Gr=green_data.col(col_Gr-1);
	Gi=green_data.col(col_Gi-1);
	
	if (G_omega_inf_in.size())
	{
		Gr=Gr-G_omega_inf;
	}
	
	indG_0=0, indG_f=Nn-1;
	wn_sign_change=false;
	wn_inverted=false;
	if (wn(0)*wn(Nn-1)<0)
	{
		wn_sign_change=true;
		j=1;
		while (wn(j)*wn(Nn-1)<0)
			j=j+1;
		if ((Nn-j)>=j+1)
		{
			indG_0=j;
			indG_f=Nn-1;
		}
		else
		{
			indG_0=0;
			indG_f=j-1;
		}
		wn=wn.rows(indG_0,indG_f);
		Nn=wn.n_rows;
		Gr=Gr.rows(indG_0,indG_f);
		Gi=Gi.rows(indG_0,indG_f);
	}
	if (wn(0)<0)
	{
		wn=abs(wn);
		Gi=-Gi;
	}
	if (wn(0)>wn(Nn-1))
	{
		wn_inverted=true;
		wn=flipud(wn);
		Gr=flipud(Gr);
		Gi=flipud(Gi);
	}
	if (Gi.max()*Gi.min()<0)
	{
		if (allow_sign_change) cout<<"warning: change of sign in Im[G]\n";
		else
		{
			cout<<"error: Im[G] must not change sign\n";
			return false;
		}
	}
	if (Gi.max()>0 && apply_sign_change)
	{
		signG=-1;
	}
	Gr=signG*Gr;
	Gi=signG*Gi;
	
	G.zeros(Nn);
	G.set_real(Gr);
	G.set_imag(Gi);
	
	for (j=0; j<Nn-1; j++)
	{
		if ((wn(j+1)-wn(j))<0)
		{
			cout<<"error: Matsubara frequency is not strictly increasing\n";
			return false;
		}
	}
	
	uvec even_ind=linspace<uvec>(0,2*Nn-2,Nn);
	uvec odd_ind=even_ind+1;
	
	Gchi2.zeros(2*Nn);
	Gchi2.rows(even_ind)=Gr;
	Gchi2.rows(odd_ind)=Gi;
	
	double 	wn_min=wn.min();
	double tem_wn=wn_min/PI;
	
	n.zeros(Nn);
	vec ntmp;
	if (tem_in.size()==0)
	{
		if ( (wn(1)-floor(wn(1)))==0 && (wn(2)-floor(wn(2)))==0 )
		{
			cout<<"If the Matsubara frequency are given by index, temperature must be provided.\n";
			return false;
		}
		tem=tem_wn;
		ntmp=(round(wn/(PI*tem))-1)/2;
		for (j=0; j<Nn; j++)	n(j)=ntmp(j);
	}
	else if ( (abs(tem-tem_wn)/tem)>tol_tem )
	{
		if ( wn(1)-floor(wn(1)) )
		{
			ntmp=(round(wn/(PI*tem_wn))-1)/2;
			for (j=0; j<Nn; j++)	n(j)=ntmp(j);
			cout<<"warning: temperature in file "<<data_file_name<<" is different from temperature given in parameter file "<<input_params_file_name<<endl;
			cout<<"provided temperature: "<<tem<<endl;
			cout<<"temperature extracted from first Matsubara frequency: "<<tem_wn<<endl;
		}
		else if ( (wn(2)-floor(wn(2)))==0 )
		{
			ntmp=round(wn);
			for (j=0; j<Nn; j++) n(j)=ntmp(j);
			n=n-n(0);
		}
		else
		{
			cout<<"oops! First Matsubara frequency is an integer but second is not!\n";
			n=linspace<uvec>(0,Nn-1,Nn);
		}
	}
	else
	{
		ntmp=(round(wn/(PI*tem_wn))-1)/2;
		for (j=0; j<Nn; j++) n(j)=ntmp(j);
	}
	
	cout<<"temperature: "<<tem<<endl;
	
	wn=PI*tem*conv_to<vec>::from(2*n+1);
	
	if (omega_n_trunc_in.size())
	{
		j=Nn-1;
		while (j>Nn_min && wn(j)>omega_n_trunc) j--;
		if (j<Nn-1 && j>Nn_min)
		{
			n=n.rows(0,j);
			wn=wn.rows(0,j);
			G=G.rows(0,j);
			Gr=Gr.rows(0,j);
			Gi=Gi.rows(0,j);
			Gchi2=Gchi2.rows(0,2*j+1);
			Nn=j+1;
		}
		else if (j==Nn_min)
		{
			cout<<"warning: truncation frequency ignored because too small.\n";
		}
	}
	
	cout<<"Number of Matsubara frequencies in the Green function: "<<Nn<<endl;
	
	if (displ_prep_figs)
	{
		graph_2D g1, g2;
		
//		graph_2D::show_commands(true);
		
		char ttlRe[]="Real part of data";
		char ttlIm[]="Imaginary part of data";
		char xl[]="$\\\\omega_n$";
		char yl1[]="$Re[G]$";
		char yl2[]="$Im[G]$";
		char attr1[]="'o-', color='b', markeredgecolor='b', markerfacecolor='b'";
		char attr2[]="'o-', color='r', markeredgecolor='r', markerfacecolor='r'";
		
		g1.add_title(ttlRe);
		plot(g1,wn,Gr,xl,yl1,attr1);
		g2.add_title(ttlIm);
		plot(g2,wn,Gi,xl,yl2,attr2);
		
		if (graph_2D::display_figures) cout<<"close the figures to resume execution\n";
		graph_2D::show_figures();
	}
	
	return true;
}

void OmegaMaxEnt_data::plot(graph_2D &g, vec x, vec y, char *xl, char *yl, char *attr)
{
	double eps=1e-6;
	double xlims[2], ylims[2];
	xlims[0]=0;
	xlims[1]=0;
	ylims[0]=0;
	ylims[1]=0;
	
	double extra_x=0.05;
	double extra_y=0.05;
	
	double xmin=x.min();
	double xmax=x.max();
	double xmid=(xmin+xmax)/2;
	if (xmid!=0)
	{
		if (abs((xmax-xmin)/xmid)>eps)
		{
			xlims[0]=xmin-extra_x*(xmax-xmin);
			xlims[1]=xmax+extra_x*(xmax-xmin);
		}
		else
		{
			if (xmin>0)
				xlims[1]=(1.0+extra_x)*xmax;
			else if (xmax<0)
				xlims[0]=(1.0+extra_x)*xmin;
			else
			{
				xlims[0]=(1.0+extra_x)*xmin;
				xlims[1]=(1.0+extra_x)*xmax;
			}
		}
	}
	else if (xmin==-xmax && xmax!=0)
	{
		xlims[0]=xmin-extra_x*(xmax-xmin);
		xlims[1]=xmax+extra_x*(xmax-xmin);
	}
	
	double ymax=y.max();
	double ymin=y.min();
	double ymid=(ymin+ymax)/2;
	if (ymid!=0)
	{
		if (abs((ymax-ymin)/ymid)>eps)
		{
			ylims[0]=ymin-extra_y*(ymax-ymin);
			ylims[1]=ymax+extra_y*(ymax-ymin);
		}
		else
		{
			if (ymin>0)
				ylims[1]=(1.0+extra_y)*ymax;
			else if (ymax<0)
				ylims[0]=(1.0+extra_y)*ymin;
			else
			{
				ylims[0]=(1.0+extra_y)*ymin;
				ylims[1]=(1.0+extra_y)*ymax;
			}
		}
	}
	else if (ymin==-ymax && ymax!=0)
	{
		ylims[0]=ymin-extra_y*(ymax-ymin);
		ylims[1]=ymax+extra_y*(ymax-ymin);
	}
	
//	cout<<"plot():x.n_rows: "<<x.n_rows<<endl;
	
	g.add_data(x.memptr(),y.memptr(),x.n_rows);
	g.set_axes_labels(xl,yl);
	g.set_axes_lims(xlims,ylims);
	if (attr) g.add_attribute(attr);
	g.curve_plot();
}

bool OmegaMaxEnt_data::load_data_file(mat &data_array, string file_name)
{
	string complete_file_name(input_dir);
	complete_file_name+=file_name;
	
	if (data_array.load(complete_file_name.c_str(),auto_detect))
		return true;
	else if (data_array.load(file_name.c_str(),auto_detect))
		return true;
	else
	{
		cout<<"file "<<file_name<<" could not be opened\n";
		return false;
	}
}

void OmegaMaxEnt_data::init_params()
{
	input_dir.assign("./");
	boson=false;
	tau_GF=false;
	Ginf_finite=false;
	col_Gr=2;
	col_Gi=3;
	col_errGr=2;
	col_errGi=3;
	col_Gtau=2;
	col_errGtau=2;
	ind_noise=0;
	N_params_noise=0;
	non_uniform_grid=false;
	use_grid_params=false;
	eval_moments=false;
	compute_Pade=false;
	displ_prep_figs=false;
	displ_adv_prep_figs=false;
	print_other_params=false;
	print_alpha=false;
	show_optimal_alpha_figs=true;
	show_lowest_alpha_figs=true;
	show_alpha_curves=true;
}

bool OmegaMaxEnt_data::load_input_params()
{
	SC_set=false;
	SW_set=false;
	M1_set=false;
	M2_set=false;
	execute_maxent=true;
	A_ref_change=false;
	w_origin_set=false;
	data_file_loaded=false;
	
	ifstream file(input_params_file_name);
	
	string data_file_line("data file:");
	
	if (file)
	{
		string str;
		cout<<"\nINPUT PARAMETERS:\n";
		getline(file,str);
		while (!file.eof() && str.compare(0,Input_files_params[INPUT_DIR].size(),Input_files_params[INPUT_DIR])) getline(file,str);
		if (!file.eof())
		{
			str=str.substr(Input_files_params[INPUT_DIR].size());
			remove_spaces_ends(str);
			if (input_dir_in.compare(str)) initialize=true;
			input_dir_in=str;
			input_dir=input_dir_in;
			if (input_dir_in.size())
			{
				if (input_dir.back()!='/') input_dir.push_back('/');
				cout<<Input_files_params[INPUT_DIR]<<" "<<input_dir<<endl;
			}
			else
				input_dir.assign("./");
		}
	/*
		while (!file.eof())
		{
			if (str.compare(0,Input_files_params[INPUT_DIR].size(),Input_files_params[INPUT_DIR])==0)
			{
				str=str.substr(Input_files_params[INPUT_DIR].size());
				remove_spaces_ends(str);
				if (input_dir_in.compare(str)) initialize=true;
				input_dir_in=str;
				input_dir=input_dir_in;
				if (input_dir_in.size())
				{
					if (input_dir.back()!='/') input_dir.push_back('/');
					cout<<Input_files_params[INPUT_DIR]<<" "<<input_dir<<endl;
				}
				else
					input_dir.assign("./");
			}
			getline(file,str);
		}
	*/
		file.clear();
		file.seekg(0);
		getline(file,str);
		while (!file.eof() && str.compare(0,data_file_param.size(),data_file_param)) getline(file,str);
		if (!file.eof())
		{
			str=str.substr(data_file_param.size());
			remove_spaces_ends(str);
			if (data_file_name_in.compare(str)) initialize=true;
			data_file_name_in=str;
			data_file_name=data_file_name_in;
			if (data_file_name.size())
			{
				cout<<data_file_param<<" "<<data_file_name<<endl;
				data_file_loaded=load_data_file(green_data, data_file_name);
				if (data_file_loaded)
					cout<<"data file loaded\n";
				else
				{
					cout<<"unable to load data file "<<data_file_name<<endl;
					return false;
				}
			}
			else
			{
				cout<<"error: parameter \"data file\" empty\n";
				return false;
			}
		}
	/*
		while (!file.eof())
		{
			//			cout<<str<<endl;
			if (str.compare(0,data_file_param.size(),data_file_param)==0)
			{
				str=str.substr(data_file_param.size());
				remove_spaces_ends(str);
				if (data_file_name_in.compare(str)) initialize=true;
				data_file_name_in=str;
				data_file_name=data_file_name_in;
				if (data_file_name.size())
				{
					cout<<data_file_param<<" "<<data_file_name<<endl;
					data_file_loaded=load_data_file(green_data, data_file_name);
					if (data_file_loaded)
						cout<<"data file loaded\n";
					else
					{
						cout<<"unable to load data file "<<data_file_name<<endl;
						return false;
					}
				}
				else
				{
					cout<<"error: parameter \"data file\" empty\n";
					return false;
				}
			}
	 		getline(file,str);
		}
	*/
		file.clear();
		file.seekg(0);
		getline(file,str);
		while (!file.eof())
		{
			//			cout<<str<<endl;
		/*
			if (str.compare(0,data_file_param.size(),data_file_param)==0)
			{
				str=str.substr(data_file_param.size());
				remove_spaces_ends(str);
				if (data_file_name_in.compare(str)) initialize=true;
				data_file_name_in=str;
				data_file_name=data_file_name_in;
				if (data_file_name.size())
				{
					cout<<data_file_param<<" "<<data_file_name<<endl;
					if (load_data_file(green_data, data_file_name))
						cout<<"data file loaded\n";
					else
						return false;
				}
				else
				{
					cout<<"error: parameter \"data file\" empty\n";
					return false;
				}
			}
			else if (str.compare(0,Data_params[BOSON].size(),Data_params[BOSON])==0)
		 */
			if (str.compare(0,Data_params[BOSON].size(),Data_params[BOSON])==0)
			{
				str=str.substr(Data_params[BOSON].size());
				remove_spaces_ends(str);
				if (boson_in.compare(str)) initialize=true;
				boson_in=str;
				boson=false;
				if (boson_in.size())
				{
					cout<<Data_params[BOSON]<<" "<<boson_in<<endl;
					if (boson_in.compare("yes")==0) boson=true;
				}
			}
			else if (str.compare(0,Data_params[TAU_GF].size(),Data_params[TAU_GF])==0)
			{
				str=str.substr(Data_params[TAU_GF].size());
				remove_spaces_ends(str);
				if (tau_GF_in.compare(str)) initialize=true;
				tau_GF_in=str;
				tau_GF=false;
				if (tau_GF_in.size())
				{
					cout<<Data_params[TAU_GF]<<" "<<tau_GF_in<<endl;
					if (tau_GF_in.compare("yes")==0) tau_GF=true;
				}
			}
			else if (str.compare(0,Data_params[TEM].size(),Data_params[TEM])==0)
			{
				str=str.substr(Data_params[TEM].size());
				remove_spaces_ends(str);
				if (tem_in.compare(str)) initialize=true;
				tem_in=str;
				if (tem_in.size())
				{
					cout<<Data_params[TEM]<<" "<<tem_in<<endl;
					tem=stod(tem_in);
				}
			}
			else if (str.compare(0,Data_params[G_INF_FINITE].size(),Data_params[G_INF_FINITE])==0)
			{
				str=str.substr(Data_params[G_INF_FINITE].size());
				remove_spaces_ends(str);
				if (Ginf_finite_in.compare(str)) initialize=true;
				Ginf_finite_in=str;
				Ginf_finite=false;
				if (Ginf_finite_in.size())
				{
					cout<<Data_params[G_INF_FINITE]<<" "<<Ginf_finite_in<<endl;
					if (Ginf_finite_in.compare("yes")==0) Ginf_finite=true;
				}
			}
			else if (str.compare(0,Data_params[G_INF].size(),Data_params[G_INF])==0)
			{
				str=str.substr(Data_params[G_INF].size());
				remove_spaces_ends(str);
				if (G_omega_inf_in.compare(str)) initialize=true;
				G_omega_inf_in=str;
				if (G_omega_inf_in.size())
				{
					G_omega_inf=stod(G_omega_inf_in);
					cout<<Data_params[G_INF]<<" "<<G_omega_inf<<endl;
					if (!G_omega_inf) G_omega_inf_in.clear();
				}
			}
			else if (str.compare(0,Data_params[NORM_A].size(),Data_params[NORM_A])==0)
			{
				str=str.substr(Data_params[NORM_A].size());
				remove_spaces_ends(str);
				if (M0_in.compare(str)) initialize=true;
				M0_in=str;
				if (M0_in.size())
				{
					cout<<Data_params[NORM_A]<<" "<<M0_in<<endl;
				}
			}
			else if (str.compare(0,Data_params[M_1].size(),Data_params[M_1])==0)
			{
				str=str.substr(Data_params[M_1].size());
				remove_spaces_ends(str);
				if (M1_in.compare(str)) initialize=true;
				M1_in=str;
				if (M1_in.size())
				{
					cout<<Data_params[M_1]<<" "<<M1_in<<endl;
				}
			}
			else if (str.compare(0,Data_params[ERR_M1].size(),Data_params[ERR_M1])==0)
			{
				str=str.substr(Data_params[ERR_M1].size());
				remove_spaces_ends(str);
				if (errM1_in.compare(str)) initialize=true;
				errM1_in=str;
				if (errM1_in.size())
				{
					cout<<Data_params[ERR_M1]<<" "<<errM1_in<<endl;
				}
			}
			else if (str.compare(0,Data_params[M_2].size(),Data_params[M_2])==0)
			{
				str=str.substr(Data_params[M_2].size());
				remove_spaces_ends(str);
				if (M2_in.compare(str)) initialize=true;
				M2_in=str;
				if (M2_in.size())
				{
					cout<<Data_params[M_2]<<" "<<M2_in<<endl;
				}
			}
			else if (str.compare(0,Data_params[ERR_M2].size(),Data_params[ERR_M2])==0)
			{
				str=str.substr(Data_params[ERR_M2].size());
				remove_spaces_ends(str);
				if (errM2_in.compare(str)) initialize=true;
				errM2_in=str;
				if (errM2_in.size())
				{
					cout<<Data_params[ERR_M2]<<" "<<errM2_in<<endl;
				}
			}
			else if (str.compare(0,Data_params[M_3].size(),Data_params[M_3])==0)
			{
				str=str.substr(Data_params[M_3].size());
				remove_spaces_ends(str);
				if (M3_in.compare(str)) initialize=true;
				M3_in=str;
				if (M3_in.size())
				{
					cout<<Data_params[M_3]<<" "<<M3_in<<endl;
				}
			}
			else if (str.compare(0,Data_params[ERR_M3].size(),Data_params[ERR_M3])==0)
			{
				str=str.substr(Data_params[ERR_M3].size());
				remove_spaces_ends(str);
				if (errM3_in.compare(str)) initialize=true;
				errM3_in=str;
				if (errM3_in.size())
				{
					cout<<Data_params[ERR_M3]<<" "<<errM3_in<<endl;
				}
			}
			else if (str.compare(0,Data_params[TRUNC_FREQ].size(),Data_params[TRUNC_FREQ])==0)
			{
				str=str.substr(Data_params[TRUNC_FREQ].size());
				remove_spaces_ends(str);
				if (omega_n_trunc_in.compare(str)) initialize=true;
				omega_n_trunc_in=str;
				if (omega_n_trunc_in.size())
				{
					cout<<Data_params[TRUNC_FREQ]<<" "<<omega_n_trunc_in<<endl;
					omega_n_trunc=stod(omega_n_trunc_in);
				}
			}
			else if (str.compare(0,Input_files_params[COL_GR].size(),Input_files_params[COL_GR])==0)
			{
				str=str.substr(Input_files_params[COL_GR].size());
				remove_spaces_ends(str);
				if (col_Gr_in.compare(str)) initialize=true;
				col_Gr_in=str;
				if (col_Gr_in.size())
				{
					cout<<Input_files_params[COL_GR]<<" "<<col_Gr_in<<endl;
					col_Gr=stoi(col_Gr_in);
				}
				else
					col_Gr=2;
			}
			else if (str.compare(0,Input_files_params[COL_GI].size(),Input_files_params[COL_GI])==0)
			{
				str=str.substr(Input_files_params[COL_GI].size());
				remove_spaces_ends(str);
				if (col_Gi_in.compare(str)) initialize=true;
				col_Gi_in=str;
				if (col_Gi_in.size())
				{
					cout<<Input_files_params[COL_GI]<<" "<<col_Gi_in<<endl;
					col_Gi=stoi(col_Gi_in);
				}
				else
					col_Gi=3;
			}
			else if (str.compare(0,Input_files_params[ERROR_FILE].size(),Input_files_params[ERROR_FILE])==0)
			{
				str=str.substr(Input_files_params[ERROR_FILE].size());
				remove_spaces_ends(str);
				if (error_file_in.compare(str)) initialize=true;
				error_file_in=str;
				if (error_file_in.size())
				{
					cout<<Input_files_params[ERROR_FILE]<<" "<<error_file_in<<endl;
					if (load_data_file(error_data, error_file_in))
					{
						cout<<Input_files_params[ERROR_FILE].substr(0,Input_files_params[ERROR_FILE].size()-1)<<" loaded\n";
						error_file=error_file_in;
					}
					else
						error_file.clear();
				}
				else
					error_file.clear();
			}
			else if (str.compare(0,Input_files_params[COL_ERR_GR].size(),Input_files_params[COL_ERR_GR])==0)
			{
				str=str.substr(Input_files_params[COL_ERR_GR].size());
				remove_spaces_ends(str);
				if (col_errGr_in.compare(str)) initialize=true;
				col_errGr_in=str;
				if (col_errGr_in.size())
				{
					cout<<Input_files_params[COL_ERR_GR]<<" "<<col_errGr_in<<endl;
					col_errGr=stoi(col_errGr_in);
				}
				else
					col_errGr=2;
			}
			else if (str.compare(0,Input_files_params[COL_ERR_GI].size(),Input_files_params[COL_ERR_GI])==0)
			{
				str=str.substr(Input_files_params[COL_ERR_GI].size());
				remove_spaces_ends(str);
				if (col_errGi_in.compare(str)) initialize=true;
				col_errGi_in=str;
				if (col_errGi_in.size())
				{
					cout<<Input_files_params[COL_ERR_GI]<<" "<<col_errGi_in<<endl;
					col_errGi=stoi(col_errGi_in);
				}
				else
					col_errGi=3;
			}
			else if (str.compare(0,Input_files_params[COVAR_RE_RE_FILE].size(),Input_files_params[COVAR_RE_RE_FILE])==0)
			{
				str=str.substr(Input_files_params[COVAR_RE_RE_FILE].size());
				remove_spaces_ends(str);
				if (covar_re_re_file_in.compare(str)) initialize=true;
				covar_re_re_file_in=str;
				if (covar_re_re_file_in.size())
				{
					cout<<Input_files_params[COVAR_RE_RE_FILE]<<" "<<covar_re_re_file_in<<endl;
					if (load_data_file(CRR, covar_re_re_file_in))
					{
						cout<<Input_files_params[COVAR_RE_RE_FILE].substr(0,Input_files_params[COVAR_RE_RE_FILE].size()-1)<<" loaded\n";
						covar_re_re_file=covar_re_re_file_in;
					}
					else
						covar_re_re_file.clear();
				}
				else
					covar_re_re_file.clear();
			}
			else if (str.compare(0,Input_files_params[COVAR_IM_IM_FILE].size(),Input_files_params[COVAR_IM_IM_FILE])==0)
			{
				str=str.substr(Input_files_params[COVAR_IM_IM_FILE].size());
				remove_spaces_ends(str);
				if (covar_im_im_file_in.compare(str)) initialize=true;
				covar_im_im_file_in=str;
				if (covar_im_im_file_in.size())
				{
					cout<<Input_files_params[COVAR_IM_IM_FILE]<<" "<<covar_im_im_file_in<<endl;
					if (load_data_file(CII, covar_im_im_file_in))
					{
						cout<<Input_files_params[COVAR_IM_IM_FILE].substr(0,Input_files_params[COVAR_IM_IM_FILE].size()-1)<<" loaded\n";
						covar_im_im_file=covar_im_im_file_in;
					}
					else
						covar_im_im_file.clear();
				}
				else
					covar_im_im_file.clear();
			}
			else if (str.compare(0,Input_files_params[COVAR_RE_IM_FILE].size(),Input_files_params[COVAR_RE_IM_FILE])==0)
			{
				str=str.substr(Input_files_params[COVAR_RE_IM_FILE].size());
				remove_spaces_ends(str);
				if (covar_re_im_file_in.compare(str)) initialize=true;
				covar_re_im_file_in=str;
				if (covar_re_im_file_in.size())
				{
					cout<<Input_files_params[COVAR_RE_IM_FILE]<<" "<<covar_re_im_file_in<<endl;
					if (load_data_file(CRI, covar_re_im_file_in))
					{
						cout<<Input_files_params[COVAR_RE_IM_FILE].substr(0,Input_files_params[COVAR_RE_IM_FILE].size()-1)<<" loaded\n";
						covar_re_im_file=covar_re_im_file_in;
					}
					else
						covar_re_im_file.clear();
				}
				else
					covar_re_im_file.clear();
			}
			else if (str.compare(0,Input_files_params[COL_G_TAU].size(),Input_files_params[COL_G_TAU])==0)
			{
				str=str.substr(Input_files_params[COL_G_TAU].size());
				remove_spaces_ends(str);
				if (col_Gtau_in.compare(str)) initialize=true;
				col_Gtau_in=str;
				if (col_Gtau_in.size())
				{
					cout<<Input_files_params[COL_G_TAU]<<" "<<col_Gtau_in<<endl;
					col_Gtau=stoi(col_Gtau_in);
				}
				else
					col_Gtau=2;
			}
			else if (str.compare(0,Input_files_params[COL_ERR_G_TAU].size(),Input_files_params[COL_ERR_G_TAU])==0)
			{
				str=str.substr(Input_files_params[COL_ERR_G_TAU].size());
				remove_spaces_ends(str);
				if (col_errGtau_in.compare(str)) initialize=true;
				col_errGtau_in=str;
				if (col_errGtau_in.size())
				{
					cout<<Input_files_params[COL_ERR_G_TAU]<<" "<<col_errGtau_in<<endl;
					col_errGtau=stoi(col_errGtau_in);
				}
				else
					col_errGtau=2;
			}
			else if (str.compare(0,Input_files_params[COVAR_TAU_FILE].size(),Input_files_params[COVAR_TAU_FILE])==0)
			{
				str=str.substr(Input_files_params[COVAR_TAU_FILE].size());
				remove_spaces_ends(str);
				if (covar_tau_file_in.compare(str))	initialize=true;
				covar_tau_file_in=str;
				if (covar_tau_file_in.size())
				{
					cout<<Input_files_params[COVAR_TAU_FILE]<<" "<<covar_tau_file_in<<endl;
					if (load_data_file(Ctau, covar_tau_file_in))
					{
						cout<<Input_files_params[COVAR_TAU_FILE].substr(0,Input_files_params[COVAR_TAU_FILE].size()-1)<<" loaded\n";
						covar_tau_file=covar_tau_file_in;
					}
					else
						covar_tau_file.clear();
				}
				else
					covar_tau_file.clear();
			}
			else if (str.compare(0,Input_files_params[NOISE_PARAMS].size(),Input_files_params[NOISE_PARAMS])==0)
			{
				str=str.substr(Input_files_params[NOISE_PARAMS].size());
				remove_spaces_ends(str);
				if (noise_params_in.compare(str))
				{
					initialize=true;
					ind_noise=0;
				}
				noise_params_in=str;
				if (noise_params_in.size())
				{
					cout<<Input_files_params[NOISE_PARAMS]<<" "<<noise_params_in<<endl;
					noise_params=rowvec(noise_params_in);
					N_params_noise=noise_params.n_cols;
					//	cout<<"N_params_noise: "<<N_params_noise<<endl;
					//	cout<<"noise_params: "<<noise_params<<endl;
				}
				else
				{
					ind_noise=0;
					N_params_noise=0;
					noise_params.reset();
				}
			}
			else if (str.compare(0,Grid_params[CUTOFF_WN].size(),Grid_params[CUTOFF_WN])==0)
			{
				str=str.substr(Grid_params[CUTOFF_WN].size());
				remove_spaces_ends(str);
				if (cutoff_wn_in.compare(str))	initialize=true;
				cutoff_wn_in=str;
				if (cutoff_wn_in.size())
				{
					cout<<Grid_params[CUTOFF_WN]<<" "<<cutoff_wn_in<<endl;
					cutoff_wn=stod(cutoff_wn_in);
				}
			}
			else if (str.compare(0,Grid_params[SPECTR_FUNC_WIDTH].size(),Grid_params[SPECTR_FUNC_WIDTH])==0)
			{
				str=str.substr(Grid_params[SPECTR_FUNC_WIDTH].size());
				remove_spaces_ends(str);
				if (SW_in.compare(str))	initialize=true;
				SW_in=str;
				if (SW_in.size())
				{
					cout<<Grid_params[SPECTR_FUNC_WIDTH]<<" "<<SW_in<<endl;
					SW=stod(SW_in);
					SW_set=true;
				}
			}
			else if (str.compare(0,Grid_params[SPECTR_FUNC_CENTER].size(),Grid_params[SPECTR_FUNC_CENTER])==0)
			{
				str=str.substr(Grid_params[SPECTR_FUNC_CENTER].size());
				remove_spaces_ends(str);
				if (SC_in.compare(str))	initialize=true;
				SC_in=str;
				if (SC_in.size())
				{
					cout<<Grid_params[SPECTR_FUNC_CENTER]<<" "<<SC_in<<endl;
					SC=stod(SC_in);
					SC_set=true;
				}
			}
			else if (str.compare(0,Grid_params[W_ORIGIN].size(),Grid_params[W_ORIGIN])==0)
			{
				str=str.substr(Grid_params[W_ORIGIN].size());
				remove_spaces_ends(str);
				if (w_origin_in.compare(str))	initialize=true;
				w_origin_in=str;
				if (w_origin_in.size())
				{
					cout<<Grid_params[W_ORIGIN]<<" "<<w_origin_in<<endl;
					w_origin=stod(w_origin_in);
					w_origin_set=true;
				}
			}
			else if (str.compare(0,Grid_params[STEP_W].size(),Grid_params[STEP_W])==0)
			{
				str=str.substr(Grid_params[STEP_W].size());
				remove_spaces_ends(str);
				if (step_omega_in.compare(str))	initialize=true;
				step_omega_in=str;
				if (step_omega_in.size())
				{
					cout<<Grid_params[STEP_W]<<" "<<step_omega_in<<endl;
					step_omega=stod(step_omega_in);
				}
			}
			else if (str.compare(0,Grid_params[GRID_W_FILE].size(),Grid_params[GRID_W_FILE])==0)
			{
				str=str.substr(Grid_params[GRID_W_FILE].size());
				remove_spaces_ends(str);
				if (grid_omega_file_in.compare(str))  initialize=true;
				grid_omega_file_in=str;
				if (grid_omega_file_in.size())
				{
					cout<<Grid_params[GRID_W_FILE]<<" "<<grid_omega_file_in<<endl;
					if (load_data_file(grid_w_data, grid_omega_file_in))
					{
						cout<<Grid_params[GRID_W_FILE].substr(0,Grid_params[GRID_W_FILE].size()-1)<<" loaded\n";
						grid_omega_file=grid_omega_file_in;
					}
					else
						grid_omega_file.clear();
				}
				else
					grid_omega_file.clear();
			}
			else if (str.compare(0,Grid_params[NON_UNIFORM_GRID].size(),Grid_params[NON_UNIFORM_GRID])==0)
			{
				str=str.substr(Grid_params[NON_UNIFORM_GRID].size());
				remove_spaces_ends(str);
				if (non_uniform_grid_in.compare(str))	initialize=true;
				non_uniform_grid_in=str;
				non_uniform_grid=false;
				if (non_uniform_grid_in.size())
				{
					cout<<Grid_params[NON_UNIFORM_GRID]<<" "<<non_uniform_grid_in<<endl;
					if (non_uniform_grid_in.compare("yes")==0) non_uniform_grid=true;
				}
			}
			else if (str.compare(0,Grid_params[USE_GRID_PARAMS].size(),Grid_params[USE_GRID_PARAMS])==0)
			{
				str=str.substr(Grid_params[USE_GRID_PARAMS].size());
				remove_spaces_ends(str);
				if (use_grid_params_in.compare(str))	initialize=true;
				use_grid_params_in=str;
				use_grid_params=false;
				if (use_grid_params_in.size())
				{
					cout<<Grid_params[USE_GRID_PARAMS]<<" "<<use_grid_params_in<<endl;
					if (use_grid_params_in.compare("yes")==0) use_grid_params=true;
				}
			}
			else if (str.compare(0,Grid_params[PARAM_GRID_PARAMS].size(),Grid_params[PARAM_GRID_PARAMS])==0)
			{
				str=str.substr(Grid_params[PARAM_GRID_PARAMS].size());
				remove_spaces_ends(str);
				if (omega_grid_params_in.compare(str))	initialize=true;
				omega_grid_params_in=str;
				if (omega_grid_params_in.size())
				{
					cout<<Grid_params[PARAM_GRID_PARAMS]<<" "<<omega_grid_params_in<<endl;
					omega_grid_params=rowvec(omega_grid_params_in);
					if (omega_grid_params.n_cols<=2) omega_grid_params.reset();
				}
				else
					omega_grid_params.reset();
			}
			else if (str.compare(0,Grid_params[OUTPUT_GRID_PARAMS].size(),Grid_params[OUTPUT_GRID_PARAMS])==0)
			{
				str=str.substr(Grid_params[OUTPUT_GRID_PARAMS].size());
				remove_spaces_ends(str);
				output_grid_params_in=str;
				if (output_grid_params_in.size())
				{
					cout<<Grid_params[OUTPUT_GRID_PARAMS]<<" "<<output_grid_params_in<<endl;
					output_grid_params=rowvec(output_grid_params_in);
					if (output_grid_params.n_cols<=2) output_grid_params.reset();
				}
				else
					output_grid_params.reset();
			}
			else if (str.compare(0,Preproc_comp_params[EVAL_MOMENTS].size(),Preproc_comp_params[EVAL_MOMENTS])==0)
			{
				str=str.substr(Preproc_comp_params[EVAL_MOMENTS].size());
				remove_spaces_ends(str);
				if (eval_moments_in.compare(str))	initialize=true;
				eval_moments_in=str;
				eval_moments=false;
				if (eval_moments_in.size())
				{
					cout<<Preproc_comp_params[EVAL_MOMENTS]<<" "<<eval_moments_in<<endl;
					if (eval_moments_in.compare("yes")==0) eval_moments=true;
				}
			}
			else if (str.compare(0,Preproc_comp_params[MAX_M].size(),Preproc_comp_params[MAX_M])==0)
			{
				str=str.substr(Preproc_comp_params[MAX_M].size());
				remove_spaces_ends(str);
				if (maxM_in.compare(str))	initialize=true;
				maxM_in=str;
				if (maxM_in.size())
				{
					cout<<Preproc_comp_params[MAX_M]<<" "<<maxM_in<<endl;
					maxM=stoi(maxM_in);
				}
			}
			else if (str.compare(0,Preproc_comp_params[DEFAULT_MODEL_CENTER].size(),Preproc_comp_params[DEFAULT_MODEL_CENTER])==0)
			{
				str=str.substr(Preproc_comp_params[DEFAULT_MODEL_CENTER].size());
				remove_spaces_ends(str);
				if (default_model_center_in.compare(str))	initialize=true;
				default_model_center_in=str;
				if (default_model_center_in.size())
				{
					cout<<Preproc_comp_params[DEFAULT_MODEL_CENTER]<<" "<<default_model_center_in<<endl;
					default_model_center=stod(default_model_center_in);
				}
			}
			else if (str.compare(0,Preproc_comp_params[DEFAULT_MODEL_WIDTH].size(),Preproc_comp_params[DEFAULT_MODEL_WIDTH])==0)
			{
				str=str.substr(Preproc_comp_params[DEFAULT_MODEL_WIDTH].size());
				remove_spaces_ends(str);
				if (default_model_width_in.compare(str))	initialize=true;
				default_model_width_in=str;
				if (default_model_width_in.size())
				{
					cout<<Preproc_comp_params[DEFAULT_MODEL_WIDTH]<<" "<<default_model_width_in<<endl;
					default_model_width=stod(default_model_width_in);
				}
			}
			else if (str.compare(0,Preproc_comp_params[DEFAULT_MODEL_SHAPE].size(),Preproc_comp_params[DEFAULT_MODEL_SHAPE])==0)
			{
				str=str.substr(Preproc_comp_params[DEFAULT_MODEL_SHAPE].size());
				remove_spaces_ends(str);
				if (default_model_shape_in.compare(str))	initialize=true;
				default_model_shape_in=str;
				if (default_model_shape_in.size())
				{
					cout<<Preproc_comp_params[DEFAULT_MODEL_SHAPE]<<" "<<default_model_shape_in<<endl;
					default_model_shape=stod(default_model_shape_in);
					if (default_model_shape<=1)
						cout<<"warning: it is recommended to use a default model shape parameter larger than 1\n";
				}
			}
			else if (str.compare(0,Preproc_comp_params[DEFAULT_MODEL_FILE].size(),Preproc_comp_params[DEFAULT_MODEL_FILE])==0)
			{
				str=str.substr(Preproc_comp_params[DEFAULT_MODEL_FILE].size());
				remove_spaces_ends(str);
				if (def_model_file_in.compare(str))	initialize=true;
				def_model_file_in=str;
				if (def_model_file_in.size())
				{
					cout<<Preproc_comp_params[DEFAULT_MODEL_FILE]<<" "<<def_model_file_in<<endl;
					if (load_data_file(def_data, def_model_file_in))
					{
						cout<<Preproc_comp_params[DEFAULT_MODEL_FILE].substr(0,Preproc_comp_params[DEFAULT_MODEL_FILE].size()-1)<<" loaded\n";
						def_model_file=def_model_file_in;
					}
					else
						def_model_file.clear();
				}
				else
					def_model_file.clear();
			}
			else if (str.compare(0,Preproc_comp_params[INIT_SPECTR_FUNC_FILE].size(),Preproc_comp_params[INIT_SPECTR_FUNC_FILE])==0)
			{
				str=str.substr(Preproc_comp_params[INIT_SPECTR_FUNC_FILE].size());
				remove_spaces_ends(str);
				if (init_spectr_func_file_in.compare(str))	initialize=true;
				init_spectr_func_file_in=str;
				if (init_spectr_func_file_in.size())
				{
					cout<<Preproc_comp_params[INIT_SPECTR_FUNC_FILE]<<" "<<init_spectr_func_file_in<<endl;
					if (load_data_file(Aw_data, init_spectr_func_file_in))
					{
						cout<<Preproc_comp_params[INIT_SPECTR_FUNC_FILE].substr(0,Preproc_comp_params[INIT_SPECTR_FUNC_FILE].size()-1)<<" loaded\n";
						init_spectr_func_file=init_spectr_func_file_in;
					}
					else
						init_spectr_func_file.clear();
				}
				else
					init_spectr_func_file.clear();
			}
			else if (str.compare(0,Preproc_comp_params[COMPUTE_PADE].size(),Preproc_comp_params[COMPUTE_PADE])==0)
			{
				str=str.substr(Preproc_comp_params[COMPUTE_PADE].size());
				remove_spaces_ends(str);
				if (compute_Pade_in.compare(str))	initialize=true;
				compute_Pade_in=str;
				compute_Pade=false;
				if (compute_Pade_in.size())
				{
					cout<<Preproc_comp_params[COMPUTE_PADE]<<" "<<compute_Pade_in<<endl;
					if (compute_Pade_in.compare("yes")==0) compute_Pade=true;
				}
			}
			else if (str.compare(0,Preproc_comp_params[N_PADE].size(),Preproc_comp_params[N_PADE])==0)
			{
				str=str.substr(Preproc_comp_params[N_PADE].size());
				remove_spaces_ends(str);
				if (N_Pade_in.compare(str))	initialize=true;
				N_Pade_in=str;
				if (N_Pade_in.size())
				{
					cout<<Preproc_comp_params[N_PADE]<<" "<<N_Pade_in<<endl;
					N_Pade=stoi(N_Pade_in);
				}
			}
			else if (str.compare(0,Preproc_comp_params[ETA_PADE].size(),Preproc_comp_params[ETA_PADE])==0)
			{
				str=str.substr(Preproc_comp_params[ETA_PADE].size());
				remove_spaces_ends(str);
				if (eta_Pade_in.compare(str))	initialize=true;
				eta_Pade_in=str;
				if (eta_Pade_in.size())
				{
					cout<<Preproc_comp_params[ETA_PADE]<<" "<<eta_Pade_in<<endl;
					eta_Pade=stod(eta_Pade_in);
				}
			}
			else if (str.compare(0,Preproc_exec_params[PREPROSSESS_ONLY].size(),Preproc_exec_params[PREPROSSESS_ONLY])==0)
			{
				str=str.substr(Preproc_exec_params[PREPROSSESS_ONLY].size());
				remove_spaces_ends(str);
				if (str.size())
				{
					cout<<Preproc_exec_params[PREPROSSESS_ONLY]<<" "<<str<<endl;
					if (str.compare("yes")==0)
					{
						execute_maxent=false;
					}
				}
				//                if (preprocess_only) initialize=true;
			}
			else if (str.compare(0,Preproc_exec_params[DISPL_PREP_FIGS].size(),Preproc_exec_params[DISPL_PREP_FIGS])==0)
			{
				str=str.substr(Preproc_exec_params[DISPL_PREP_FIGS].size());
				remove_spaces_ends(str);
				displ_prep_figs=false;
				if (str.size())
				{
					cout<<Preproc_exec_params[DISPL_PREP_FIGS]<<" "<<str<<endl;
					if (str.compare("yes")==0) displ_prep_figs=true;
				}
			}
			else if (str.compare(0,Preproc_exec_params[DISPL_ADV_PREP_FIGS].size(),Preproc_exec_params[DISPL_ADV_PREP_FIGS])==0)
			{
				str=str.substr(Preproc_exec_params[DISPL_ADV_PREP_FIGS].size());
				remove_spaces_ends(str);
				displ_adv_prep_figs=false;
				if (str.size())
				{
					cout<<Preproc_exec_params[DISPL_ADV_PREP_FIGS]<<" "<<str<<endl;
					if (str.compare("yes")==0) displ_adv_prep_figs=true;
				}
			}
			else if (str.compare(0,Preproc_exec_params[PRINT_OTHER_PARAMS].size(),Preproc_exec_params[PRINT_OTHER_PARAMS])==0)
			{
				str=str.substr(Preproc_exec_params[PRINT_OTHER_PARAMS].size());
				remove_spaces_ends(str);
				print_other_params=false;
				if (str.size())
				{
					cout<<Preproc_exec_params[PRINT_OTHER_PARAMS]<<" "<<str<<endl;
					if (str.compare("yes")==0) print_other_params=true;
				}
			}
			else if (str.compare(0,Output_files_params[OUTPUT_DIR].size(),Output_files_params[OUTPUT_DIR])==0)
			{
				str=str.substr(Output_files_params[OUTPUT_DIR].size());
				remove_spaces_ends(str);
				if (output_dir_in.compare(str))	initialize_maxent=true;
				output_dir_in=str;
				if (output_dir_in.size())
				{
					output_dir=output_dir_in;
					cout<<Output_files_params[OUTPUT_DIR]<<" "<<output_dir<<endl;
				}
			}
			else if (str.compare(0,Output_files_params[OUTPUT_NAME_SUFFIX].size(),Output_files_params[OUTPUT_NAME_SUFFIX])==0)
			{
				str=str.substr(Output_files_params[OUTPUT_NAME_SUFFIX].size());
				remove_spaces_ends(str);
				output_name_suffix=str;
				if (output_name_suffix.size())
					cout<<Output_files_params[OUTPUT_NAME_SUFFIX]<<" "<<output_name_suffix<<endl;
			}
			else if (str.compare(0,Output_files_params[ALPHA_SAVE_MAX].size(),Output_files_params[ALPHA_SAVE_MAX])==0)
			{
				str=str.substr(Output_files_params[ALPHA_SAVE_MAX].size());
				remove_spaces_ends(str);
				alpha_save_max_in=str;
				if (str.size())
				{
					alpha_save_max=stod(str);
					cout<<Output_files_params[ALPHA_SAVE_MAX]<<" "<<alpha_save_max<<endl;
				}
			}
			else if (str.compare(0,Output_files_params[ALPHA_SAVE_MIN].size(),Output_files_params[ALPHA_SAVE_MIN])==0)
			{
				str=str.substr(Output_files_params[ALPHA_SAVE_MIN].size());
				remove_spaces_ends(str);
				alpha_save_min_in=str;
				if (str.size())
				{
					alpha_save_min=stod(str);
					cout<<Output_files_params[ALPHA_SAVE_MIN]<<" "<<alpha_save_min<<endl;
				}
			}
			else if (str.compare(0,Output_files_params[W_SAMPLE].size(),Output_files_params[W_SAMPLE])==0)
			{
				str=str.substr(Output_files_params[W_SAMPLE].size());
				remove_spaces_ends(str);
				if (w_sample_in.compare(str))	initialize_maxent=true;
				w_sample_in=str;
				if (w_sample_in.size())
				{
					cout<<Output_files_params[W_SAMPLE]<<" "<<w_sample_in<<endl;
					w_sample=rowvec(w_sample_in);
					if (!w_sample.n_cols)
					{
						w_sample.reset();
						w_sample_in.clear();
					}
				}
			}
			else if (str.compare(0,Optim_comp_params[ALPHA_INIT].size(),Optim_comp_params[ALPHA_INIT])==0)
			{
				str=str.substr(Optim_comp_params[ALPHA_INIT].size());
				remove_spaces_ends(str);
				if (alpha_init_in.compare(str))	initialize_maxent=true;
				alpha_init_in=str;
				if (alpha_init_in.size())
				{
					alpha0=stod(alpha_init_in);
					cout<<Optim_comp_params[ALPHA_INIT]<<" "<<alpha_init_in<<endl;
				}
			}
			else if (str.compare(0,Optim_comp_params[ALPHA_MIN].size(),Optim_comp_params[ALPHA_MIN])==0)
			{
				str=str.substr(Optim_comp_params[ALPHA_MIN].size());
				remove_spaces_ends(str);
				alpha_min_in=str;
				if (str.size())
				{
					alpha_min=stod(str);
					cout<<Optim_comp_params[ALPHA_MIN]<<" "<<str<<endl;
				}
			}
			else if (str.compare(0,Optim_comp_params[ALPHA_OPT_MAX].size(),Optim_comp_params[ALPHA_OPT_MAX])==0)
			{
				str=str.substr(Optim_comp_params[ALPHA_OPT_MAX].size());
				remove_spaces_ends(str);
				alpha_opt_max_in=str;
				if (str.size())
				{
					alpha_opt_max=stod(str);
					cout<<Optim_comp_params[ALPHA_OPT_MAX]<<" "<<str<<endl;
				}
			}
			else if (str.compare(0,Optim_comp_params[ALPHA_OPT_MIN].size(),Optim_comp_params[ALPHA_OPT_MIN])==0)
			{
				str=str.substr(Optim_comp_params[ALPHA_OPT_MIN].size());
				remove_spaces_ends(str);
				alpha_opt_min_in=str;
				if (str.size())
				{
					alpha_opt_min=stod(str);
					cout<<Optim_comp_params[ALPHA_OPT_MIN]<<" "<<str<<endl;
				}
			}
			else if (str.compare(0,Optim_exec_params[N_ALPHA].size(),Optim_exec_params[N_ALPHA])==0)
			{
				str=str.substr(Optim_exec_params[N_ALPHA].size());
				remove_spaces_ends(str);
				Nalpha_in=str;
				if (str.size())
				{
					Nalpha=stoi(str);
					cout<<Optim_exec_params[N_ALPHA]<<" "<<str<<endl;
				}
			}
			else if (str.compare(0,Optim_exec_params[INITIALIZE_MAXENT].size(),Optim_exec_params[INITIALIZE_MAXENT])==0)
			{
				str=str.substr(Optim_exec_params[INITIALIZE_MAXENT].size());
				remove_spaces_ends(str);
				if (str.size())
				{
					cout<<Optim_exec_params[INITIALIZE_MAXENT]<<" "<<str<<endl;
					if (str.compare("yes")==0) initialize_maxent=true;
				}
			}
			else if (str.compare(0,Optim_exec_params[INITIALIZE_PREPROC].size(),Optim_exec_params[INITIALIZE_PREPROC])==0)
			{
				str=str.substr(Optim_exec_params[INITIALIZE_PREPROC].size());
				remove_spaces_ends(str);
				if (str.size())
				{
					cout<<Optim_exec_params[INITIALIZE_PREPROC]<<" "<<str<<endl;
					if (str.compare("yes")==0) initialize=true;
				}
			}
			else if (str.compare(0,Optim_exec_params[INTERACTIVE_MODE].size(),Optim_exec_params[INTERACTIVE_MODE])==0)
			{
				str=str.substr(Optim_exec_params[INTERACTIVE_MODE].size());
				remove_spaces_ends(str);
				interactive_mode=true;
				if (str.size())
				{
					cout<<Optim_exec_params[INTERACTIVE_MODE]<<" "<<str<<endl;
					if (str.compare("no")==0)
					{
						interactive_mode=false;
						graph_2D::display_figures=false;
					}
				}
			}
			else if (str.compare(0,Optim_displ_params[PRINT_ALPHA].size(),Optim_displ_params[PRINT_ALPHA])==0)
			{
				str=str.substr(Optim_displ_params[PRINT_ALPHA].size());
				remove_spaces_ends(str);
				print_alpha=false;
				if (str.size())
				{
					cout<<Optim_displ_params[PRINT_ALPHA]<<" "<<str<<endl;
					if (str.compare("yes")==0) print_alpha=true;
				}
			}
			else if (str.compare(0,Optim_displ_params[SHOW_OPTIMAL_ALPHA_FIGS].size(),Optim_displ_params[SHOW_OPTIMAL_ALPHA_FIGS])==0)
			{
				str=str.substr(Optim_displ_params[SHOW_OPTIMAL_ALPHA_FIGS].size());
				remove_spaces_ends(str);
				show_optimal_alpha_figs=true;
				if (str.size())
				{
					cout<<Optim_displ_params[SHOW_OPTIMAL_ALPHA_FIGS]<<" "<<str<<endl;
					if (str.compare("no")==0) show_optimal_alpha_figs=false;
				}
			}
			else if (str.compare(0,Optim_displ_params[SHOW_LOWEST_ALPHA_FIGS].size(),Optim_displ_params[SHOW_LOWEST_ALPHA_FIGS])==0)
			{
				str=str.substr(Optim_displ_params[SHOW_LOWEST_ALPHA_FIGS].size());
				remove_spaces_ends(str);
				show_lowest_alpha_figs=true;
				if (str.size())
				{
					cout<<Optim_displ_params[SHOW_LOWEST_ALPHA_FIGS]<<" "<<str<<endl;
					if (str.compare("no")==0) show_lowest_alpha_figs=false;
				}
			}
			else if (str.compare(0,Optim_displ_params[SHOW_ALPHA_CURVES].size(),Optim_displ_params[SHOW_ALPHA_CURVES])==0)
			{
				str=str.substr(Optim_displ_params[SHOW_ALPHA_CURVES].size());
				remove_spaces_ends(str);
				show_alpha_curves=true;
				if (str.size())
				{
					cout<<Optim_displ_params[SHOW_ALPHA_CURVES]<<" "<<str<<endl;
					if (str.compare("no")==0) show_alpha_curves=false;
				}
			}
			else if (str.compare(0,Optim_displ_params[REF_SPECTR_FILE].size(),Optim_displ_params[REF_SPECTR_FILE])==0)
			{
				str=str.substr(Optim_displ_params[REF_SPECTR_FILE].size());
				remove_spaces_ends(str);
				if (A_ref_file_in.compare(str))	A_ref_change=true;
				A_ref_file_in=str;
				if (str.size())
				{
					cout<<Optim_displ_params[REF_SPECTR_FILE]<<" "<<str<<endl;
					if (load_data_file(Aref_data,str))
					{
						cout<<Optim_displ_params[REF_SPECTR_FILE].substr(0,Optim_displ_params[REF_SPECTR_FILE].size()-1)<<" loaded\n";
						A_ref_file=A_ref_file_in;
					}
					else
						A_ref_file.clear();
				}
				else
					A_ref_file.clear();
			}
			getline(file,str);
		}
		file.close();
		cout<<endl;
	}
	else
	{
		cout<<"Input parameters file not found. Creating default one.\n";
		create_default_input_params_file();
		return false;
	}
	
	return true;
}

bool OmegaMaxEnt_data::load_other_params()
{
    ifstream file(other_params_file_name);
    
    if (file)
    {
        string str;
		
		int j=0;
		
        if (!print_other_params)
            cout<<"\nOTHER PARAMETERS (different from default ones):\n\n";
        else
            cout<<"\nOTHER PARAMETERS:\n\n";
        
        getline(file,str);
        while (!file.eof())
        {
			if (str.compare(0,Other_params_int[NN_MIN].size(),Other_params_int[NN_MIN])==0)
			{
				str=str.substr(Other_params_int[NN_MIN].size());
				Nn_min=stoi(str);
				if (Nn_min!=Other_params_int_default_values[NN_MIN] || print_other_params)
					cout<<Other_params_int[NN_MIN]<<" "<<Nn_min<<endl;
				j++;
			}
            else if (str.compare(0,Other_params_int[NN_MAX].size(),Other_params_int[NN_MAX])==0)
            {
                str=str.substr(Other_params_int[NN_MAX].size());
                Nn_max=stoi(str);
                if (Nn_max!=Other_params_int_default_values[NN_MAX] || print_other_params)
                    cout<<Other_params_int[NN_MAX]<<" "<<Nn_max<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_int[NW_MIN].size(),Other_params_int[NW_MIN])==0)
            {
                str=str.substr(Other_params_int[NW_MIN].size());
                Nw_min=stoi(str);
                if (Nw_min!=Other_params_int_default_values[NW_MIN] || print_other_params)
                    cout<<Other_params_int[NW_MIN]<<" "<<Nw_min<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_int[NW_MAX].size(),Other_params_int[NW_MAX])==0)
            {
                str=str.substr(Other_params_int[NW_MAX].size());
                Nw_max=stoi(str);
                if (Nw_max!=Other_params_int_default_values[NW_MAX] || print_other_params)
                    cout<<Other_params_int[NW_MAX]<<" "<<Nw_max<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_int[NN_FIT_MAX].size(),Other_params_int[NN_FIT_MAX])==0)
            {
                str=str.substr(Other_params_int[NN_FIT_MAX].size());
                Nn_fit_max=stoi(str);
                if (Nn_fit_max!=Other_params_int_default_values[NN_FIT_MAX] || print_other_params)
                    cout<<Other_params_int[NN_FIT_MAX]<<" "<<Nn_fit_max<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_int[NN_FIT_FIN].size(),Other_params_int[NN_FIT_FIN])==0)
            {
                str=str.substr(Other_params_int[NN_FIT_FIN].size());
                Nn_fit_fin=stoi(str);
                if (Nn_fit_fin!=Other_params_int_default_values[NN_FIT_FIN] || print_other_params)
                    cout<<Other_params_int[NN_FIT_FIN]<<" "<<Nn_fit_fin<<endl;
				j++;
            }
			else if (str.compare(0,Other_params_int[NN_AS_MIN].size(),Other_params_int[NN_AS_MIN])==0)
			{
				str=str.substr(Other_params_int[NN_AS_MIN].size());
				Nn_as_min=stoi(str);
				if (Nn_as_min!=Other_params_int_default_values[NN_AS_MIN] || print_other_params)
					cout<<Other_params_int[NN_AS_MIN]<<" "<<Nn_as_min<<endl;
				j++;
			}
			/*
            else if (str.compare(0,Other_params_int[NWN_TEST_METAL].size(),Other_params_int[NWN_TEST_METAL])==0)
            {
                str=str.substr(Other_params_int[NWN_TEST_METAL].size());
                Nwn_test_metal=stoi(str);
                if (Nwn_test_metal!=Other_params_int_default_values[NWN_TEST_METAL] || print_other_params)
                    cout<<Other_params_int[NWN_TEST_METAL]<<" "<<Nwn_test_metal<<endl;
            }
			 */
            else if (str.compare(0,Other_params_int[N_ITER_DA_MAX].size(),Other_params_int[N_ITER_DA_MAX])==0)
            {
                str=str.substr(Other_params_int[N_ITER_DA_MAX].size());
                Niter_dA_max=stoi(str);
                if (Niter_dA_max!=Other_params_int_default_values[N_ITER_DA_MAX] || print_other_params)
                    cout<<Other_params_int[N_ITER_DA_MAX]<<" "<<Niter_dA_max<<endl;
				j++;
            }
			/*
            else if (str.compare(0,Other_params_int[N_ALPHA_MAX_FIGS].size(),Other_params_int[N_ALPHA_MAX_FIGS])==0)
            {
                str=str.substr(Other_params_int[N_ALPHA_MAX_FIGS].size());
                Nalpha_max_figs=stoi(str);
                if (Nalpha_max_figs!=Other_params_int_default_values[N_ALPHA_MAX_FIGS] || print_other_params)
                    cout<<Other_params_int[N_ALPHA_MAX_FIGS]<<" "<<Nalpha_max_figs<<endl;
            }
			 */
            else if (str.compare(0,Other_params_int[NW_SAMP].size(),Other_params_int[NW_SAMP])==0)
            {
                str=str.substr(Other_params_int[NW_SAMP].size());
                Nwsamp=stoi(str);
                if (Nwsamp!=Other_params_int_default_values[NW_SAMP] || print_other_params)
                    cout<<Other_params_int[NW_SAMP]<<" "<<Nwsamp<<endl;
				j++;
            }
			else if (str.compare(0,Other_params_int[NSMOOTH_ERRG].size(),Other_params_int[NSMOOTH_ERRG])==0)
			{
				str=str.substr(Other_params_int[NSMOOTH_ERRG].size());
				Nsmooth_errG=stoi(str);
				if (Nsmooth_errG!=Other_params_int_default_values[NSMOOTH_ERRG] || print_other_params)
					cout<<Other_params_int[NSMOOTH_ERRG]<<" "<<Nsmooth_errG<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[F_SW_STD_OMEGA].size(),Other_params_fl[F_SW_STD_OMEGA])==0)
			{
				str=str.substr(Other_params_fl[F_SW_STD_OMEGA].size());
				f_SW_std_omega=stod(str);
				if (f_SW_std_omega!=Other_params_fl_default_values[F_SW_STD_OMEGA] || print_other_params)
					cout<<Other_params_fl[F_SW_STD_OMEGA]<<" "<<f_SW_std_omega<<endl;
				j++;
			}
            else if (str.compare(0,Other_params_fl[F_W_RANGE].size(),Other_params_fl[F_W_RANGE])==0)
            {
                str=str.substr(Other_params_fl[F_W_RANGE].size());
                f_w_range=stod(str);
                if (f_w_range!=Other_params_fl_default_values[F_W_RANGE] || print_other_params)
                    cout<<Other_params_fl[F_W_RANGE]<<" "<<f_w_range<<endl;
				j++;
            }
			/*
			else if (str.compare(0,Other_params_fl[F_WIDTH_GRID_DENS].size(),Other_params_fl[F_WIDTH_GRID_DENS])==0)
			{
				str=str.substr(Other_params_fl[F_WIDTH_GRID_DENS].size());
				f_width_grid_dens=stod(str);
				if (f_width_grid_dens!=Other_params_fl_default_values[F_WIDTH_GRID_DENS] || print_other_params)
					cout<<Other_params_fl[F_WIDTH_GRID_DENS]<<" "<<f_width_grid_dens<<endl;
			}
			 */
			else if (str.compare(0,Other_params_fl[RMIN_SW_DW].size(),Other_params_fl[RMIN_SW_DW])==0)
			{
				str=str.substr(Other_params_fl[RMIN_SW_DW].size());
				Rmin_SW_dw=stod(str);
				if (Rmin_SW_dw!=Other_params_fl_default_values[RMIN_SW_DW] || print_other_params)
					cout<<Other_params_fl[RMIN_SW_DW]<<" "<<Rmin_SW_dw<<endl;
				j++;
			}
            else if (str.compare(0,Other_params_fl[TOL_TEM].size(),Other_params_fl[TOL_TEM])==0)
            {
                str=str.substr(Other_params_fl[TOL_TEM].size());
                tol_tem=stod(str);
                if (tol_tem!=Other_params_fl_default_values[TOL_TEM] || print_other_params)
                    cout<<Other_params_fl[TOL_TEM]<<" "<<tol_tem<<endl;
				j++;
            }
			else if (str.compare(0,Other_params_fl[TOL_GINF].size(),Other_params_fl[TOL_GINF])==0)
			{
				str=str.substr(Other_params_fl[TOL_GINF].size());
				tol_G_inf=stod(str);
				if (tol_G_inf!=Other_params_fl_default_values[TOL_GINF] || print_other_params)
					cout<<Other_params_fl[TOL_GINF]<<" "<<tol_G_inf<<endl;
				j++;
			}
            else if (str.compare(0,Other_params_fl[TOL_NORM].size(),Other_params_fl[TOL_NORM])==0)
            {
                str=str.substr(Other_params_fl[TOL_NORM].size());
                tol_norm=stod(str);
                if (tol_norm!=Other_params_fl_default_values[TOL_NORM] || print_other_params)
                    cout<<Other_params_fl[TOL_NORM]<<" "<<tol_norm<<endl;
				j++;
            }
			/*
            else if (str.compare(0,Other_params_fl[TOL_R_G0_BETA].size(),Other_params_fl[TOL_R_G0_BETA])==0)
            {
                str=str.substr(Other_params_fl[TOL_R_G0_BETA].size());
                tol_R_G0_Gbeta=stod(str);
                if (tol_R_G0_Gbeta!=Other_params_fl_default_values[TOL_R_G0_BETA] || print_other_params)
                    cout<<Other_params_fl[TOL_R_G0_BETA]<<" "<<tol_R_G0_Gbeta<<endl;
            }
			 */
            else if (str.compare(0,Other_params_fl[TOL_M1].size(),Other_params_fl[TOL_M1])==0)
            {
                str=str.substr(Other_params_fl[TOL_M1].size());
                tol_M1=stod(str);
                if (tol_M1!=Other_params_fl_default_values[TOL_M1] || print_other_params)
                    cout<<Other_params_fl[TOL_M1]<<" "<<tol_M1<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[TOL_M2].size(),Other_params_fl[TOL_M2])==0)
            {
                str=str.substr(Other_params_fl[TOL_M2].size());
                tol_M2=stod(str);
                if (tol_M2!=Other_params_fl_default_values[TOL_M2] || print_other_params)
                    cout<<Other_params_fl[TOL_M2]<<" "<<tol_M2<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[TOL_M3].size(),Other_params_fl[TOL_M3])==0)
            {
                str=str.substr(Other_params_fl[TOL_M3].size());
                tol_M3=stod(str);
                if (tol_M3!=Other_params_fl_default_values[TOL_M3] || print_other_params)
                    cout<<Other_params_fl[TOL_M3]<<" "<<tol_M3<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[DEFAULT_ERROR_G].size(),Other_params_fl[DEFAULT_ERROR_G])==0)
            {
                str=str.substr(Other_params_fl[DEFAULT_ERROR_G].size());
                default_error_G=stod(str);
                if (default_error_G!=Other_params_fl_default_values[DEFAULT_ERROR_G] || print_other_params)
                    cout<<Other_params_fl[DEFAULT_ERROR_G]<<" "<<default_error_G<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[ERR_NORM].size(),Other_params_fl[ERR_NORM])==0)
            {
                str=str.substr(Other_params_fl[ERR_NORM].size());
                err_norm=stod(str);
                if (err_norm!=Other_params_fl_default_values[ERR_NORM] || print_other_params)
                    cout<<Other_params_fl[ERR_NORM]<<" "<<err_norm<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[DEFAULT_ERROR_M].size(),Other_params_fl[DEFAULT_ERROR_M])==0)
            {
                str=str.substr(Other_params_fl[DEFAULT_ERROR_M].size());
                default_error_M=stod(str);
                if (default_error_M!=Other_params_fl_default_values[DEFAULT_ERROR_M] || print_other_params)
                    cout<<Other_params_fl[DEFAULT_ERROR_M]<<" "<<default_error_M<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[TOL_MEAN_C1].size(),Other_params_fl[TOL_MEAN_C1])==0)
            {
                str=str.substr(Other_params_fl[TOL_MEAN_C1].size());
                tol_mean_C1=stod(str);
                if (tol_mean_C1!=Other_params_fl_default_values[TOL_MEAN_C1] || print_other_params)
                    cout<<Other_params_fl[TOL_MEAN_C1]<<" "<<tol_mean_C1<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[TOL_STD_C1].size(),Other_params_fl[TOL_STD_C1])==0)
            {
                str=str.substr(Other_params_fl[TOL_STD_C1].size());
                tol_std_C1=stod(str);
                if (tol_std_C1!=Other_params_fl_default_values[TOL_STD_C1] || print_other_params)
                    cout<<Other_params_fl[TOL_STD_C1]<<" "<<tol_std_C1<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[TOL_RDW].size(),Other_params_fl[TOL_RDW])==0)
            {
                str=str.substr(Other_params_fl[TOL_RDW].size());
                tol_rdw=stod(str);
                if (tol_rdw!=Other_params_fl_default_values[TOL_RDW] || print_other_params)
                    cout<<Other_params_fl[TOL_RDW]<<" "<<tol_rdw<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[RMIN_DW_DW].size(),Other_params_fl[RMIN_DW_DW])==0)
            {
                str=str.substr(Other_params_fl[RMIN_DW_DW].size());
                Rmin_Dw_dw=stod(str);
                if (Rmin_Dw_dw!=Other_params_fl_default_values[RMIN_DW_DW] || print_other_params)
                    cout<<Other_params_fl[RMIN_DW_DW]<<" "<<Rmin_Dw_dw<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[RDW_MAX].size(),Other_params_fl[RDW_MAX])==0)
            {
                str=str.substr(Other_params_fl[RDW_MAX].size());
                Rdw_max=stod(str);
                if (Rdw_max!=Other_params_fl_default_values[RDW_MAX] || print_other_params)
                    cout<<Other_params_fl[RDW_MAX]<<" "<<Rdw_max<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[RW_GRID].size(),Other_params_fl[RW_GRID])==0)
            {
                str=str.substr(Other_params_fl[RW_GRID].size());
                RW_grid=stod(str);
                if (RW_grid!=Other_params_fl_default_values[RW_GRID] || print_other_params)
                    cout<<Other_params_fl[RW_GRID]<<" "<<RW_grid<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[RWD_GRID].size(),Other_params_fl[RWD_GRID])==0)
            {
                str=str.substr(Other_params_fl[RWD_GRID].size());
                RWD_grid=stod(str);
                if (RWD_grid!=Other_params_fl_default_values[RWD_GRID] || print_other_params)
                    cout<<Other_params_fl[RWD_GRID]<<" "<<RWD_grid<<endl;
				j++;
            }
			/*
            else if (str.compare(0,Other_params_fl[TOL_QUAD].size(),Other_params_fl[TOL_QUAD])==0)
            {
                str=str.substr(Other_params_fl[TOL_QUAD].size());
                tol_quad=stod(str);
                if (tol_quad!=Other_params_fl_default_values[TOL_QUAD] || print_other_params)
                    cout<<Other_params_fl[TOL_QUAD]<<" "<<tol_quad<<endl;
            }*/
            else if (str.compare(0,Other_params_fl[MIN_DEF_M].size(),Other_params_fl[MIN_DEF_M])==0)
            {
                str=str.substr(Other_params_fl[MIN_DEF_M].size());
                minDefM=stod(str);
                if (minDefM!=Other_params_fl_default_values[MIN_DEF_M] || print_other_params)
                    cout<<Other_params_fl[MIN_DEF_M]<<" "<<minDefM<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[F_ALPHA_INIT].size(),Other_params_fl[F_ALPHA_INIT])==0)
            {
                str=str.substr(Other_params_fl[F_ALPHA_INIT].size());
                f_alpha_init=stod(str);
                if (f_alpha_init!=Other_params_fl_default_values[F_ALPHA_INIT] || print_other_params)
                    cout<<Other_params_fl[F_ALPHA_INIT]<<" "<<f_alpha_init<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[R_WIDTH_ASMIN].size(),Other_params_fl[R_WIDTH_ASMIN])==0)
            {
                str=str.substr(Other_params_fl[R_WIDTH_ASMIN].size());
                R_width_ASmin=stod(str);
                if (R_width_ASmin!=Other_params_fl_default_values[R_WIDTH_ASMIN] || print_other_params)
                    cout<<Other_params_fl[R_WIDTH_ASMIN]<<" "<<R_width_ASmin<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[F_SMIN].size(),Other_params_fl[F_SMIN])==0)
            {
                str=str.substr(Other_params_fl[F_SMIN].size());
                f_Smin=stod(str);
                if (f_Smin!=Other_params_fl_default_values[F_SMIN] || print_other_params)
                    cout<<Other_params_fl[F_SMIN]<<" "<<f_Smin<<endl;
				j++;
            }
			/*
            else if (str.compare(0,Other_params_fl[F_CHI2_SAVE].size(),Other_params_fl[F_CHI2_SAVE])==0)
            {
                str=str.substr(Other_params_fl[F_CHI2_SAVE].size());
                f_chi2save=stod(str);
                if (f_chi2save!=Other_params_fl_default_values[F_CHI2_SAVE] || print_other_params)
                    cout<<Other_params_fl[F_CHI2_SAVE]<<" "<<f_chi2save<<endl;
            }*/
            else if (str.compare(0,Other_params_fl[R_CHI2_MIN].size(),Other_params_fl[R_CHI2_MIN])==0)
            {
                str=str.substr(Other_params_fl[R_CHI2_MIN].size());
                R_chi2_min=stod(str);
                if (R_chi2_min!=Other_params_fl_default_values[R_CHI2_MIN] || print_other_params)
                    cout<<Other_params_fl[R_CHI2_MIN]<<" "<<R_chi2_min<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[TOL_INT_DA].size(),Other_params_fl[TOL_INT_DA])==0)
            {
                str=str.substr(Other_params_fl[TOL_INT_DA].size());
                tol_int_dA=stod(str);
                if (tol_int_dA!=Other_params_fl_default_values[TOL_INT_DA] || print_other_params)
                    cout<<Other_params_fl[TOL_INT_DA]<<" "<<tol_int_dA<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[R_C2_H].size(),Other_params_fl[R_C2_H])==0)
            {
                str=str.substr(Other_params_fl[R_C2_H].size());
                rc2H=stod(str);
                if (rc2H!=Other_params_fl_default_values[R_C2_H] || print_other_params)
                    cout<<Other_params_fl[R_C2_H]<<" "<<rc2H<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[POW_ALPHA_STEP_INIT].size(),Other_params_fl[POW_ALPHA_STEP_INIT])==0)
            {
                str=str.substr(Other_params_fl[POW_ALPHA_STEP_INIT].size());
                pow_alpha_step_init=stod(str);
                if (pow_alpha_step_init!=Other_params_fl_default_values[POW_ALPHA_STEP_INIT] || print_other_params)
                    cout<<Other_params_fl[POW_ALPHA_STEP_INIT]<<" "<<pow_alpha_step_init<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[POW_ALPHA_STEP_MIN].size(),Other_params_fl[POW_ALPHA_STEP_MIN])==0)
            {
                str=str.substr(Other_params_fl[POW_ALPHA_STEP_MIN].size());
                pow_alpha_step_min=stod(str);
                if (pow_alpha_step_min!=Other_params_fl_default_values[POW_ALPHA_STEP_MIN] || print_other_params)
                    cout<<Other_params_fl[POW_ALPHA_STEP_MIN]<<" "<<pow_alpha_step_min<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[CHI2_ALPHA_SMOOTH_RANGE].size(),Other_params_fl[CHI2_ALPHA_SMOOTH_RANGE])==0)
            {
                str=str.substr(Other_params_fl[CHI2_ALPHA_SMOOTH_RANGE].size());
                chi2_alpha_smooth_range=stod(str);
                if (chi2_alpha_smooth_range!=Other_params_fl_default_values[CHI2_ALPHA_SMOOTH_RANGE] || print_other_params)
                    cout<<Other_params_fl[CHI2_ALPHA_SMOOTH_RANGE]<<" "<<chi2_alpha_smooth_range<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[F_SCALE_LALPHA_LCHI2].size(),Other_params_fl[F_SCALE_LALPHA_LCHI2])==0)
            {
                str=str.substr(Other_params_fl[F_SCALE_LALPHA_LCHI2].size());
                f_scale_lalpha_lchi2=stod(str);
                if (f_scale_lalpha_lchi2!=Other_params_fl_default_values[F_SCALE_LALPHA_LCHI2] || print_other_params)
                    cout<<Other_params_fl[F_SCALE_LALPHA_LCHI2]<<" "<<f_scale_lalpha_lchi2<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[FN_FIT_TAU_W].size(),Other_params_fl[FN_FIT_TAU_W])==0)
            {
                str=str.substr(Other_params_fl[FN_FIT_TAU_W].size());
                FNfitTauW=stod(str);
                if (FNfitTauW!=Other_params_fl_default_values[FN_FIT_TAU_W] || print_other_params)
                    cout<<Other_params_fl[FN_FIT_TAU_W]<<" "<<FNfitTauW<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[STD_NORM_PEAK_MAX].size(),Other_params_fl[STD_NORM_PEAK_MAX])==0)
            {
                str=str.substr(Other_params_fl[STD_NORM_PEAK_MAX].size());
                std_norm_peak_max=stod(str);
                if (std_norm_peak_max!=Other_params_fl_default_values[STD_NORM_PEAK_MAX] || print_other_params)
                    cout<<Other_params_fl[STD_NORM_PEAK_MAX]<<" "<<std_norm_peak_max<<endl;
				j++;
            }
            else if (str.compare(0,Other_params_fl[VAR_M2_PEAK_MAX].size(),Other_params_fl[VAR_M2_PEAK_MAX])==0)
            {
                str=str.substr(Other_params_fl[VAR_M2_PEAK_MAX].size());
                varM2_peak_max=stod(str);
                if (varM2_peak_max!=Other_params_fl_default_values[VAR_M2_PEAK_MAX] || print_other_params)
                    cout<<Other_params_fl[VAR_M2_PEAK_MAX]<<" "<<varM2_peak_max<<endl;
				j++;
            }
			else if (str.compare(0,Other_params_fl[PEAK_WEIGHT_MIN].size(),Other_params_fl[PEAK_WEIGHT_MIN])==0)
			{
				str=str.substr(Other_params_fl[PEAK_WEIGHT_MIN].size());
				peak_weight_min=stod(str);
				if (peak_weight_min!=Other_params_fl_default_values[PEAK_WEIGHT_MIN] || print_other_params)
					cout<<Other_params_fl[PEAK_WEIGHT_MIN]<<" "<<peak_weight_min<<endl;
				j++;
			}
			/*
            else if (str.compare(0,Other_params_fl[R_D2G_CHI_PEAK].size(),Other_params_fl[R_D2G_CHI_PEAK])==0)
            {
                str=str.substr(Other_params_fl[R_D2G_CHI_PEAK].size());
                R_d2G_chi_peak=stod(str);
                if (R_d2G_chi_peak!=Other_params_fl_default_values[R_D2G_CHI_PEAK] || print_other_params)
                    cout<<Other_params_fl[R_D2G_CHI_PEAK]<<" "<<R_d2G_chi_peak<<endl;
            }
			 */
			else if (str.compare(0,Other_params_fl[RMAX_DLCHI2_LALPHA].size(),Other_params_fl[RMAX_DLCHI2_LALPHA])==0)
			{
				str=str.substr(Other_params_fl[RMAX_DLCHI2_LALPHA].size());
				RMAX_dlchi2_lalpha=stod(str);
				if (RMAX_dlchi2_lalpha!=Other_params_fl_default_values[RMAX_DLCHI2_LALPHA] || print_other_params)
					cout<<Other_params_fl[RMAX_DLCHI2_LALPHA]<<" "<<RMAX_dlchi2_lalpha<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[F_ALPHA_MIN].size(),Other_params_fl[F_ALPHA_MIN])==0)
			{
				str=str.substr(Other_params_fl[F_ALPHA_MIN].size());
				f_alpha_min=stod(str);
				if (f_alpha_min!=Other_params_fl_default_values[F_ALPHA_MIN] || print_other_params)
					cout<<Other_params_fl[F_ALPHA_MIN]<<" "<<f_alpha_min<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[SAVE_ALPHA_RANGE].size(),Other_params_fl[SAVE_ALPHA_RANGE])==0)
			{
				str=str.substr(Other_params_fl[SAVE_ALPHA_RANGE].size());
				save_alpha_range=stod(str);
				if (save_alpha_range!=Other_params_fl_default_values[SAVE_ALPHA_RANGE] || print_other_params)
					cout<<Other_params_fl[SAVE_ALPHA_RANGE]<<" "<<save_alpha_range<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_PEAK_WIDTH_DW].size(),Other_params_fl[R_PEAK_WIDTH_DW])==0)
			{
				str=str.substr(Other_params_fl[R_PEAK_WIDTH_DW].size());
				R_peak_width_dw=stod(str);
				if (R_peak_width_dw!=Other_params_fl_default_values[R_PEAK_WIDTH_DW] || print_other_params)
					cout<<Other_params_fl[R_PEAK_WIDTH_DW]<<" "<<R_peak_width_dw<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_WNCUTOFF_WR].size(),Other_params_fl[R_WNCUTOFF_WR])==0)
			{
				str=str.substr(Other_params_fl[R_WNCUTOFF_WR].size());
				R_wncutoff_wr=stod(str);
				if (R_wncutoff_wr!=Other_params_fl_default_values[R_WNCUTOFF_WR] || print_other_params)
					cout<<Other_params_fl[R_WNCUTOFF_WR]<<" "<<R_wncutoff_wr<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_DW_DW].size(),Other_params_fl[R_DW_DW])==0)
			{
				str=str.substr(Other_params_fl[R_DW_DW].size());
				R_Dw_dw=stod(str);
				if (R_Dw_dw!=Other_params_fl_default_values[R_DW_DW] || print_other_params)
					cout<<Other_params_fl[R_DW_DW]<<" "<<R_Dw_dw<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_SW_WR].size(),Other_params_fl[R_SW_WR])==0)
			{
				str=str.substr(Other_params_fl[R_SW_WR].size());
				R_SW_wr=stod(str);
				if (R_SW_wr!=Other_params_fl_default_values[R_SW_WR] || print_other_params)
					cout<<Other_params_fl[R_SW_WR]<<" "<<R_SW_wr<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_WMAX_WR_MIN].size(),Other_params_fl[R_WMAX_WR_MIN])==0)
			{
				str=str.substr(Other_params_fl[R_WMAX_WR_MIN].size());
				R_wmax_wr_min=stod(str);
				if (R_wmax_wr_min!=Other_params_fl_default_values[R_WMAX_WR_MIN] || print_other_params)
					cout<<Other_params_fl[R_WMAX_WR_MIN]<<" "<<R_wmax_wr_min<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[WGT_MIN_SM].size(),Other_params_fl[WGT_MIN_SM])==0)
			{
				str=str.substr(Other_params_fl[WGT_MIN_SM].size());
				wgt_min_sm=stod(str);
				if (wgt_min_sm!=Other_params_fl_default_values[WGT_MIN_SM] || print_other_params)
					cout<<Other_params_fl[WGT_MIN_SM]<<" "<<wgt_min_sm<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_SW_G_RE_W_RANGE].size(),Other_params_fl[R_SW_G_RE_W_RANGE])==0)
			{
				str=str.substr(Other_params_fl[R_SW_G_RE_W_RANGE].size());
				R_SW_G_Re_w_range=stod(str);
				if (R_SW_G_Re_w_range!=Other_params_fl_default_values[R_SW_G_RE_W_RANGE] || print_other_params)
					cout<<Other_params_fl[R_SW_G_RE_W_RANGE]<<" "<<R_SW_G_Re_w_range<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_DW_MIN_DW_DENSE].size(),Other_params_fl[R_DW_MIN_DW_DENSE])==0)
			{
				str=str.substr(Other_params_fl[R_DW_MIN_DW_DENSE].size());
				R_dw_min_dw_dense=stod(str);
				if (R_dw_min_dw_dense!=Other_params_fl_default_values[R_DW_MIN_DW_DENSE] || print_other_params)
					cout<<Other_params_fl[R_DW_MIN_DW_DENSE]<<" "<<R_dw_min_dw_dense<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_WKK_SW].size(),Other_params_fl[R_WKK_SW])==0)
			{
				str=str.substr(Other_params_fl[R_WKK_SW].size());
				R_wKK_SW=stod(str);
				if (R_wKK_SW!=Other_params_fl_default_values[R_WKK_SW] || print_other_params)
					cout<<Other_params_fl[R_WKK_SW]<<" "<<R_wKK_SW<<endl;
				j++;
			}
			else if (str.compare(0,Other_params_fl[R_SV_MIN].size(),Other_params_fl[R_SV_MIN])==0)
			{
				str=str.substr(Other_params_fl[R_SV_MIN].size());
				R_sv_min=stod(str);
				if (R_sv_min!=Other_params_fl_default_values[R_SV_MIN] || print_other_params)
					cout<<Other_params_fl[R_SV_MIN]<<" "<<R_sv_min<<endl;
				j++;
			}
			
            getline(file,str);
        }
        file.close();
		
		cout<<endl;
		
		int Ntmp=Other_params_int.size()+Other_params_fl.size();
		if (j!=Ntmp)
		{
			cout<<"load_other_params() error: number of parameters loaded not equal to the total number of parameters\n";
			cout<<"total number of parameters: "<<Ntmp<<endl;
			cout<<"number of parameters loaded: "<<j<<endl;
			
			return false;
		}
    }
    else
    {
        cout<<"Internal parameters file not found. Creating default one.\n";
		return create_default_other_params_file();
    }
	
    return true;
}

bool OmegaMaxEnt_data::create_default_other_params_file()
{
	initialize=true;
	
    ofstream file(other_params_file_name);
    if (file)
    {
        auto ptr_int=Other_params_int_default_values.cbegin();
        for (auto item: Other_params_int)
        {
            file<<item.second<<" "<<ptr_int->second<<'\n';
            ptr_int++;
        }
        file<<setprecision(2);
        auto ptr_fl=Other_params_fl_default_values.cbegin();
        for (auto item: Other_params_fl)
        {
            file<<item.second<<" "<<ptr_fl->second<<'\n';
            ptr_fl++;
        }
        file.close();
		
		cout<<"The file "<<other_params_file_name<<" was created.\n";
		
		return load_other_params();
    }
    else
    {
        cout<<"cannot create file "<<other_params_file_name<<'\n';
        return false;
    }
}

bool OmegaMaxEnt_data::create_default_input_params_file()
{
	initialize=true;
	
    ofstream file(input_params_file_name);
    if (file)
    {
        file<<"data file:\n\nOPTIONAL PREPROCESSING TIME PARAMETERS\n\nDATA PARAMETERS\n";
        for (auto item: Data_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\nINPUT FILES PARAMETERS\n";
        for (auto item: Input_files_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\nFREQUENCY GRID PARAMETERS\n";
        for (auto item: Grid_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\nCOMPUTATION OPTIONS\n";
        for (auto item: Preproc_comp_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\nPREPROCESSING EXECUTION OPTIONS\n";
        for (auto item: Preproc_exec_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\n\nOPTIONAL MINIMIZATION TIME PARAMETERS\n\nOUTPUT FILES PARAMETERS\n";
        for (auto item: Output_files_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\nCOMPUTATION PARAMETERS\n";
        for (auto item: Optim_comp_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\nMINIMIZATION EXECUTION OPTIONS\n";
        for (auto item: Optim_exec_params)
        {
            file<<item.second<<'\n';
        }
        file<<"\nDISPLAY OPTIONS\n";
        for (auto item: Optim_displ_params)
        {
            file<<item.second<<'\n';
        }
        file.close();
		
		cout<<"The file "<<input_params_file_name<<" was created.\n";
		cout<<"Fill it according to the instructions given in the User Guide before restarting the program.\n";
    }
    else
    {
        cout<<"cannot create file "<<input_params_file_name<<'\n';
        return false;
    }
	
//	copy_file(input_params_file_name, "./", "./", template_input_params_file_name);
    
    return true;
}

void OmegaMaxEnt_data::display_license()
{
	char lic_option;
	string buf;
	
	cout<<OmegaMaxEnt_copyright_notice;
	
	cin.clear();
	cin.get(lic_option);
	if (lic_option!='\n')
	{
		cin.clear();
		getline(cin,buf);
	}
	
	if (lic_option=='l')
	{
		cout<<GPL3_complete_license;
		cout<<"Press ENTER to continue.\n";
		cin.clear();
		getline(cin,buf);
	}
}

void OmegaMaxEnt_data::display_notice()
{
	char option1, option2;
	string buf;
	
	cout<<short_notice<<endl;
	cout<<"Press ENTER to continue.\n";
	
	cin.clear();
	cin.get(option1);
	if (option1!='\n')
	{
		cin.clear();
		cin.get(option2);
		if (option2!='\n')
		{
			cin.clear();
			getline(cin,buf);
		}
	}
	if (option1=='w' || option2=='w')
	{
		cout<<warranty_notice;
		cin.clear();
		getline(cin,buf);
	}
	if (option1=='c' || option2=='c')
	{
		cout<<distribute_notice;
		cin.clear();
		getline(cin,buf);
	}
	
}

bool polyval(double x0, vec cfs, vec x, vec &y)
{
	int D=cfs.n_rows-1;
	vec Dx=x-x0;
	y=cfs(0)*pow(Dx,D);
	for (int j=1; j<D-1; j++)
		y=y+cfs(j)*pow(Dx,D-j);
	
	y=y+cfs(D-1)*Dx+cfs(D);
	
	return y.is_finite();
}

bool polyfit(vec x, vec y, int D, double x0, vec &cfs)
{
	int N=x.n_rows;
	
	mat X(N,D+1);
	X.zeros();
	
	int j;
	
	vec Dx=x-x0;
	for (j=0; j<D-1; j++)
	{
		X.col(j)=pow(Dx,D-j);
	}
	X.col(D-1)=Dx;
	X.col(D)=ones<vec>(N);
	
	mat A=X.t()*X;
	vec B=X.t()*y;
	
	cfs=solve(A,B);
	
	return cfs.is_finite();
}

void pascal(int n, imat &P)
{
	P.zeros(n,n);
	P.col(0)=ones<ivec>(n);
	P.row(0)=ones<irowvec>(n);
	int j,l;
	for (j=1; j<n; j++)
		for (l=1; l<n; l++)
			P(j,l)=P(j,l-1)+P(j-1,l);
}

void remove_spaces_front(string &str)
{
    int j=0;
    while (str[j]==' ' || str[j]=='\t') j++;
    str=str.substr(j);
}

void remove_spaces_back(string &str)
{
    int j=str.size()-1;
    while (str[j]==' ' || str[j]=='\t') j--;
    str.resize(j+1);
}

void remove_spaces_ends(string &str)
{
	remove_spaces_front(str);
	remove_spaces_back(str);
}

