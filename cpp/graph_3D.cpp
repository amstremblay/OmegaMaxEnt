/*
 file graph_3D.cpp
 functions definitions for class "graph_3D" defined in file graph_3D.h
 It is part of the program OmegaMaxEnt (other source files: OmegaMaxEnt_main.cpp, OmegaMaxEnt_data.h, OmegaMaxEnt_data.cpp, generique.h, generique.cpp, graph_2D.h, graph_2D.cpp)
 
 Copyright (C) 2018 Dominic Bergeron (dominic.bergeron@usherbrooke.ca)
 
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

#include "graph_3D.h"

FILE *graph_3D::plot_pipe=NULL;
FILE *graph_3D::file_pipe=NULL;
bool graph_3D::display_figures=true;
bool graph_3D::print_to_file=true;
int graph_3D::file_ind=0;
int graph_3D::fig_ind=1;
char graph_3D::program_name[100];
char graph_3D::config_command[200];
char graph_3D::fig_command_format[100];
char graph_3D::load_x_command_format[100];
char graph_3D::load_y_command_format[100];
char graph_3D::load_z_command_format[100];
char graph_3D::xlabel_command_format[100];
char graph_3D::ylabel_command_format[100];
char graph_3D::zlabel_command_format[100];
char graph_3D::title_command_format[100];
char graph_3D::xlims_command_format[100];
char graph_3D::ylims_command_format[100];
char graph_3D::zlims_command_format[100];
char graph_3D::show_figures_command[100];
bool graph_3D::show_command=false;
char graph_3D::file_name[100];
char graph_3D::file_name_format[100];
char graph_3D::figs_dir[100];
char graph_3D::mesh_command[100];
char graph_3D::x_file_name_format[100];
char graph_3D::y_file_name_format[100];
char graph_3D::z_file_name_format[100];
char graph_3D::plot_command_format[500];

ofstream graph_3D::figs_ind_file;

graph_3D::graph_3D(int fig_ind_p)
{
    if (fig_ind_p>0)
        fig_ind=fig_ind_p;
	
	if (figs_ind_file)
	{
		figs_ind_file<<setiosflags(ios::left);
	}
    
    labels_fontsize=16;
	title_fontsize=18;
    xlims[0]=0;
    xlims[1]=0;
    ylims[0]=0;
    ylims[1]=0;
	zlims[0]=0;
	zlims[1]=0;
	
	title[0]='\0';
	xlabel[0]='\0';
	ylabel[0]='\0';
	zlabel[0]='\0';
	
	strcpy(figs_dir,"OmegaMaxEnt_figs_data");
	strcpy(file_name_format,"OmegaMaxEnt_surf_figs_%d.py");
	strcpy(program_name,"python");
	strcpy(config_command,"from mpl_toolkits.mplot3d import Axes3D\nfrom matplotlib import pyplot\nfrom matplotlib import cm\nfrom numpy import loadtxt\nfrom matplotlib.ticker import LinearLocator, FormatStrFormatter\nimport numpy as np\n");
	strcpy(load_x_command_format,"x=loadtxt(\"%s\")\n");
	strcpy(load_y_command_format,"y=loadtxt(\"%s\")\n");
	strcpy(load_z_command_format,"Z=loadtxt(\"%s\")\n");
	strcpy(fig_command_format,"fig=pyplot.figure(%d)\n");
	strcpy(mesh_command,"X, Y = np.meshgrid(x, y)\n");
	strcpy(plot_command_format,"ax = fig.gca(projection='3d')\nsurf = ax.plot_surface(X, Y, Z%s)\n");
	strcpy(xlabel_command_format,"ax.set_xlabel(\"%s\",fontsize=%g)\n");
	strcpy(ylabel_command_format,"ax.set_ylabel(\"%s\",fontsize=%g)\n");
	strcpy(zlabel_command_format,"ax.set_zlabel(\"%s\",fontsize=%g)\n");
	strcpy(title_command_format,"pyplot.title(\"%s\",fontsize=%g)\n");
	strcpy(xlims_command_format,"ax.set_xlim(%g,%g)\n");
	strcpy(ylims_command_format,"ax.set_ylim(%g,%g)\n");
	strcpy(zlims_command_format,"ax.set_xlim(%g,%g)\n");
	strcpy(x_file_name_format,figs_dir);
	strcpy(y_file_name_format,figs_dir);
	strcpy(z_file_name_format,figs_dir);
	strcat(x_file_name_format,"/fig_file_x_%d.dat");
	strcat(y_file_name_format,"/fig_file_y_%d.dat");
	strcat(z_file_name_format,"/fig_file_z_%d.dat");
	strcpy(show_figures_command,"pyplot.show()\n");
	
	struct stat file_stat;
	if (stat(figs_dir,&file_stat))
		mkdir(figs_dir, S_IRWXU | S_IRWXG | S_IRWXO);
	
}

void graph_3D::open_pipe()
{
	if (display_figures && !plot_pipe)
	{
		plot_pipe=popen(program_name,"w");
		fputs(config_command,plot_pipe);
	}
	
	if (print_to_file && !file_pipe)
	{
		sprintf(file_name,file_name_format,file_ind);
		file_pipe=fopen(file_name,"w");
		if (figs_ind_file)
		{
			figs_ind_file<<setw(10)<<file_ind;
		}
		fputs(config_command,file_pipe);
	}
	
	if (show_command) cout<<config_command;
}

void graph_3D::close_pipe()
{
	if (plot_pipe)
	{
		fflush(plot_pipe);
		pclose(plot_pipe);
		plot_pipe=NULL;
	}
	if (file_pipe)
	{
		fflush(file_pipe);
		fclose(file_pipe);
		file_pipe=NULL;
	}
}

graph_3D::~graph_3D()
{
	close_pipe();
}

void graph_3D::set_axes_lims(double *xlims_par, double *ylims_par, double *zlims_par)
{
    if (xlims_par)
    {
        xlims[0]=xlims_par[0];
        xlims[1]=xlims_par[1];
    }
    if (ylims_par)
    {
        ylims[0]=ylims_par[0];
        ylims[1]=ylims_par[1];
    }
	if (zlims_par)
	{
		zlims[0]=zlims_par[0];
		zlims[1]=zlims_par[1];
	}
}

void graph_3D::set_axes_labels(const char *xlabel_par, const char *ylabel_par, const char *zlabel_par)
{
    if (xlabel_par) strcpy(xlabel,xlabel_par);
    if (ylabel_par) strcpy(ylabel,ylabel_par);
	if (zlabel_par) strcpy(zlabel,zlabel_par);
}

void graph_3D::show_figures()
{
	
    if (plot_pipe)
	{
		fputs(show_figures_command,plot_pipe);
		fflush(plot_pipe);
	}
	if (file_pipe)
	{
		fputs(show_figures_command,file_pipe);
		fflush(file_pipe);
	}

	file_ind++;
	
	close_pipe();
	
	if (!print_to_file)
	{
		char tmp_file_name[100];
		for (int j=0; j<file_ind; ++j)
		{
			sprintf(tmp_file_name,x_file_name_format,j);
			remove(tmp_file_name);
			sprintf(tmp_file_name,y_file_name_format,j);
			remove(tmp_file_name);
			sprintf(tmp_file_name,z_file_name_format,j);
			remove(tmp_file_name);
		}
		file_ind=0;
	}
}

void graph_3D::add_data(vec xp, vec yp, mat Zp)
{
	x=xp;
	y=yp;
	Z=Zp;
}

void graph_3D::add_title(const char *ttl)
{
	strcpy(title,ttl);
}

void graph_3D::plot_surface(vec x, vec y, mat Z, map<string,string> extra_options, int fig_ind_p)
{
    add_data(x, y, Z);
    plot_surface(extra_options, fig_ind_p);
}

void graph_3D::plot_surface(map<string,string> extra_options, int fig_ind_p)
{
    fstream data_file;
    char tmp_file_name[100];
	char file_name[200];
    char load_x_command[200];
	char load_y_command[200];
	char load_z_command[200];
    char fig_command[100];
    char plot_command[1000];
    char xlabel_command[100];
    char ylabel_command[100];
	char zlabel_command[100];
    char xlims_command[100];
    char ylims_command[100];
	char zlims_command[100];
	char title_command[400];
	string options_str;
	
	string cm_str="cmap=";
	string cm_opt=cm_str+"cm.coolwarm";
	string line_width_str="linewidth=";
	string line_width_opt=line_width_str+"0";
	string antialiased_str="antialiased=";
	string antialiased_opt=antialiased_str+"False";
	
	options_list options({{"cmap","cm.coolwarm"},{"linewidth","0"},{"antialiased","False"}});
	
	for (auto& it : extra_options)
	{
		options[it.first]=it.second;
	}
	
	for (auto& it : options)
	{
		options_str=options_str+", "+it.first+"="+it.second;
	}
    
    if (fig_ind_p>0)
        fig_ind=fig_ind_p;
    
    sprintf(fig_command,fig_command_format,fig_ind);
	
	open_pipe();
	
	if (!plot_pipe && !file_pipe)
	{
		cout<<"plot_surface(): no pipe open\n";
		return;
	}

    int j=fig_ind;
	
	sprintf(tmp_file_name,x_file_name_format,j);
	x.save(tmp_file_name,raw_ascii);
	sprintf(load_x_command,load_x_command_format,tmp_file_name);
	sprintf(tmp_file_name,y_file_name_format,j);
	y.save(tmp_file_name,raw_ascii);
	sprintf(load_y_command,load_y_command_format,tmp_file_name);
	sprintf(tmp_file_name,z_file_name_format,j);
	Z.save(tmp_file_name,raw_ascii);
	sprintf(load_z_command,load_z_command_format,tmp_file_name);
	
	sprintf(plot_command,plot_command_format,options_str.c_str());
	
	if (plot_pipe)
	{
		fputs(load_x_command,plot_pipe);
		fputs(load_y_command,plot_pipe);
		fputs(load_z_command,plot_pipe);
		fputs(mesh_command,plot_pipe);
		fputs(fig_command,plot_pipe);
		fputs(plot_command,plot_pipe);
	}
	if (file_pipe)
	{
		fputs(load_x_command,file_pipe);
		fputs(load_y_command,file_pipe);
		fputs(load_z_command,file_pipe);
		fputs(mesh_command,file_pipe);
		fputs(fig_command,file_pipe);
		fputs(plot_command,file_pipe);
	}
	
	if (title[0])
	{
		sprintf(title_command, title_command_format, title, title_fontsize);
		if (plot_pipe) fputs(title_command,plot_pipe);
		if (file_pipe) fputs(title_command,file_pipe);
	}
    if (xlabel[0])
	{
		sprintf(xlabel_command,xlabel_command_format,xlabel,labels_fontsize);
		if (plot_pipe) fputs(xlabel_command,plot_pipe);
		if (file_pipe) fputs(xlabel_command,file_pipe);
	}
    if (ylabel[0])
	{
		sprintf(ylabel_command,ylabel_command_format,ylabel,labels_fontsize);
		if (plot_pipe) fputs(ylabel_command,plot_pipe);
		if (file_pipe) fputs(ylabel_command,file_pipe);
	}
	if (zlabel[0])
	{
		sprintf(zlabel_command,zlabel_command_format,zlabel,labels_fontsize);
		if (plot_pipe) fputs(zlabel_command,plot_pipe);
		if (file_pipe) fputs(zlabel_command,file_pipe);
	}
    if (xlims[0]!=xlims[1])
	{
		sprintf(xlims_command,xlims_command_format,xlims[0],xlims[1]);
		if (plot_pipe) fputs(xlims_command,plot_pipe);
		if (file_pipe) fputs(xlims_command,file_pipe);
	}
    if (ylims[0]!=ylims[1])
	{
		sprintf(ylims_command,ylims_command_format,ylims[0],ylims[1]);
		if (plot_pipe) fputs(ylims_command,plot_pipe);
		if (file_pipe) fputs(ylims_command,file_pipe);
	}
	if (zlims[0]!=zlims[1])
	{
		sprintf(zlims_command,zlims_command_format,zlims[0],zlims[1]);
		if (plot_pipe) fputs(zlims_command,plot_pipe);
		if (file_pipe) fputs(zlims_command,file_pipe);
	}

    if (plot_pipe) fflush(plot_pipe);
	if (file_pipe) fflush(file_pipe);
	
	fig_ind++;
}
