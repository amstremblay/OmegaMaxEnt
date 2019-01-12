/*
 file graph_2D.cpp
 functions definitions for class "graph_2D" defined in file graph_2D.h
 It is part of the program OmegaMaxEnt (other source files: OmegaMaxEnt_main.cpp, OmegaMaxEnt_data.h, OmegaMaxEnt_data.cpp, generique.h, generique.cpp)
 
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

#include "graph_2D.h"

FILE *graph_2D::plot_pipe=NULL;
FILE *graph_2D::file_pipe=NULL;
bool graph_2D::display_figures=true;
bool graph_2D::print_to_file=true;
int graph_2D::ind_file=0;
int graph_2D::fig_ind_max=0;
char graph_2D::program_name[100];
char graph_2D::config_command[200];
char graph_2D::fig_command_format[100];
char graph_2D::plot_function[100];
char graph_2D::load_data_command_format[100];
char graph_2D::plot_data_command_format[100];
char graph_2D::plot_data_command_separator[100];
char graph_2D::xlabel_command_format[100];
char graph_2D::ylabel_command_format[100];
char graph_2D::title_command_format[100];
char graph_2D::xlims_command_format[100];
char graph_2D::ylims_command_format[100];
char graph_2D::legend_function[100];
char graph_2D::plot_closing_command[100];
char graph_2D::show_figures_command[100];
char graph_2D::close_figures_command[100];
char graph_2D::close_figures_command_format[100];
char graph_2D::tmp_file_name_format[100];
bool graph_2D::show_command=false;
plot_prog graph_2D::plotting_program=PYPLOT;
char graph_2D::file_name[100];
char graph_2D::file_name_format[100];
char graph_2D::figs_dir[100];
//char graph_2D::scripts_dir[100];
//ofstream graph_2D::figs_ind_file(figs_ind_file_name);
ofstream graph_2D::figs_ind_file;

graph_2D::graph_2D(plot_prog plot_prog_init, int fig_ind_p)
{
    if (fig_ind_p>0)
        fig_ind=fig_ind_p;
    else
    {
        fig_ind_max++;
        fig_ind=fig_ind_max;
    }
	
	if (figs_ind_file)
	{
		figs_ind_file<<setiosflags(ios::left);
	}
    
    labels_fontsize=20;
    legend_fontsize=20;
	title_fontsize=20;
    legend_loc=0;
    xlims[0]=0;
    xlims[1]=0;
    ylims[0]=0;
    ylims[1]=0;
	
//	fig_name[0]='\0';
	title[0]='\0';
	xlabel[0]='\0';
	ylabel[0]='\0';
	
	plotting_program=plot_prog_init;
	
	switch (plotting_program)
	{
		case PYPLOT:
			strcpy(figs_dir,"OmegaMaxEnt_figs_data");
			strcpy(file_name_format,"OmegaMaxEnt_figs_%d.py");
			strcpy(program_name,"python");
			strcpy(config_command,"from matplotlib import pyplot\nfrom numpy import loadtxt\n");
			strcpy(load_data_command_format,"x%d,y%d=loadtxt(\"%s\",unpack=True)\n");
			strcpy(fig_command_format,"pyplot.figure(%d)\n");
			strcpy(plot_function,"pyplot.plot(");
			strcpy(plot_data_command_format,"x%d,y%d");
			strcpy(plot_data_command_separator,",");
			strcpy(xlabel_command_format,"pyplot.xlabel(\"%s\",fontsize=%g)\n");
			strcpy(ylabel_command_format,"pyplot.ylabel(\"%s\",fontsize=%g)\n");
			strcpy(title_command_format,"pyplot.title(\"%s\",fontsize=%g)\n");
			strcpy(xlims_command_format,"pyplot.xlim(%g,%g)\n");
			strcpy(ylims_command_format,"pyplot.ylim(%g,%g)\n");
			strcpy(legend_function,"pyplot.legend(");
			strcpy(plot_closing_command,")\n");
			strcpy(tmp_file_name_format,figs_dir);
			strcat(tmp_file_name_format,"/fig_file_%d.dat");
			strcpy(show_figures_command,"pyplot.show()\n");
			strcpy(close_figures_command,"pyplot.close(\"all\")\n");
			strcpy(close_figures_command_format,"pyplot.close(%d)\n");
			break;
		case GNUPLOT:
			strcpy(program_name,"gnuplot");
			strcpy(config_command,"set terminal x11\n");
			strcpy(plot_function,"plot");
			strcpy(plot_data_command_format," \"%s\" with lines");
			strcpy(plot_data_command_separator,",");
			strcpy(tmp_file_name_format,"fig_file_%d.dat");
			strcpy(show_figures_command,"\n");
			break;
		default:
			break;
	}
	
	struct stat file_stat;
	if (stat(figs_dir,&file_stat))
		mkdir(figs_dir, S_IRWXU | S_IRWXG | S_IRWXO);
	
}

void graph_2D::open_pipe()
{
	if (display_figures && !plot_pipe)
	{
		plot_pipe=popen(program_name,"w");
		fputs(config_command,plot_pipe);
	}
	
	if (print_to_file && !file_pipe)
	{
		sprintf(file_name,file_name_format,ind_file);
		file_pipe=fopen(file_name,"w");
		if (figs_ind_file)
		{
			figs_ind_file<<setw(10)<<ind_file;
		}
		fputs(config_command,file_pipe);
	}
	
	if (show_command) cout<<config_command;
}

void graph_2D::close_pipe()
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

graph_2D::~graph_2D()
{
	close_pipe();
}

void graph_2D::set_axes_lims(double *xlims_par, double *ylims_par)
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
}

void graph_2D::set_axes_labels(const char *xlabel_par, const char *ylabel_par)
{
    if (xlabel_par) strcpy(xlabel,xlabel_par);
    if (ylabel_par) strcpy(ylabel,ylabel_par);
}

void graph_2D::add_to_legend(const char *lgd_entry)
{
    string str(lgd_entry);
    curves_names.push_back(str);
}

void graph_2D::add_attribute(const char *attr)
{
	string str(attr);
	curves_attributes.push_back(str);
}

void graph_2D::close_figures()
{
    char close_command[100];
    sprintf(close_command,close_figures_command_format,fig_ind_max);
    if (plot_pipe) fputs(close_command,plot_pipe);
	if (file_pipe) fputs(close_command,file_pipe);
//    fputs(close_figures_command,plot_pipe);
 //   pclose(plot_pipe);
 //   plot_pipe=NULL;
}

void graph_2D::show_figures()
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

	close_pipe();
	
	if (!print_to_file)
	{
		char tmp_file_name[100];
		for (int j=0; j<ind_file; ++j)
		{
			sprintf(tmp_file_name,tmp_file_name_format,j);
			remove(tmp_file_name);
		}
		ind_file=0;
	}
}

void graph_2D::add_data(double *x, double *y, int size_data)
{
    vector<point> vtmp(size_data);
    for (int i=0; i<size_data; i++)
    {
        vtmp[i][0]=x[i];
        vtmp[i][1]=y[i];
    }
    list_curves.push_back(vtmp);
}

void graph_2D::add_title(const char *ttl)
{
	strcpy(title,ttl);
}

void graph_2D::print_data()
{
    cout<<"size of list_curves: "<<list_curves.size()<<'\n';
    
    for (auto list_ptr=list_curves.begin(); list_ptr!=list_curves.end(); list_ptr++)
    {
        cout<<"size of element: "<<list_ptr->size()<<'\n';
        cout<<setiosflags(ios::left);
        for (int i=0; i<list_ptr->size(); i++)
        {
            cout<<setw(30)<<(*list_ptr)[i][0]<<(*list_ptr)[i][1]<<'\n';
        }
    }
}

void graph_2D::curve_plot(double *x, double *y, int size_data, char *extra_commands, int fig_ind_p)
{
    add_data(x, y, size_data);
    curve_plot(extra_commands, fig_ind_p);
}

void graph_2D::curve_plot(char *extra_commands, int fig_ind_p)
{
    switch (plotting_program)
    {
        case PYPLOT:
            plot_with_pyplot(extra_commands, fig_ind_p);
            break;
        case GNUPLOT:
            plot_with_gnuplot(extra_commands, fig_ind_p);
            break;
        default:
            plot_with_pyplot(extra_commands, fig_ind_p);
            break;
    }
}

void graph_2D::plot_with_pyplot(char *extra_commands, int fig_ind_p)
{
    fstream data_file;
    char tmp_file_name[100];
	char file_name[200];
    char load_data_command[200];
    char fig_command[100];
    char plot_command[10000];
    char data_plot_command[100];
    char legend_command[10000];
    char legend_loc_command[10];
    char xlabel_command[100];
    char ylabel_command[100];
    char xlims_command[100];
    char ylims_command[100];
	char title_command[400];
    
    if (fig_ind_p>0)
        fig_ind=fig_ind_p;
    
    if (fig_ind>fig_ind_max)
        fig_ind_max=fig_ind;
    
    sprintf(fig_command,fig_command_format,fig_ind);
	
	open_pipe();
	
	if (!plot_pipe && !file_pipe)
	{
		cout<<"plot_with_pyplot(): no pipe open\n";
		return;
	}

    strcpy(plot_command,plot_function);
	
    strcpy(legend_command,legend_function);
    sprintf(legend_loc_command,",loc=%d",legend_loc);
	sprintf(title_command, title_command_format, title, title_fontsize);
    sprintf(xlabel_command,xlabel_command_format,xlabel,labels_fontsize);
    sprintf(ylabel_command,ylabel_command_format,ylabel,labels_fontsize);
    sprintf(xlims_command,xlims_command_format,xlims[0],xlims[1]);
    sprintf(ylims_command,ylims_command_format,ylims[0],ylims[1]);
    
    auto list_ptr=list_curves.begin();
    auto lgd_ptr=curves_names.begin();
	auto attr_ptr=curves_attributes.begin();
    int j=ind_file;
	
/*
	strcpy(file_name,figs_dir);
	sprintf(tmp_file_name,tmp_file_name_format,j);
	strcat(file_name,"/");
	strcat(file_name,tmp_file_name);
	cout<<"tmp_file_name: "<<tmp_file_name<<endl;
	cout<<"file_name: "<<file_name<<endl;
 	data_file.open(file_name,ios::out);
 */
	sprintf(tmp_file_name,tmp_file_name_format,j);
    data_file.open(tmp_file_name,ios::out);
    if (data_file && (plot_pipe || file_pipe))
    {
        data_file<<setprecision(16)<<setiosflags(ios::left)<<setiosflags(ios::scientific);
        for (int i=0; i<list_ptr->size(); i++)
        {
            data_file<<setw(30)<<(*list_ptr)[i][0]<<(*list_ptr)[i][1]<<'\n';
        }
        data_file.close();
        sprintf(load_data_command,load_data_command_format,j,j,tmp_file_name);
        if (show_command) cout<<load_data_command;
        if (plot_pipe) fputs(load_data_command,plot_pipe);
		if (file_pipe) fputs(load_data_command,file_pipe);
        sprintf(data_plot_command,plot_data_command_format,j,j);
        strcat(plot_command,data_plot_command);
		if (curves_attributes.size()>0)
		{
			strcat(plot_command,plot_data_command_separator);
			strcat(plot_command,attr_ptr->data());
		}
        if (curves_names.size()>0)
        {
            if (list_curves.size()>1) strcat(legend_command,"(");
            strcat(legend_command,"\'");
            strcat(legend_command,(*lgd_ptr).data());
            strcat(legend_command,"\'");
        }
		strcat(plot_command,")\n");
    }
    else
    {
        if (!data_file)
			cout<<"curve_plot: file "<<tmp_file_name<<" could not be opened\n";
        else
            cout<<"curve_plot: plotting program not found\n";
    }
    list_ptr++;
    lgd_ptr++;
	attr_ptr++;
    j++;
    while (list_ptr!=list_curves.end())
    {
/*
		strcpy(file_name,figs_dir);
		sprintf(tmp_file_name,tmp_file_name_format,j);
		strcat(file_name,"/");
		strcat(file_name,tmp_file_name);
		data_file.open(file_name,ios::out);
 */
		sprintf(tmp_file_name,tmp_file_name_format,j);
		data_file.open(tmp_file_name,ios::out);
        if (data_file && (plot_pipe || file_pipe))
        {
			strcat(plot_command,plot_function);
            data_file<<setprecision(16)<<setiosflags(ios::left)<<setiosflags(ios::scientific);
            for (int i=0; i<list_ptr->size(); i++)
            {
                data_file<<setw(30)<<(*list_ptr)[i][0]<<(*list_ptr)[i][1]<<'\n';
            }
            data_file.close();
            sprintf(load_data_command,load_data_command_format,j,j,tmp_file_name);
            if (show_command) cout<<load_data_command;
            if (plot_pipe) fputs(load_data_command,plot_pipe);
			if (file_pipe) fputs(load_data_command,file_pipe);
//            strcat(plot_command,plot_data_command_separator);
            sprintf(data_plot_command,plot_data_command_format,j,j);
            strcat(plot_command,data_plot_command);
			if (curves_attributes.size()>(j-ind_file))
			{
				strcat(plot_command,plot_data_command_separator);
				strcat(plot_command,attr_ptr->data());
			}
            if (curves_names.size()>(j-ind_file))
            {
                strcat(legend_command,",\'");
                strcat(legend_command,(*lgd_ptr).data());
                strcat(legend_command,"\'");
            }
			strcat(plot_command,")\n");
        }
        else
        {
            if (!data_file)
                cout<<"curve_plot: file "<<tmp_file_name<<" could not be opened\n";
            else
                cout<<"curve_plot: plotting program not found\n";
        }
        list_ptr++;
        lgd_ptr++;
		attr_ptr++;
        j++;
    }
    if (list_curves.size()>1) strcat(legend_command,")");
    strcat(legend_command,legend_loc_command);
    strcat(legend_command,")\n");
    ind_file=j;
    if (plot_pipe) fputs(fig_command, plot_pipe);
	if (file_pipe) fputs(fig_command, file_pipe);
    if (show_command) cout<<fig_command;
//    strcat(plot_command,plot_closing_command);
    if (show_command) cout<<plot_command;
    if (plot_pipe) fputs(plot_command,plot_pipe);
	if (file_pipe) fputs(plot_command,file_pipe);
    if (curves_names.size()>0)
    {
        if (plot_pipe) fputs(legend_command,plot_pipe);
		if (file_pipe) fputs(legend_command,file_pipe);
        if (show_command) cout<<legend_command;
    }
	if (title[0])
	{
		if (plot_pipe) fputs(title_command,plot_pipe);
		if (file_pipe) fputs(title_command,file_pipe);
	}
    if (xlabel[0])
	{
		if (plot_pipe) fputs(xlabel_command,plot_pipe);
		if (file_pipe) fputs(xlabel_command,file_pipe);
	}
    if (ylabel[0])
	{
		if (plot_pipe) fputs(ylabel_command,plot_pipe);
		if (file_pipe) fputs(ylabel_command,file_pipe);
	}
    if (xlims[0]!=xlims[1])
	{
		if (plot_pipe) fputs(xlims_command,plot_pipe);
		if (file_pipe) fputs(xlims_command,file_pipe);
	}
    if (ylims[0]!=ylims[1])
	{
		if (plot_pipe) fputs(ylims_command,plot_pipe);
		if (file_pipe) fputs(ylims_command,file_pipe);
	}
	if (extra_commands)
	{
		if (plot_pipe) fputs(extra_commands,plot_pipe);
		if (file_pipe) fputs(extra_commands,file_pipe);
	}

    if (plot_pipe) fflush(plot_pipe);
	if (file_pipe) fflush(file_pipe);

}

void graph_2D::plot_with_gnuplot(char *extra_commands, int fig_ind_p)
{
    fstream data_file;
    char tmp_file_name[100];
    char plot_command[10000];
    char data_plot_command[100];
	
	open_pipe();
    
    strcpy(plot_command,plot_function);
    
    auto list_ptr=list_curves.begin();
    int j=ind_file;
    sprintf(tmp_file_name,tmp_file_name_format,j);
    data_file.open(tmp_file_name,ios::out);
    if (data_file && plot_pipe)
    {
        data_file<<setprecision(16)<<setiosflags(ios::left)<<setiosflags(ios::scientific);
        for (int i=0; i<list_ptr->size(); i++)
        {
            data_file<<setw(30)<<(*list_ptr)[i][0]<<(*list_ptr)[i][1]<<'\n';
        }
        data_file.close();
        sprintf(data_plot_command,plot_data_command_format,tmp_file_name);
        strcat(plot_command,data_plot_command);
    }
    else
    {
        if (!data_file)
            cout<<"curve_plot: open file failed\n";
        else
            cout<<"curve_plot: plotting program not found\n";
    }
    list_ptr++;
    j++;
    while (list_ptr!=list_curves.end())
    {
        sprintf(tmp_file_name,tmp_file_name_format,j);
        data_file.open(tmp_file_name,ios::out);
        if (data_file && plot_pipe)
        {
            data_file<<setprecision(16)<<setiosflags(ios::left)<<setiosflags(ios::scientific);
            for (int i=0; i<list_ptr->size(); i++)
            {
                data_file<<setw(30)<<(*list_ptr)[i][0]<<(*list_ptr)[i][1]<<'\n';
            }
            data_file.close();
            strcat(plot_command,plot_data_command_separator);
            sprintf(data_plot_command,plot_data_command_format,tmp_file_name);
            strcat(plot_command,data_plot_command);
        }
        else
        {
            if (!data_file)
                cout<<"curve_plot: open file failed\n";
            else
                cout<<"curve_plot: plotting program not found\n";
        }
        list_ptr++;
        j++;
    }
    ind_file=j;
    strcat(plot_command,"\n");
    if (show_command) cout<<plot_command<<'\n';
    fputs(plot_command,plot_pipe);
    fflush(plot_pipe);
    getchar();
//    fflush(plot_pipe);
//    pclose(plot_pipe);
/*
    for (int j=0; j<list_curves.size(); ++j)
    {
        sprintf(tmp_file_name,tmp_file_name_format,j);
        remove(tmp_file_name);
    }
*/
}
