/*
 file graph_2D.h
 definition of class "graph_2D" that uses pyplot from the Python library Matplotlib (or GNUPLOT but in a rudimentary way) to plot 1D functions.
 It is part of the program OmegaMaxEnt (other source files: graph_2D.cpp, OmegaMaxEnt_main.cpp, OmegaMaxEnt_data.h, OmegaMaxEnt_data.cpp, generique.h, generique.cpp)
 
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


#ifndef GRAPH_2D_H
#define GRAPH_2D_H

#include "includeDef.h"
#include <list>
#include <array>
#include <vector>
#include <string>

static string figs_ind_file_name("figs_ind.dat");

extern "C++"
{
    typedef array<double, 2> point;
    
    enum plot_prog {GNUPLOT, PYPLOT};
    
    class graph_2D
    {
    public:
        graph_2D(plot_prog =PYPLOT, int =0);
        ~graph_2D();
        
        void add_data(double*, double*, int);
        void print_data();
        void curve_plot(char* =NULL, int=0);
        void curve_plot(double*, double*, int, char* =NULL, int=0);
        void set_axes_lims(double*, double*);
        void set_axes_labels(const char*, const char*);
        void add_to_legend(const char*);
		void add_attribute(const char *attr);
		void add_title(const char *ttl);
        
        double labels_fontsize;
        double legend_fontsize;
		double title_fontsize;
        int legend_loc;
		
		static void open_pipe();
		static void close_pipe();
        static void show_figures();
        static void close_figures();
		static void show_commands(bool show_comm){show_command=show_comm;}
		static void reset_figs_ind_file(){if (figs_ind_file) figs_ind_file.close(); figs_ind_file.open(figs_ind_file_name); ind_file=0;}
		
        static FILE *plot_pipe;
		static FILE *file_pipe;
		static bool display_figures;
		static bool print_to_file;
        static int ind_file;
        static int fig_ind_max;
        static char program_name[100];
        static char config_command[200];
        static char fig_command_format[100];
        static char plot_function[100];
        static char load_data_command_format[100];
        static char plot_data_command_format[100];
        static char plot_data_command_separator[100];
		static char title_command_format[100];
        static char xlabel_command_format[100];
        static char ylabel_command_format[100];
        static char xlims_command_format[100];
        static char ylims_command_format[100];
        static char legend_function[100];
        static char plot_closing_command[100];
        static char show_figures_command[100];
        static char close_figures_command[100];
        static char close_figures_command_format[100];
        static char tmp_file_name_format[100];
		static bool show_command;
		static plot_prog plotting_program;
		static char file_name[100];
		static char file_name_format[100];
		static char figs_dir[100];
//		static char scripts_dir[100];
		static ofstream figs_ind_file;
        
    private:
        void plot_with_pyplot(char* =NULL, int=0);
        void plot_with_gnuplot(char* =NULL, int=0);
        
//        char fig_name[100];
        char title[400];
        char xlabel[100];
        char ylabel[100];

        double xlims[2];
        double ylims[2];
        
        list< vector<point> > list_curves;
        list< string > curves_names;
		list< string > curves_attributes;
		
		int fig_ind;
    };
    
}

#endif
