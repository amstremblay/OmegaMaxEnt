/*
 file OmegaMaxEnt_main.cpp
 main function for the program OmegaMaxEnt that performs the analytic continuation of numerical Matsubara Green and correlation functions.
 
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

int main(int arg_N, char *args[])
{
	OmegaMaxEnt_data maxent1(arg_N, args);
    return maxent1.loop_run();
}
