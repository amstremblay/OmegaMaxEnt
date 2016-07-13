= Installation instructions =

Omega MaxEnt : analytic continuation of numerical Matsubara data.

1) To run Omega MaxEnt, you need Python https://www.python.org with the Matplotlib http://matplotlib.org package installed. Also make sure you have version 4.8, or higher, of the GNU Compiler Collection (gcc) https://gcc.gnu.org GNU and BLAS and LAPACK installed (Linux) or the most recent version of Xcode compatible with your system (Mac).

2) To install from Source, Download and extract (with command "tar zxf file_name.tar.gz") the source files:
Source_OmegaMaxEnt_source_2016-06-10.tar.gz

3) Download armadillo-5.600.2.tar.gz and extract it in the OmegaMaxEnt source code directory. Note: There are more recent versions of [http://arma.sourceforge.net armadillo], but there seems to be an incompatibility with those versions. This will be investigated, but for now, please use the version provided.

4) Copy the makefile from the directory corresponding to your system (OSX or Linux) to the source code directory and execute the command "make".

5) If the executable file OmegaMaxEnt was created successfully, execute it. If the program works well, 6 figures should open after a few seconds, one titled "Spectrum at optimal \alpha" with five curves plotted, four of which almost identical. Close all the figures and enter any letter other than 'y' in the terminal.
