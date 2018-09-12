
OmegaMaxEnt : analytic continuation of numerical Matsubara data.


1) To run OmegaMaxEnt, you need Python [https://www.python.org/downloads/] with the Matplotlib package installed [http://matplotlib.org/users/installing.html]. Also make sure you have version 4.8, or higher, of the GNU Compiler Collection (gcc) and BLAS and LAPACK installed. 

Mac users: The lapack version that comes with Xcode used to work well in the past, but the version included in Xcode 9.4 causes some problems. Therefore, it is best to use another version. To install one, the simplest way is to use homebrew (https://brew.sh/) with the command "brew install lapack". The makefile provided in the Mac directory is written for that particular case. The instructions to compile with the library are also provided at the end of its installation by homebrew.

2) Extract the file armadillo-5.600.2.tar.gz (tar zxf armadillo-5.600.2.tar.gz) in the OmegaMaxEnt source code directory. There are more recent versions of armadillo [http://arma.sourceforge.net], but I recommend using the version provided to avoid any incompatibility between versions. 

3) Copy the makefile from the directory corresponding to your system (Mac or Linux) to the source code directory and execute the command "make"

4) If the executable file OmegaMaxEnt was created successfully, execute it. If the program works well, 6 figures should open after a few seconds, one with the title "Spectrum at optimal alpha", with five curves plotted, four of which almost identical. Close all the figures and enter any letter other than 'y' in the terminal.
