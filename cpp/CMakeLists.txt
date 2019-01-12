cmake_minimum_required (VERSION 2.6)

project(OmegaMaxEnt)

add_executable(OmegaMaxEnt graph_2D.cpp graph_3D.cpp generique.cpp OmegaMaxEnt_data.cpp OmegaMaxEnt_main.cpp)

target_compile_options(OmegaMaxEnt PRIVATE -std=c++11 -O0)

target_compile_definitions(OmegaMaxEnt PRIVATE ARMA_DONT_USE_WRAPPER)

target_link_libraries(OmegaMaxEnt blas lapack fftw3 gsl)

install(TARGETS OmegaMaxEnt DESTINATION bin)

