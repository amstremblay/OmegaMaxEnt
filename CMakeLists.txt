#CMakeLists.txt modified by Moise Rousseau (September 21st 2022)

cmake_minimum_required (VERSION 3.18)

project(
    OmegaMaxEnt 
    LANGUAGES CXX 
    VERSION 1.0 
    DESCRIPTION "Analytic continuation of imaginary time/frequency Green or response functions"
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ftree-vectorize")

add_executable(OmegaMaxEnt 
    cpp/graph_2D.cpp
    cpp/graph_3D.cpp
    cpp/generique.cpp
    cpp/OmegaMaxEnt_data.cpp
    cpp/OmegaMaxEnt_main.cpp
)

# Looking for deps (LAPACK,  Armadillo, FFTW3)
find_package(LAPACK REQUIRED)
find_package(Armadillo REQUIRED) #Moise: works with version 10.8.2
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external")
find_package(FFTW REQUIRED)

target_include_directories (
    OmegaMaxEnt PRIVATE cpp ${LAPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS}
)

target_compile_definitions(OmegaMaxEnt PRIVATE ARMA_DONT_USE_WRAPPER)

target_link_libraries(OmegaMaxEnt ${LAPACK_LIBRARIES} ${ARMADILLO_LIBRARIES} ${FFTW_LIBRARIES})

install(TARGETS OmegaMaxEnt DESTINATION bin)
