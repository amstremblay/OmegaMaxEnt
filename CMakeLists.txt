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
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fopenmp -ffast-math -funroll-loops -march=native -ftree-vectorize")

add_executable(OmegaMaxEnt 
    cpp/graph_2D.cpp
    cpp/graph_3D.cpp
    cpp/generique.cpp
    cpp/OmegaMaxEnt_data.cpp
    cpp/OmegaMaxEnt_main.cpp
)

# Looking for deps (LAPACK, FFTW3)
find_package(LAPACK REQUIRED)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external")
find_package(FFTW REQUIRED)

# Check suitable Armadillo version or install 5.600.2
if (DOWNLOAD_ARMADILLO)
    include(ExternalProject)
    ExternalProject_Add(Armadillo
        PREFIX ${CMAKE_SOURCE_DIR}/external
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/Armadillo
        URL https://www.physique.usherbrooke.ca/MaxEnt/images/c/cb/Armadillo-5.600.2.tar.gz
        URL_HASH MD5=268dd13d56c8c6423037a867ed2b3f4b
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
        STEP_TARGETS build
        BUILD_BYPRODUCTS ${CMAKE_SOURCE_DIR}/external/Armadillo/libarmadillo.so
    )
    add_dependencies(OmegaMaxEnt Armadillo)
    set(ARMADILLO_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/external/Armadillo/include)
    set(ARMADILLO_LIBRARIES ${CMAKE_SOURCE_DIR}/external/Armadillo/libarmadillo.so)
else()
    find_package(Armadillo REQUIRED)
    if (${ARMADILLO_VERSION_MAJOR} VERSION_GREATER_EQUAL "6")
      message(FATAL_ERROR "OmegaMaxEnt requires Armadillo version less than 6.0.0. Please try configuring with option -DDOWNLOAD_ARMADILLO=1")
    endif()
endif()

target_include_directories (
    OmegaMaxEnt PRIVATE cpp ${ARMADILLO_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS}
)

#target_compile_definitions(OmegaMaxEnt PRIVATE ARMA_DONT_USE_WRAPPER)

target_link_libraries(OmegaMaxEnt ${LAPACK_LIBRARIES} ${ARMADILLO_LIBRARIES} ${FFTW_LIBRARIES})

install(TARGETS OmegaMaxEnt DESTINATION bin)
