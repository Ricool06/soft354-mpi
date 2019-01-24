cmake_minimum_required(VERSION 3.10)
project(soft354_mpi)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(EXECUTABLE soft354_mpi)
set(LIBRARY GaussLib)

find_path(LODEPNG_INCLUDE_DIR lodepng.h)
include_directories(${LODEPNG_INCLUDE_DIR})
find_library(LODEPNG_LIB lodepng REQUIRED)

include_directories(lib)

add_subdirectory(lib)

configure_file(img/tiger.png img/tiger.png COPYONLY)
configure_file(img/flowers.png img/flowers.png COPYONLY)
configure_file(img/hmm.png img/hmm.png COPYONLY)
configure_file(img/tiny.png img/tiny.png COPYONLY)

add_executable(soft354_mpi main.cpp)
target_link_libraries(${EXECUTABLE} ${LIBRARY})
target_link_libraries(${EXECUTABLE} ${LODEPNG_LIB})
