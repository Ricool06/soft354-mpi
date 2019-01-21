cmake_minimum_required(VERSION 3.12)
project(soft354_mpi)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE soft354_mpi)
set(LIBRARY CannyLib)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

find_path(LODEPNG_INCLUDE_DIR lodepng.h)
include_directories(${LODEPNG_INCLUDE_DIR})
find_library(LODEPNG_LIB lodepng REQUIRED)

find_package(GTest MODULE REQUIRED)

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(test)

configure_file(img/tiger.png img/tiger.png COPYONLY)
configure_file(img/flowers.png img/flowers.png COPYONLY)

add_executable(soft354_mpi main.cpp)
target_link_libraries(${EXECUTABLE} ${LIBRARY})
target_link_libraries(${EXECUTABLE} ${LODEPNG_LIB})
