cmake_minimum_required(VERSION 3.16)
project(gpusoa)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    add_subdirectory(src)
endif()
