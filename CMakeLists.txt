cmake_minimum_required(VERSION 3.22)
project(KroneckerProduct)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")

add_executable(KroneckerProduct main.cpp Utils.h Matrix.h)
