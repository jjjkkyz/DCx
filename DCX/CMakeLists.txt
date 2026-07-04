cmake_minimum_required(VERSION 3.15)
project(DCX_implementation)

find_package(pybind11 CONFIG REQUIRED)
set(SOURCES DCX_main.cpp)
pybind11_add_module(DCX_main MODULE ${SOURCES})
install(TARGETS DCX_main DESTINATION dcx_pkg)