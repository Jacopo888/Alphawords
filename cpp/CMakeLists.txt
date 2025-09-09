cmake_minimum_required(VERSION 3.18)
project(qlex)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(pybind11 REQUIRED)

# Include Quackle headers
include_directories(../third_party/quackle/src)

# Source files
set(SOURCES
    qlex_wrapper.cpp
)

# Create Python module
pybind11_add_module(qlex ${SOURCES})

# Link with Quackle libraries
target_link_libraries(qlex PRIVATE
    ../third_party/quackle/build/src/libquackle.a
)

# Compiler flags
target_compile_definitions(qlex PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_compile_options(qlex PRIVATE -fPIC -O3)
