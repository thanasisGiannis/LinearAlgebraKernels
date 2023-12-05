cmake_minimum_required(VERSION 3.5)

project(LinearAlgebraKernels)

# header files with definitions
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# header files with declarations of templates
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# tests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
