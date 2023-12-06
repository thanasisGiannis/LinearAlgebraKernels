cmake_minimum_required(VERSION 3.5)

project(LinearAlgebraKernels)

#add_definitions(-std=c++2a)
#set(CMAKE_CXX_COMPILER "nvc++")
set(CMAKE_CXX_FLAGS "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30 -DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM} )
list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)

find_package(OpenMP REQUIRED)
include(FetchContent)
FetchContent_Declare(
  blaspp
  URL https://github.com/icl-utk-edu/blaspp/archive/refs/heads/master.zip
)
FetchContent_Declare(
  lapackpp
  URL https://github.com/icl-utk-edu/lapackpp/archive/refs/heads/master.zip
)
FetchContent_MakeAvailable(blaspp)
FetchContent_MakeAvailable(lapackpp)

add_subdirectory(${ROOT_DIR}/dev)
