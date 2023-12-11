cmake_minimum_required(VERSION 3.5)


project(LinearAlgebraKernels)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(FetchContent)

# check if blasspp is going to use cuda
if(NOT gpu_acceleration)
  include(cmake/CMakelists.NoGpuAcceleration.txt.in)
else()
  include(cmake/CMakelists.GpuAcceleration.txt.in)
endif()

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
