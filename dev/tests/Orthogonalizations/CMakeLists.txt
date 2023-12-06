cmake_minimum_required(VERSION 3.5)

project(tests LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

set(EXECUTABLES TestHouseholder
                TestCholesky
                TestMGS) # update with new tests here

include(GoogleTest)
enable_testing()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra")
  foreach (test ${EXECUTABLES})
    add_executable(${test} ${test}.cu)
    target_link_libraries(
      ${test}
      GTest::gtest_main
      gmock_main
      blaspp
      lapackpp
      cusolver
    )
  gtest_discover_tests(${test})
endforeach ()

set_target_properties(${EXECUTABLES}
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                      "${ROOT_DIR}/dist/bin/")
set_target_properties(${EXECUTABLES}
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
