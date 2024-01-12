cmake_minimum_required(VERSION 3.5)

project(tests LANGUAGES C CXX CUDA)

set(EXECUTABLES TestHouseholder
                TestCholesky
                TestMGS) # update with new tests here

include(GoogleTest)
enable_testing()
  foreach (test ${EXECUTABLES})

    if(NOT gpu_acceleration)
      set_source_files_properties(${test}.cpp PROPERTIES LANGUAGE CXX)
      add_executable(${test} ${test}.cpp)
    else()
      set_source_files_properties(${test}.cpp PROPERTIES LANGUAGE CUDA)
      add_executable(${test} ${test}.cpp)
    endif()
    target_link_libraries(
      ${test}
      GTest::gtest_main
      gmock_main
      blaspp
      lapackpp
    )
  gtest_discover_tests(${test})
endforeach ()

set_target_properties(${EXECUTABLES}
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                      "${ROOT_DIR}/dist/bin/Orthogonalizations")

set_target_properties(${EXECUTABLES} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
