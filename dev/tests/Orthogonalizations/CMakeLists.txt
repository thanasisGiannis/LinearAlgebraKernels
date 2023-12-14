cmake_minimum_required(VERSION 3.5)

project(tests LANGUAGES C CXX CUDA)

set(EXECUTABLES TestHouseholder
                TestCholesky
                TestMGS) # update with new tests here


include(GoogleTest)
enable_testing()
  foreach (test ${EXECUTABLES})
    add_executable(${test} ${test}.cu)
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