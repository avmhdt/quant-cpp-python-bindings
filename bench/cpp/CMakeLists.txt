include("${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  GIT_TAG v1.10.0
  VERSION 1.10.0
  SYSTEM YES
  OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF"
          "BENCHMARK_ENABLE_GTEST_TESTS OFF" "BENCHMARK_ENABLE_WERROR OFF")

foreach(_bm_target benchmark benchmark_main)
  if(TARGET ${_bm_target})
    set_target_properties(${_bm_target} PROPERTIES CXX_CLANG_TIDY "")
  endif()
endforeach()

add_executable(qkern_bench bench_kernels.cpp)
target_link_libraries(qkern_bench PRIVATE qkern::core benchmark::benchmark_main)
qkern_set_warnings(qkern_bench)
