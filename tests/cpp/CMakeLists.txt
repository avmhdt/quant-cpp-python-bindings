include("${PROJECT_SOURCE_DIR}/cmake/CPM.cmake")

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.18.0
  VERSION 1.18.0
  SYSTEM YES
  OPTIONS "INSTALL_GTEST OFF" "BUILD_GMOCK OFF" "gtest_force_shared_crt ON")

include(GoogleTest)

# Third-party code is not linted.
foreach(_gt_target gtest gtest_main)
  if(TARGET ${_gt_target})
    set_target_properties(${_gt_target} PROPERTIES CXX_CLANG_TIDY "")
  endif()
endforeach()

add_executable(qkern_tests test_views.cpp test_rolling_mean.cpp)
target_link_libraries(qkern_tests PRIVATE qkern::core GTest::gtest_main)
qkern_set_warnings(qkern_tests)
gtest_discover_tests(qkern_tests DISCOVERY_MODE PRE_TEST)
