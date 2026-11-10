# A C++ project using qkern::core (FR-16). Two ways to get it:
#
#   find_package (after `cmake --install` of qkern):
#     cmake -S examples/cpp_consumer -B build/consumer -DCMAKE_PREFIX_PATH=<qkern prefix>
#   CPM (fetches the repo, or uses a local checkout):
#     cmake -S examples/cpp_consumer -B build/consumer -DQKERN_CONSUMER_USE_CPM=ON \
#           [-DCPM_qkern_SOURCE=<path to a qkern checkout>]
cmake_minimum_required(VERSION 3.28...4.4)
project(qkern_cpp_consumer LANGUAGES CXX)

option(QKERN_CONSUMER_USE_CPM "Get qkern through CPM.cmake instead of find_package" OFF)

if(QKERN_CONSUMER_USE_CPM)
  include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/CPM.cmake")
  CPMAddPackage(
    NAME qkern
    GITHUB_REPOSITORY avmhdt/quant-cpp-python-bindings
    GIT_TAG main)
else()
  find_package(qkern 0.1 CONFIG REQUIRED)
endif()

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE qkern::core)

enable_testing()
add_test(NAME consumer COMMAND consumer)
