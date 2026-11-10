cmake_minimum_required(VERSION 3.28...4.4)

# The version is defined once, in pyproject.toml (FR-22).
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _qkern_version_line
     REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _qkern_version "${_qkern_version_line}")
if(NOT _qkern_version)
  message(FATAL_ERROR "Could not read [project].version from pyproject.toml")
endif()

project(
  qkern
  VERSION ${_qkern_version}
  DESCRIPTION "Streaming and rolling time-series kernels"
  LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
# No C++20 modules here; skip dependency scanning (it needs clang-scan-deps with Clang).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include(GNUInstallDirs)
include(cmake/warnings.cmake)

option(QKERN_BUILD_TESTS "Build the GoogleTest suite (downloads GoogleTest through CPM)" OFF)
if(DEFINED SKBUILD)
  set(_qkern_python_default ON)
else()
  set(_qkern_python_default OFF)
endif()
option(QKERN_BUILD_PYTHON "Build the _qkern nanobind extension" ${_qkern_python_default})
option(QKERN_STUBGEN "Generate _qkern.pyi with nanobind stubgen after building the module" OFF)

# ---------------------------------------------------------------------------------------
# L1: header-only kernel core
# ---------------------------------------------------------------------------------------
configure_file(cmake/version.hpp.in "${PROJECT_BINARY_DIR}/generated/include/qkern/version.hpp"
               @ONLY)

add_library(qkern_core INTERFACE)
add_library(qkern::core ALIAS qkern_core)
target_include_directories(
  qkern_core INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated/include>
                       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(qkern_core INTERFACE cxx_std_20)
qkern_set_fp_flags(qkern_core)

# ---------------------------------------------------------------------------------------
# L2: nanobind extension (built by scikit-build-core; never downloads anything)
# ---------------------------------------------------------------------------------------
if(QKERN_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module
                                                ${SKBUILD_SABI_COMPONENT})
  if(NOT DEFINED nanobind_DIR AND NOT DEFINED SKBUILD)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE nanobind_DIR COMMAND_ERROR_IS_FATAL ANY)
  endif()
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(
    _qkern
    STABLE_ABI
    NB_STATIC
    LTO
    src/bindings/module.cpp
    src/bindings/bind_rolling.cpp)
  target_link_libraries(_qkern PRIVATE qkern::core)
  target_include_directories(_qkern PRIVATE src/bindings)
  qkern_set_warnings(_qkern)

  # nanobind's and Python's headers are third-party: compile them as system headers so the
  # strict warning flags above apply only to this repo's code.
  foreach(_nb_target nanobind-static nanobind-static-abi3 nanobind-static-ft)
    if(TARGET ${_nb_target})
      set_target_properties(${_nb_target} PROPERTIES SYSTEM ON CXX_CLANG_TIDY "")
    endif()
  endforeach()

  # Build metadata for qkern.build_info() (FR-30).
  find_package(Git QUIET)
  set(QKERN_GIT_SHA "unknown")
  if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --short=7 HEAD
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE QKERN_GIT_SHA
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT QKERN_GIT_SHA)
      set(QKERN_GIT_SHA "unknown")
    endif()
  endif()
  if(MSVC)
    set(_qkern_fp_flags "/fp:precise")
  else()
    set(_qkern_fp_flags "-ffp-contract=off")
  endif()
  if(CMAKE_BUILD_TYPE)
    set(_qkern_build_type "${CMAKE_BUILD_TYPE}")
  else()
    set(_qkern_build_type "multi-config")
  endif()
  target_compile_definitions(
    _qkern
    PRIVATE QKERN_GIT_SHA="${QKERN_GIT_SHA}"
            QKERN_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
            QKERN_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
            QKERN_BUILD_TYPE="${_qkern_build_type}"
            QKERN_FP_FLAGS="${_qkern_fp_flags}"
            QKERN_NANOBIND_VERSION="${nanobind_VERSION}")

  if(DEFINED SKBUILD)
    install(TARGETS _qkern LIBRARY DESTINATION qkern)
  endif()

  if(QKERN_STUBGEN AND NOT CMAKE_CROSSCOMPILING)
    nanobind_add_stub(
      _qkern_stub
      MODULE _qkern
      INCLUDE_PRIVATE
      OUTPUT "${PROJECT_BINARY_DIR}/_qkern.pyi"
      PYTHON_PATH $<TARGET_FILE_DIR:_qkern>
      DEPENDS _qkern)
  endif()
endif()

# ---------------------------------------------------------------------------------------
# Tests (GoogleTest through CPM; off by default so Python builds never download)
# ---------------------------------------------------------------------------------------
if(QKERN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
