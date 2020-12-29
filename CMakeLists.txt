# *****************************************************************************
# INTEL CONFIDENTIAL
# Copyright 2020 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software
# or the related documents without Intel's prior written permission.
# *****************************************************************************

project(intel_lattice LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.13)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
message(STATUS "CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

if(CMAKE_BUILD_TYPE)
  set(RELEASE_TYPES
      Debug
      Release
      RelWithDebInfo
      MinSizeRel)
  list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
  if(${INDEX_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel"
      )
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17) # TODO(fboemer): use C++14 or 11?
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_RPATH "\$ORIGIN")

#------------------------------------------------------------------------------
# Compiler options...
#------------------------------------------------------------------------------

option(LATTICE_DEBUG "Enable runtime debug checking; will yield much slower code " OFF)
option(LATTICE_BENCHMARK "Enable benchmarking" ON)
option(LATTICE_TESTING "Enables unit-tests" ON)
option(LATTICE_EXPORT "Enables export of intel_lattice target" OFF)
option(LATTICE_COVERAGE "Enables coverage for unit tests" OFF)
option(LATTICE_SHARED_LIB "Generate a shared library" OFF)
option(LATTICE_DOCS "Enable documentation building" OFF)
# Sanitizers
option(ENABLE_ADDRESS_SANITIZER "Compiles and links with Address Sanitizer" OFF)
option(ENABLE_THREAD_SANITIZER "Compiles and links with Thread Sanitizer" OFF)
option(ENABLE_UB_SANITIZER "Compiles and links with Undefined Behavior Sanitizer" OFF)

message(STATUS "CMAKE_BUILD_TYPE:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_COMPILER:         ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER:       ${CMAKE_CXX_COMPILER}")
message(STATUS "LATTICE_DEBUG:            ${LATTICE_DEBUG}")
message(STATUS "LATTICE_BENCHMARK:        ${LATTICE_BENCHMARK}")
message(STATUS "LATTICE_TESTING:          ${LATTICE_TESTING}")
message(STATUS "LATTICE_EXPORT:           ${LATTICE_EXPORT}")
message(STATUS "LATTICE_COVERAGE:         ${LATTICE_COVERAGE}")
message(STATUS "LATTICE_SHARED_LIB:       ${LATTICE_SHARED_LIB}")
message(STATUS "LATTICE_DOCS:             ${LATTICE_DOCS}")
message(STATUS "ENABLE_ADDRESS_SANITIZER: ${ENABLE_ADDRESS_SANITIZER}")
message(STATUS "ENABLE_THREAD_SANITIZER:  ${ENABLE_THREAD_SANITIZER}")
message(STATUS "ENABLE_UB_SANITIZER:      ${ENABLE_UB_SANITIZER}")

if (LATTICE_COVERAGE)
  if (NOT LATTICE_TESTING)
    message(FATAL_ERROR "LATTICE_COVERAGE enabled, but LATTICE_TESTING not enabled. Enable LATTICE_TESTING for best proper coverage.")
  endif()
  add_compile_options(--coverage -O0 -g)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_compile_options(-fkeep-inline-functions)
  add_link_options(--coverage)
  # link_libraries(gcov)
endif()

if (LATTICE_DEBUG)
  add_definitions(-DLATTICE_DEBUG)
endif()

function(check_flag FLAG OUTPUT_FLAG)
  execute_process(COMMAND cat /proc/cpuinfo OUTPUT_VARIABLE FLAG_OUT )
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Checking flag ${FLAG}...")
  string(FIND ${FLAG_OUT} "${FLAG}" INDEX_FOUND)
  if(NOT ${INDEX_FOUND} STREQUAL -1)
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo "Found. Setting ${OUTPUT_FLAG}")
    set(${OUTPUT_FLAG} "1" PARENT_SCOPE)
  else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo "Not found.")
  endif()
endfunction()

# Check for AVX512 support
if (NOT APPLE)
  check_flag("avx512f" HAS_AVX512F)
  check_flag("avx512dq" HAS_AVX512DQ)
  check_flag("avx512ifma" HAS_AVX512IFMA)

  if (HAS_AVX512IFMA)
    # Check compiler version
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      # recommend at least gcc 8.0
      if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
          message(WARNING "GCC version should be at least 8.0 for best performane on CPUs with AVX512IFMA support.")
          set(HAS_AVX512IFMA "0")
      endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # recommend at least clang 6.0
      if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
          message(WARNING "Clang version should be at least 6.0 for best performane on CPUs with AVX512IFMA support.")
          set(HAS_AVX512IFMA "0")
      endif()
    endif()
  endif()
endif()

if (HAS_AVX512IFMA)
  message(STATUS "Setting LATTICE_HAS_AVX512IFMA")
  add_definitions(-DLATTICE_HAS_AVX512IFMA)
endif()

if (HAS_AVX512F AND HAS_AVX512DQ)
  message(STATUS "Setting LATTICE_HAS_AVX512DQ")
  add_definitions(-DLATTICE_HAS_AVX512DQ)
endif()

if(ENABLE_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
endif()

if(ENABLE_UB_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fsanitize=implicit-conversion -fPIE")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fsanitize=implicit-conversion  -fPIE -pie")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fsanitize=implicit-conversion -fPIE -pie")
endif()

if(ENABLE_THREAD_SANITIZER)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
endif()

if(ENABLE_ADDRESS_SANITIZER OR ENABLE_THREAD_SANITIZER OR ENABLE_UB_SANITIZER)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
endif()

set(LATTICE_SRC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/intel-lattice)
set(LATTICE_INC_ROOT_DIR ${LATTICE_SRC_ROOT_DIR}/include) # Public headers


# ------------------------------------------------------------------------------
# Installation logic...
# ------------------------------------------------------------------------------

if(NOT CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(LINUX)
  include(GNUInstallDirs)
else()
  set(CMAKE_INSTALL_BINDIR "bin")
  set(CMAKE_INSTALL_INCLUDEDIR "include")
  set(CMAKE_INSTALL_DOCDIR "doc")
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

message(STATUS "CMAKE_INSTALL_LIBDIR:     ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_PREFIX:     ${CMAKE_INSTALL_PREFIX}")

set(LATTICE_FORWARD_CMAKE_ARGS
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
  -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
  -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=${CMAKE_EXPORT_COMPILE_COMMANDS}
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  -DCMAKE_CXX_FLAGS="-march=native"
  )

message(STATUS "LATTICE_FORWARD_CMAKE_ARGS ${LATTICE_FORWARD_CMAKE_ARGS}")

#------------------------------------------------------------------------------
# Third-party code...
#------------------------------------------------------------------------------
if (LATTICE_TESTING OR LATTICE_BENCHMARK OR LATTICE_DEBUG)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
endif()

if (LATTICE_TESTING)
  include(cmake/gtest.cmake)
endif()
if (LATTICE_BENCHMARK)
  include(cmake/gbenchmark.cmake)
endif()

include(cmake/easylogging.cmake)
include(cmake/gflags.cmake)
include(cmake/cpu_features.cmake)

#------------------------------------------------------------------------------
# Subfolders...
#------------------------------------------------------------------------------
add_subdirectory(logging)
add_subdirectory(intel-lattice)

if (LATTICE_BENCHMARK)
  add_subdirectory(benchmark)
  add_custom_target( bench COMMAND ./benchmark/bench_lattice DEPENDS bench_lattice)
endif()

if (LATTICE_TESTING)
  add_subdirectory(test)
  add_custom_target(test COMMAND ./test/unit-test DEPENDS unit-test)
endif()

if (LATTICE_DOCS)
  add_subdirectory(docs)
endif()


#------------------------------------------------------------------------------
# Config export...
#------------------------------------------------------------------------------

if (LATTICE_EXPORT)
  if (LATTICE_SHARED_LIB OR LATTICE_DEBUG)
    message(FATAL_ERROR "LATTICE_EXPORT=ON only valid when
                         LATTICE_SHARED_LIB=OFF and LATTICE_DEBUG=OFF ")
  endif()

  # Config
  set(INTEL_LATTICE_TARGET_FILENAME ${CMAKE_CURRENT_BINARY_DIR}/cmake/IntelLatticeTargets.cmake)
  set(INTEL_LATTICE_CONFIG_IN_FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/cmake/IntelLatticeConfig.cmake.in)
  set(INTEL_LATTICE_CONFIG_FILENAME ${CMAKE_CURRENT_SOURCE_DIR}/cmake/IntelLatticeConfig.cmake)
  set(INTEL_LATTICE_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/)

  # Create and install the CMake config and target file
  install(
      EXPORT IntelLatticeTargets
      DESTINATION ${INTEL_LATTICE_CONFIG_INSTALL_DIR}
  )

  include(CMakePackageConfigHelpers)
      configure_package_config_file(
          ${INTEL_LATTICE_CONFIG_IN_FILENAME} ${INTEL_LATTICE_CONFIG_FILENAME}
          INSTALL_DESTINATION ${INTEL_LATTICE_CONFIG_INSTALL_DIR}
      )

  install(
      TARGETS intel_lattice
      EXPORT IntelLatticeTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      )

  install( FILES ${INTEL_LATTICE_CONFIG_FILENAME}
           DESTINATION ${INTEL_LATTICE_CONFIG_INSTALL_DIR})

  export(EXPORT IntelLatticeTargets
         FILE ${INTEL_LATTICE_TARGET_FILENAME})

endif()


add_custom_target( check COMMAND pre-commit install && pre-commit run --all-files)
