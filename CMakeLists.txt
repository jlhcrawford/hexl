# *****************************************************************************
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# *****************************************************************************

project(ntt-avx CXX)
cmake_minimum_required(VERSION 3.12)

# Global project name
project(intel_lattice LANGUAGES C CXX)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
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
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17) # TODO(fboemer): use C++14 or 11?
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT ${CMAKE_INSTALL_PREFIX})
  set(${CMAKE_INSTALL_PREFIX} ./)
endif()

# Create compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INSTALL_RPATH "\$ORIGIN")


#------------------------------------------------------------------------------
# Compiler options...
#------------------------------------------------------------------------------

option(LATTICE_DEBUG "Enable runtime debug checking; will yield much slower code " OFF)
option(LATTICE_BENCHMARK "Enable benchmarking" ON)

message(STATUS "CMAKE_C_COMPILER:         ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER:       ${CMAKE_CXX_COMPILER}")
message(STATUS "LATTICE_DEBUG:            ${LATTICE_DEBUG}")
message(STATUS "LATTICE_BENCHMARK:        ${LATTICE_BENCHMARK}")
message(STATUS "ENABLE_ADDRESS_SANITIZER: ${ENABLE_ADDRESS_SANITIZER}")
message(STATUS "ENABLE_THREAD_SANITIZER:  ${ENABLE_THREAD_SANITIZER}")
message(STATUS "ENABLE_UB_SANITIZER:      ${ENABLE_UB_SANITIZER}")

# Sanitizers
option(ENABLE_ADDRESS_SANITIZER "Compiles and links with Address Sanitizer" OFF)
option(ENABLE_THREAD_SANITIZER "Compiles and links with Thread Sanitizer" OFF)
option(ENABLE_UB_SANITIZER "Compiles and links with Undefined Behavior Sanitizer" OFF)


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
# TODO(fboemer): Support avx512 on macOS
if (NOT APPLE)
  check_flag("avx512f" HAS_AVX512F)
  check_flag("avx512dq" HAS_AVX512DQ)
  check_flag("avx512ifma" HAS_AVX512IFMA)
endif()

if (HAS_AVX512IFMA)
message(STATUS "Setting LATTICE_HAS_AVX512IFMA")
  add_definitions(-DLATTICE_HAS_AVX512IFMA)
endif()

if (HAS_AVX512F AND HAS_AVX512DQ)
  message(STATUS "Setting LATTICE_HAS_AVX512F")
  add_definitions(-DLATTICE_HAS_AVX512F)
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
set(LATTICE_INC_ROOT_DIR ${LATTICE_SRC_ROOT_DIR}/include)


# ------------------------------------------------------------------------------
# Installation logic...
# ------------------------------------------------------------------------------

if(LINUX)
  include(GNUInstallDirs)
else()
  set(CMAKE_INSTALL_BINDIR "bin")
  set(CMAKE_INSTALL_INCLUDEDIR "include")
  set(CMAKE_INSTALL_DOCDIR "doc")
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
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
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(cmake/gtest.cmake)
if (${LATTICE_BENCHMARK})
  include(cmake/gbenchmark.cmake)
endif()
include(cmake/easylogging.cmake)
include(cmake/gflags.cmake)

#------------------------------------------------------------------------------
# Subfolders...
#------------------------------------------------------------------------------
add_subdirectory(logging)
add_subdirectory(intel-lattice)
if (${LATTICE_BENCHMARK})
  add_subdirectory(benchmark)
endif()
add_subdirectory(test)

add_custom_target( test COMMAND ./test/unit-test DEPENDS unit-test)
add_custom_target( bench COMMAND ./benchmark/bench_lattice DEPENDS bench_lattice)

add_custom_target( check COMMAND pre-commit install && pre-commit run --all-files DEPENDS test)
