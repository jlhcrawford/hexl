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
project(ntt_avx LANGUAGES C CXX)

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



#------------------------------------------------------------------------------
# Compiler options...
#------------------------------------------------------------------------------

option(NTT_DEBUG "Enable runtime debug checking; will yield much slower code " OFF)
option(NTT_AVX512 "Enable optimized NTT using AVX512 instruction set" ON)
option(NTT_BENCHMARK "Enable benchmarking" ON)

message(STATUS "CMAKE_C_COMPILER:   ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "NTT_DEBUG:          ${NTT_DEBUG}")
message(STATUS "NTT_AVX512:         ${NTT_AVX512}")
message(STATUS "NTT_BENCHMARK:      ${NTT_BENCHMARK}")

if (${NTT_AVX512})
  check_cxx_source_compiles("#include <immintrin.h>\nint main() { return 0;}" NTT_HAS_AVX)
  if (NOT ${NTT_HAS_AVX})
    message( FATAL_ERROR "Error with AVX512 support")
  endif()
endif()

#------------------------------------------------------------------------------
# Third-party code...
#------------------------------------------------------------------------------
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(cmake/gtest.cmake)
if (${NTT_BENCHMARK})
  include(cmake/gbenchmark.cmake)
endif()
include(cmake/easylogging.cmake)
include(cmake/gflags.cmake)

#------------------------------------------------------------------------------
# Subfolders...
#------------------------------------------------------------------------------
add_subdirectory(src)
if (${NTT_BENCHMARK})
  add_subdirectory(benchmark)
endif()
add_subdirectory(test)

add_custom_target( test COMMAND ./test/unit-test DEPENDS unit-test)
add_custom_target( bench COMMAND ./benchmark/bench_ntt DEPENDS bench_ntt)

add_custom_target( check COMMAND pre-commit install && pre-commit run --all-files DEPENDS test)
