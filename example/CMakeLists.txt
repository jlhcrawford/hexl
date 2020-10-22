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

project(intel_lattice_example)
cmake_minimum_required(VERSION 3.12)

if (NOT INTEL_LATTICE_LIBRARY)
    message(FATAL_ERROR "INTEL_LATTICE_LIBRARY must be defined")
endif()
if (NOT INTEL_LATTICE_HEADER)
    message(FATAL_ERROR "INTEL_LATTICE_HEADER must be defined")
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
# TODO(fboemer): Support avx512 on macOS?
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
  message(STATUS "Setting LATTICE_HAS_AVX512DQ")
  add_definitions(-DLATTICE_HAS_AVX512DQ)
endif()


add_library(libintel_lattice STATIC IMPORTED)
set_property(TARGET libintel_lattice PROPERTY IMPORTED_LOCATION ${INTEL_LATTICE_LIBRARY})
include_directories(${INTEL_LATTICE_HEADER})

add_executable(example example.cpp)
target_link_libraries(example libintel_lattice)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
