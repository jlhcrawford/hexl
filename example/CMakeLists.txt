# *****************************************************************************
# INTEL CONFIDENTIAL
# Copyright 2020 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software
# or the related documents without Intel's prior written permission.
# ****************************************************************************

project(intel_lattice_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.12)

if (NOT INTEL_LATTICE_LIBRARY)
    message(FATAL_ERROR "INTEL_LATTICE_LIBRARY must be defined")
endif()
if (NOT INTEL_LATTICE_HEADER)
    message(FATAL_ERROR "INTEL_LATTICE_HEADER must be defined")
endif()

add_library(libintel_lattice STATIC IMPORTED)
set_property(TARGET libintel_lattice PROPERTY IMPORTED_LOCATION ${INTEL_LATTICE_LIBRARY})
include_directories(${INTEL_LATTICE_HEADER})

add_executable(example example.cpp)
target_link_libraries(example libintel_lattice)
