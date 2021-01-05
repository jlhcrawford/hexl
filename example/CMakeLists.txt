# *****************************************************************************
# INTEL CONFIDENTIAL
# Copyright 2020-2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software
# or the related documents without Intel's prior written permission.
# ****************************************************************************

project(intel_lattice_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.12)

find_package(IntelLattice
    HINTS ${INTEL_LATTICE_HINT_DIR}
    REQUIRED)
if (NOT TARGET intel_lattice)
    FATAL_ERROR("TARGET intel_lattice not found")
endif()

add_executable(example example.cpp)
target_link_libraries(example intel_lattice)
