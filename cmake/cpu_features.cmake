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

# Enable ExternalProject CMake module
include(ExternalProject)

# ------------------------------------------------------------------------------
# Download and install cpu_features ...
# ------------------------------------------------------------------------------

set(CPU_FEATURES_GIT_REPO_URL https://github.com/google/cpu_features.git)
set(CPU_FEATURES_GIT_LABEL master)
set(CPU_FEATURES_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_cpu_features)
set(CPU_FEATURES_SRC_DIR ${CPU_FEATURES_PREFIX}/src)

ExternalProject_Add(
  ext_cpu_features
  PREFIX ${CPU_FEATURES_PREFIX}
  GIT_REPOSITORY ${CPU_FEATURES_GIT_REPO_URL}
  GIT_TAG ${CPU_FEATURES_GIT_LABEL}
  CMAKE_ARGS ${LATTICE_FORWARD_CMAKE_ARGS}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DBUILD_PIC=ON
    -DCMAKE_INSTALL_PREFIX=${CPU_FEATURES_PREFIX}
  UPDATE_COMMAND ""
  EXCLUDE_FROM_ALL TRUE
)

# ------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_cpu_features SOURCE_DIR BINARY_DIR)

add_library(libcpu_features STATIC IMPORTED)
add_dependencies(libcpu_features ext_cpu_features)

if(NOT IS_DIRECTORY ${CPU_FEATURES_PREFIX}/include)
  file(MAKE_DIRECTORY ${CPU_FEATURES_PREFIX}/include)
endif()

set_target_properties(libcpu_features
                      PROPERTIES IMPORTED_LOCATION
                      ${BINARY_DIR}/libcpu_features.a)
set_target_properties(libcpu_features
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                      ${CPU_FEATURES_PREFIX}/include)
