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

# Enable ExternalProject CMake module
include(ExternalProject)

# ------------------------------------------------------------------------------
# Download and install GFlags ...
# ------------------------------------------------------------------------------

set(GFLAGS_GIT_REPO_URL https://github.com/gflags/gflags.git)
set(GFLAGS_GIT_LABEL v2.2.2)

ExternalProject_Add(
  ext_gflags
  PREFIX ext_gflags
  GIT_REPOSITORY ${GFLAGS_GIT_REPO_URL}
  GIT_TAG ${GFLAGS_GIT_LABEL}
  CMAKE_ARGS ${LATTICE_FORWARD_CMAKE_ARGS}
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
  EXCLUDE_FROM_ALL TRUE)

# ------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gflags SOURCE_DIR BINARY_DIR)

add_library(libgflags STATIC IMPORTED)
add_dependencies(libgflags ext_gflags)

if(NOT IS_DIRECTORY ${BINARY_DIR}/include)
  file(MAKE_DIRECTORY ${BINARY_DIR}/include)
endif()

set_target_properties(libgflags
                      PROPERTIES IMPORTED_LOCATION
                      ${BINARY_DIR}/lib/libgflags.a)
set_target_properties(libgflags
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                      ${BINARY_DIR}/include)
