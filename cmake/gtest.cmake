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
# Download and install GoogleTest ...
# ------------------------------------------------------------------------------

set(GTEST_GIT_REPO_URL https://github.com/google/googletest.git)
set(GTEST_GIT_LABEL release-1.10.0)

ExternalProject_Add(
  ext_gtest
  PREFIX ext_gtest
  GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
  GIT_TAG ${GTEST_GIT_LABEL}
  CMAKE_ARGS ${LATTICE_FORWARD_CMAKE_ARGS}
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
  EXCLUDE_FROM_ALL TRUE)

# ------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gtest SOURCE_DIR BINARY_DIR)

add_library(gtest INTERFACE)
add_dependencies(gtest ext_gtest)

target_include_directories(gtest SYSTEM
                           INTERFACE ${SOURCE_DIR}/googletest/include)
target_link_libraries(gtest
                      INTERFACE ${BINARY_DIR}/lib/libgtest.a)
