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

include(ExternalProject)

set(GBENCHMARK_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_gbenchmark)

set(GBENCHMARK_SRC_DIR ${GBENCHMARK_PREFIX}/src/ext_gbenchmark/)
set(GBENCHMARK_BUILD_DIR ${GBENCHMARK_PREFIX}/src/ext_gbenchmark-build/)
set(GBENCHMARK_REPO_URL https://github.com/google/benchmark.git)
set(GBENCHMARK_GIT_TAG master)

set(GBENCHMARK_PATHS ${GBENCHMARK_SRC_DIR} ${GBENCHMARK_BUILD_DIR}/src/libbenchmark.a)

ExternalProject_Add(
  ext_gbenchmark
  GIT_REPOSITORY ${GBENCHMARK_REPO_URL}
  GIT_TAG ${GBENCHMARK_GIT_TAG}
  PREFIX  ${GBENCHMARK_PREFIX}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
             -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
             -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
             -DCMAKE_BUILD_TYPE=Release
  BUILD_BYPRODUCTS ${GBENCHMARK_PATHS}
  # Skip updates
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
)

add_library(gbenchmark INTERFACE)
add_dependencies(gbenchmark ext_gbenchmark)

ExternalProject_Get_Property(ext_gbenchmark SOURCE_DIR BINARY_DIR)

target_link_libraries(gbenchmark INTERFACE ${GBENCHMARK_BUILD_DIR}/src/libbenchmark.a)

target_include_directories(gbenchmark SYSTEM
                                    INTERFACE ${GBENCHMARK_SRC_DIR}/include)
