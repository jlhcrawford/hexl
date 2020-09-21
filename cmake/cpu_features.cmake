# ******************************************************************************
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
# ******************************************************************************

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
