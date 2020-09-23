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
