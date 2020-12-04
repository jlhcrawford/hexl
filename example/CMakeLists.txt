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
