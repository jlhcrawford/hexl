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

set(LATTICE_SRC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SRC ntt/ntt.cpp
        number-theory/number-theory.cpp
        poly/poly-mult.cpp
)

if (${HAS_AVX512F})
    list(APPEND SRC ntt/ntt-avx512.cpp)
endif()

add_library(intel_lattice SHARED ${SRC})

target_include_directories(intel_lattice PUBLIC ${LATTICE_SRC_ROOT_DIR})
target_link_libraries(intel_lattice PUBLIC libgflags logging libgflags Threads::Threads)
target_compile_options(intel_lattice PRIVATE -Wall -Wextra -march=native)

message("CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")


install(TARGETS intel_lattice LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )

install(DIRECTORY ${LATTICE_SRC_ROOT_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/intel-lattice
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")
