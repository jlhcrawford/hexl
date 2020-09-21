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


set(SRC ntt/ntt.cpp
        number-theory/number-theory.cpp
        poly/poly-mult.cpp
        poly/poly-fma.cpp
        poly/poly-cmp-add.cpp
        poly/poly-cmp-sub-mod.cpp
)

if (${HAS_AVX512F})
    list(APPEND SRC
        ntt/ntt-avx512.cpp
        poly/poly-mult-avx512.cpp
        poly/poly-fma-avx512.cpp
        )
endif()

add_library(intel_lattice_internal STATIC ${SRC})
set_property(TARGET intel_lattice_internal PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(intel_lattice_internal PUBLIC
        ${LATTICE_SRC_ROOT_DIR}
        ${LATTICE_INC_ROOT_DIR}/intel-lattice
        ${LATTICE_INC_ROOT_DIR})

target_link_libraries(intel_lattice_internal PUBLIC
        libcpu_features
        libgflags
        logging
        )
target_compile_options(intel_lattice_internal PRIVATE -Wall -Wextra -Wno-unknown-pragmas -march=native -O3)

install(TARGETS intel_lattice_internal LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )

install(DIRECTORY ${LATTICE_INC_ROOT_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

# Create combined library for intel-lattice containing all its dependencies
if (${LATTICE_DEBUG})
        set(intel_lattice_lib ${CMAKE_BINARY_DIR}/libintel_lattice_debug.a)
else()
        set(intel_lattice_lib ${CMAKE_BINARY_DIR}/libintel_lattice.a)
endif()

add_custom_target(intel_lattice_combined ALL
        COMMAND ar -x $<TARGET_FILE:intel_lattice_internal>
        COMMAND ar -x $<TARGET_FILE:libcpu_features>
        # COMMAND ar -x $<TARGET_FILE:libgflags>
        # COMMAND ar -x $<TARGET_FILE:logging>
        COMMAND ar -qcs ${intel_lattice_lib} *.o
        COMMAND rm -f *.o
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        # DEPENDS intel_lattice_internal libcpu_features libgflags logging
        )
