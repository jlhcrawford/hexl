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


set(SRC ntt/ntt-internal.cpp
    number-theory/number-theory.cpp
    eltwise/eltwise-mult-mod.cpp
    eltwise/eltwise-add-mod.cpp
    eltwise/eltwise-fma.cpp
    eltwise/eltwise-cmp-add.cpp
    eltwise/eltwise-cmp-sub-mod.cpp
)

if (${HAS_AVX512F})
    list(APPEND SRC
        ntt/fwd-ntt-avx512.cpp
        ntt/inv-ntt-avx512.cpp
        eltwise/eltwise-mult-mod-avx512.cpp
        eltwise/eltwise-add-mod-avx512.cpp
        eltwise/eltwise-fma-avx512.cpp
        )
endif()

if (LATTICE_SHARED_LIB)
    add_library(intel_lattice_internal SHARED ${SRC})
else()
     add_library(intel_lattice_internal STATIC ${SRC})
endif()

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
target_compile_options(intel_lattice_internal PRIVATE -Wall -Wextra -Wno-unknown-pragmas -march=native -O3 -fomit-frame-pointer)

install(DIRECTORY ${LATTICE_INC_ROOT_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

if (LATTICE_SHARED_LIB)
    set_target_properties(intel_lattice_internal PROPERTIES OUTPUT_NAME "intel_lattice")
    install(TARGETS intel_lattice_internal DESTINATION ${CMAKE_INSTALL_LIBDIR})
else ()
    # Create combined library for intel-lattice containing all its dependencies
    if (${LATTICE_DEBUG})
        set(intel_lattice_lib ${CMAKE_BINARY_DIR}/libintel_lattice_debug.a)
        add_custom_target(intel_lattice_combined ALL
            COMMAND ar -x $<TARGET_FILE:intel_lattice_internal>
            COMMAND ar -x $<TARGET_FILE:libcpu_features>
            COMMAND ar -x $<TARGET_FILE:libgflags>
            COMMAND ar -x $<TARGET_FILE:logging>
            COMMAND ar -x $<TARGET_FILE:easylogging>
            COMMAND ar -qcs ${intel_lattice_lib} *.o
            COMMAND rm -f *.o
            BYPRODUCTS ${intel_lattice_lib}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS intel_lattice_internal libcpu_features logging easylogging libgflags
        )
    else()
        set(intel_lattice_lib ${CMAKE_BINARY_DIR}/libintel_lattice.a)
        # Don't export logging unless we are debugging
        add_custom_target(intel_lattice_combined ALL
            COMMAND ar -x $<TARGET_FILE:intel_lattice_internal>
            COMMAND ar -x $<TARGET_FILE:libcpu_features>
            # COMMAND ar -x $<TARGET_FILE:libgflags>
            # COMMAND ar -x $<TARGET_FILE:logging>
            COMMAND ar -qcs ${intel_lattice_lib} *.o
            COMMAND rm -f *.o
            BYPRODUCTS ${intel_lattice_lib}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS intel_lattice_internal libcpu_features #libgflags logging
        )
    endif()
    install(FILES ${intel_lattice_lib} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
