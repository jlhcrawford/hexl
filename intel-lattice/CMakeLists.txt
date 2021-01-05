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
    add_library(intel_lattice SHARED ${SRC})
else()
     add_library(intel_lattice STATIC ${SRC})
endif()

set_property(TARGET intel_lattice PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(intel_lattice
    PRIVATE ${LATTICE_SRC_ROOT_DIR}                               # Private headers
    PUBLIC  $<BUILD_INTERFACE:${LATTICE_INC_ROOT_DIR}>            # Public headers
    PUBLIC  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/> # Public headers
    )

target_compile_options(intel_lattice PRIVATE -Wall -Wextra -Wno-unknown-pragmas -march=native -O3 -fomit-frame-pointer)

install(DIRECTORY ${LATTICE_INC_ROOT_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

if (LATTICE_SHARED_LIB)
    target_link_libraries(intel_lattice PRIVATE
        cpu_features
        gflags
        logging
    )
else ()
    # For static library, we include all the dependencies for intel-lattice in
    # the libintel_lattice.a.
    #  For proper export of IntelLatticeConfig.cmake / IntelLatticeTargts.cmake,
    # we avoid explicitly linking dependencies via target_link_libraries, since
    # this would add dependencies to the exported intel_lattice target.
    add_dependencies(intel_lattice cpu_features gflags logging)

    # Manually add cpu_features include directory
    target_include_directories(intel_lattice
      PRIVATE $<TARGET_PROPERTY:cpu_features,INTERFACE_INCLUDE_DIRECTORIES>)

    # Manually add logging include directory
    target_include_directories(intel_lattice
      PRIVATE $<TARGET_PROPERTY:logging,INTERFACE_INCLUDE_DIRECTORIES>)

    if (${LATTICE_DEBUG})
        set_target_properties(intel_lattice PROPERTIES OUTPUT_NAME "intel_lattice_debug")
        set(intel_lattice_lib_filename ${CMAKE_CURRENT_BINARY_DIR}/libintel_lattice_debug.a)
        add_custom_command(TARGET intel_lattice POST_BUILD
            COMMAND ar -x $<TARGET_FILE:intel_lattice>
            COMMAND ar -x $<TARGET_FILE:cpu_features>
            COMMAND ar -x $<TARGET_FILE:gflags>
            COMMAND ar -x $<TARGET_FILE:logging>
            COMMAND ar -x $<TARGET_FILE:easylogging>
            COMMAND ar -qcs ${intel_lattice_lib_filename} *.o
            COMMAND rm -f *.o
            BYPRODUCTS ${intel_lattice_lib_filename}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS intel_lattice cpu_features logging easylogging gflags
        )
    else()
        set(intel_lattice_lib_filename ${CMAKE_CURRENT_BINARY_DIR}/libintel_lattice.a)
        # Don't export logging unless we are debugging
        add_custom_command(TARGET intel_lattice POST_BUILD
            COMMAND ar -x $<TARGET_FILE:intel_lattice>
            COMMAND ar -x $<TARGET_FILE:cpu_features>
            # COMMAND ar -x $<TARGET_FILE:gflags>
            # COMMAND ar -x $<TARGET_FILE:logging>
            COMMAND ar -qcs ${intel_lattice_lib_filename} *.o
            COMMAND rm -f *.o
            BYPRODUCTS ${intel_lattice_lib_filename}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS intel_lattice cpu_features #gflags logging
        )
    endif()
endif()

install(TARGETS intel_lattice DESTINATION ${CMAKE_INSTALL_LIBDIR})
