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

set(SRC main.cpp
    test-number-theory.cpp
    test-ntt.cpp
    test-eltwise-mult-mod.cpp
    test-eltwise-add-mod.cpp
    test-eltwise-fma.cpp
    test-eltwise-cmp-add.cpp
    test-eltwise-cmp-sub-mod.cpp
)

if (${HAS_AVX512F})
    list(APPEND SRC test-avx512-util.cpp)
endif()

add_executable(unit-test ${SRC})

target_compile_options(unit-test PRIVATE -Wall -Wextra -march=native)

target_include_directories(unit-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LATTICE_SRC_ROOT_DIR} # Private headers
)
target_link_libraries(unit-test PRIVATE gtest intel_lattice)
target_link_libraries(unit-test PRIVATE Threads::Threads)
target_link_libraries(unit-test PRIVATE logging)

# Make sure that public include folder doesn't use private headers
# and that public headers are self-contained
add_executable(test-public-api test-public-api.cpp)
target_include_directories(test-public-api PRIVATE ${LATTICE_INC_ROOT_DIR})
