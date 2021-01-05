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
    bench-ntt.cpp
    bench-eltwise-add.cpp
    bench-eltwise-fma.cpp
    bench-eltwise-mult.cpp
    bench-eltwise-cmp-add.cpp
    bench-eltwise-cmp-sub-mod.cpp
    )

add_executable(bench_lattice ${SRC})

target_include_directories(bench_lattice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LATTICE_SRC_ROOT_DIR} # Private headers
    )

target_link_libraries(bench_lattice PRIVATE intel_lattice)
target_link_libraries(bench_lattice PRIVATE gbenchmark gflags Threads::Threads)
target_link_libraries(bench_lattice PRIVATE logging)

target_compile_options(bench_lattice PRIVATE -Wall -Wextra -march=native -O3)
