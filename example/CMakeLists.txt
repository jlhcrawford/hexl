# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

project(hexl_example LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 11)

find_package(HEXL 1.1.1
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
if (NOT TARGET HEXL::hexl)
    message(FATAL_ERROR "TARGET HEXL::hexl not found")
endif()

add_executable(example example.cpp)
target_link_libraries(example PRIVATE HEXL::hexl)
