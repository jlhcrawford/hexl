# *****************************************************************************
# INTEL CONFIDENTIAL
# Copyright 2020 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software
# or the related documents without Intel's prior written permission.
# *****************************************************************************

set(SRC logging.cpp)

add_library(logging STATIC ${SRC})

set_target_properties(logging PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(logging PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(logging PUBLIC easylogging libgflags)

set(LOGGING_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/logging
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

install(TARGETS logging LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} )
