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

# Enable ExternalProject CMake module
include(ExternalProject)

# ------------------------------------------------------------------------------
# Download and install EasyLogging ...
# ------------------------------------------------------------------------------

set(EASYLOGGING_GIT_REPO_URL https://github.com/amrayn/easyloggingpp.git)
set(EASYLOGGING_GIT_LABEL master)
set(EASYLOGGING_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_easy_logging)
set(EASYLOGGING_SRC_DIR ${EASYLOGGING_PREFIX}/src)

# Note, pass -DELPP_THREAD_SAFE to ensure logging is thread-safe. This will add pthread linking requirement
# Currently, intel-lattice is single-threaded, so we omit this to remove dependency on pthread
set(EASYLOGGING_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DELPP_CUSTOM_COUT=std::cerr -DELPP_STL_LOGGING -DELPP_LOG_STD_ARRAY -DELPP_LOG_UNORDERED_MAP -DELPP_LOG_UNORDERED_SET -DELPP_NO_LOG_TO_FILE -DELPP_DISABLE_DEFAULT_CRASH_HANDLING -DELPP_WINSOCK2")

ExternalProject_Add(
  ext_easylogging
  PREFIX ${EASYLOGGING_PREFIX}
  GIT_REPOSITORY ${EASYLOGGING_GIT_REPO_URL}
  GIT_TAG ${EASYLOGGING_GIT_LABEL}
  CMAKE_ARGS ${LATTICE_FORWARD_CMAKE_ARGS}
    -DCMAKE_CXX_FLAGS=${EASYLOGGING_CXX_FLAGS}
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
  EXCLUDE_FROM_ALL TRUE
  BUILD_BYPRODUCTS ${EASYLOGGING_SRC_DIR}/ext_easylogging/src/easylogging++.cc
)

# ------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_easylogging SOURCE_DIR BINARY_DIR)

add_library(easylogging ${SOURCE_DIR}/src/easylogging++.cc)
target_include_directories(easylogging PUBLIC ${SOURCE_DIR}/src)
add_dependencies(easylogging ext_easylogging)

install(DIRECTORY ${SOURCE_DIR}/src/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")
