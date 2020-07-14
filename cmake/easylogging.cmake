# ******************************************************************************
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
# ******************************************************************************

# Enable ExternalProject CMake module
include(ExternalProject)

# ------------------------------------------------------------------------------
# Download and install EasyLogging ...
# ------------------------------------------------------------------------------

set(EASYLOGGING_GIT_REPO_URL https://github.com/amrayn/easyloggingpp.git)
set(EASYLOGGING_GIT_LABEL master)
set(EASYLOGGING_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_easy_logging)
set(EASYLOGGING_SRC_DIR ${EASYLOGGING_PREFIX}/src)

set(EASYLOGGING_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DELPP_THREAD_SAFE -DELPP_CUSTOM_COUT=std::cerr -DELPP_STL_LOGGING -DELPP_LOG_STD_ARRAY -DELPP_LOG_UNORDERED_MAP -DELPP_LOG_UNORDERED_SET -DELPP_NO_LOG_TO_FILE -DELPP_DISABLE_DEFAULT_CRASH_HANDLING -DELPP_WINSOCK2")

ExternalProject_Add(
  ext_easylogging
  PREFIX ${EASYLOGGING_PREFIX}
  GIT_REPOSITORY ${EASYLOGGING_GIT_REPO_URL}
  GIT_TAG ${EASYLOGGING_GIT_LABEL}
  CMAKE_ARGS -DCMAKE_CXX_FLAGS=${EASYLOGGING_CXX_FLAGS}
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
