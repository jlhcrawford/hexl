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

set(SRC logging.cpp)

add_library(logging SHARED ${SRC})

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
