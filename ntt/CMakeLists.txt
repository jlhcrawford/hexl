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

set(NTT_SRC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (${NTT_AVX512})
    set(SRC number-theory.cpp ntt-avx512.cpp)
else()
    set(SRC number-theory.cpp ntt.cpp)
endif()

add_library(ntt SHARED ${SRC})

target_include_directories(ntt PUBLIC ${NTT_SRC_ROOT_DIR})
target_link_libraries(ntt PUBLIC libgflags logging libgflags Threads::Threads)
target_compile_options(ntt PRIVATE -Wall -Wextra -march=native)

install(DIRECTORY ${NTT_SRC_ROOT_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ntt
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

install(TARGETS ntt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
