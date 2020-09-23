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

set(SRC main.cpp
    test-number-theory.cpp
    test-ntt.cpp
    test-eltwise-mult-mod.cpp
    test-eltwise-fma.cpp
    test-eltwise-cmp-add.cpp
    test-eltwise-cmp-sub-mod.cpp
)

if (${HAS_AVX512F})
    list(APPEND SRC test-avx512-util.cpp)
endif()

add_executable(unit-test ${SRC})

target_compile_options(unit-test PRIVATE -Wall -Wextra -march=native)

target_include_directories(unit-test PRIVATE ".")
target_link_libraries(unit-test PRIVATE libgtest intel_lattice_internal)
target_link_libraries(unit-test PRIVATE Threads::Threads)
target_link_libraries(unit-test PRIVATE logging)

# Make sure that public include folder doesn't use private headers
# and that public headers are self-contained
add_executable(test-public-api test-public-api.cpp)
target_include_directories(test-public-api PRIVATE ${LATTICE_INC_ROOT_DIR})
