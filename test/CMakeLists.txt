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

set(SRC main.cpp test_number_theory.cpp test_ntt.cpp test_avx512_util.cpp)

add_executable(unit-test ${SRC})

target_compile_options(unit-test PRIVATE -Wall -Wextra -march=native)

target_include_directories(unit-test PRIVATE ".")
target_link_libraries(unit-test PRIVATE libgtest intel_lattice)
target_link_libraries(unit-test PRIVATE Threads::Threads)
target_link_libraries(unit-test PRIVATE logging)
