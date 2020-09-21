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
    bench-ntt.cpp
    bench-poly-fma.cpp
    bench-poly-mult.cpp
    bench-poly-cmp-add.cpp
    bench-poly-cmp-sub-mod.cpp
    )

add_executable(bench_lattice ${SRC})

target_include_directories(bench_lattice PRIVATE ".")
target_link_libraries(bench_lattice PRIVATE intel_lattice_internal)
target_link_libraries(bench_lattice PRIVATE libgbenchmark libgflags Threads::Threads)
target_link_libraries(bench_lattice PRIVATE logging)

target_compile_options(bench_lattice PRIVATE -Wall -Wextra -march=native -O3)
