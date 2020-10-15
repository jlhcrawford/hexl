//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-mult-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_EltwiseMultModNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  std::vector<uint64_t> input1(input_size, 1);
  std::vector<uint64_t> input2(input_size, 2);

  for (auto _ : state) {
    EltwiseMultModNative(input1.data(), input2.data(), input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseMultModNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef LATTICE_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseMultModAVX512Float(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;

  std::vector<uint64_t> input1(input_size, 1);
  std::vector<uint64_t> input2(input_size, 2);

  for (auto _ : state) {
    EltwiseMultModAVX512Float(input1.data(), input2.data(), input_size,
                              modulus);
  }
}

BENCHMARK(BM_EltwiseMultModAVX512Float)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});
#endif

//=================================================================

#ifdef LATTICE_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseMultModAVX512Int(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 1152921504606877697;

  std::vector<uint64_t> input1(input_size, 1);
  std::vector<uint64_t> input2(input_size, 2);

  for (auto _ : state) {
    EltwiseMultModAVX512Int(input1.data(), input2.data(), input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseMultModAVX512Int)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});
#endif

}  // namespace lattice
}  // namespace intel
