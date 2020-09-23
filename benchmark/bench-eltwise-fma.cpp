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

#include "eltwise/eltwise-fma-internal.hpp"
#include "eltwise/eltwise-fma.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-fma-avx512.hpp"
#endif

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_EltwiseFMANative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  std::vector<uint64_t> op1(input_size, 1);
  uint64_t op2 = 1;
  std::vector<uint64_t> op3(input_size, 2);

  for (auto _ : state) {
    EltwiseFMAMod(op1.data(), op2, op3.data(), op1.data(), op1.size(), modulus);
  }
}

BENCHMARK(BM_EltwiseFMANative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef LATTICE_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseFMAAVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;

  std::vector<uint64_t> input1(input_size, 1);
  uint64_t input2 = 3;
  std::vector<uint64_t> input3(input_size, 2);

  for (auto _ : state) {
    EltwiseFMAModAVX512<64>(input1.data(), input2, input3.data(), input1.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAAVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

//=================================================================

#ifdef LATTICE_HAS_AVX512IFMA
// state[0] is the degree
static void BM_EltwiseFMAAVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;

  std::vector<uint64_t> input1(input_size, 1);
  uint64_t input2 = 3;
  std::vector<uint64_t> input3(input_size, 2);

  for (auto _ : state) {
    EltwiseFMAModAVX512<52>(input1.data(), input2, input3.data(), input1.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAAVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

}  // namespace lattice
}  // namespace intel
