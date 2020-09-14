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

#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "poly/poly-mult-internal.hpp"
#include "poly/poly-mult.hpp"

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_PolyMultNative(benchmark::State& state) {  //  NOLINT
  size_t poly_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  std::vector<uint64_t> input1(poly_size, 1);
  std::vector<uint64_t> input2(poly_size, 2);

  for (auto _ : state) {
    MultiplyModInPlace64(input1.data(), input2.data(), poly_size, modulus);
  }
}

BENCHMARK(BM_PolyMultNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({512})
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768});

//=================================================================

#ifdef LATTICE_HAS_AVX512F
// state[0] is the degree
// state[1] is the number of bits in the modulus
static void BM_PolyMultAVX512(benchmark::State& state) {  //  NOLINT
  size_t poly_size = state.range(0);
  uint64_t modulus = MaximumValue(state.range(1)) - 10;

  std::vector<uint64_t> input1(poly_size, 1);
  std::vector<uint64_t> input2(poly_size, 2);

  for (auto _ : state) {
    MultiplyModInPlace(input1.data(), input2.data(), poly_size, modulus);
  }
}

BENCHMARK(BM_PolyMultAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({512, 49})
    ->Args({512, 62})
    ->Args({1024, 49})
    ->Args({1024, 62})
    ->Args({2048, 49})
    ->Args({2048, 62})
    ->Args({4096, 49})
    ->Args({4096, 62})
    ->Args({8192, 49})
    ->Args({8192, 62})
    ->Args({16384, 49})
    ->Args({16384, 62})
    ->Args({32768, 49})
    ->Args({32768, 62});
#endif

}  // namespace lattice
}  // namespace intel
