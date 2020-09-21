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

#ifdef LATTICE_HAS_AVX512DQ
#include "poly/poly-mult-avx512.hpp"
#endif

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
    MultiplyModInPlaceNative(input1.data(), input2.data(), poly_size, modulus);
  }
}

BENCHMARK(BM_PolyMultNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef LATTICE_HAS_AVX512DQ
// state[0] is the degree
static void BM_PolyMultAVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t poly_size = state.range(0);
  size_t modulus = 100;

  std::vector<uint64_t> input1(poly_size, 1);
  std::vector<uint64_t> input2(poly_size, 2);

  for (auto _ : state) {
    MultiplyModInPlaceAVX512<64>(input1.data(), input2.data(), poly_size,
                                 modulus);
  }
}

BENCHMARK(BM_PolyMultAVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

//=================================================================

#ifdef LATTICE_HAS_AVX512IFMA
// state[0] is the degree
static void BM_PolyMultAVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t poly_size = state.range(0);
  size_t modulus = 100;

  std::vector<uint64_t> input1(poly_size, 1);
  std::vector<uint64_t> input2(poly_size, 2);

  for (auto _ : state) {
    MultiplyModInPlaceAVX512<52>(input1.data(), input2.data(), poly_size,
                                 modulus);
  }
}

BENCHMARK(BM_PolyMultAVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

}  // namespace lattice
}  // namespace intel
