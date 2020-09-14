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

#include <random>
#include <vector>

#include "logging/logging.hpp"
#include "poly/poly-cmp-add-internal.hpp"
#include "poly/poly-cmp-add.hpp"

#ifdef LATTICE_HAS_AVX512F
#include "poly/poly-cmp-add-avx512.hpp"
#endif

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_PolyCmpAddNative(benchmark::State& state) {  //  NOLINT
  size_t poly_size = state.range(0);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> distrib(0, 100);

  uint64_t cmp = distrib(gen);
  uint64_t diff = distrib(gen);
  std::vector<uint64_t> input1(poly_size);
  for (size_t i = 0; i < poly_size; ++i) {
    input1[i] = distrib(gen);
  }

  for (auto _ : state) {
    CmpGtAddNative(input1.data(), cmp, diff, poly_size);
  }
}

BENCHMARK(BM_PolyCmpAddNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768});

//=================================================================

#ifdef LATTICE_HAS_AVX512F
// state[0] is the degree
static void BM_PolyCmpAddAVX512(benchmark::State& state) {  //  NOLINT
  size_t poly_size = state.range(0);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> distrib(0, 100);

  uint64_t cmp = distrib(gen);
  uint64_t diff = distrib(gen);
  std::vector<uint64_t> input1(poly_size);
  for (size_t i = 0; i < poly_size; ++i) {
    input1[i] = distrib(gen);
  }

  for (auto _ : state) {
    CmpGtAddAVX512(input1.data(), cmp, diff, poly_size);
  }
}

BENCHMARK(BM_PolyCmpAddAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384})
    ->Args({32768});
#endif

}  // namespace lattice
}  // namespace intel
