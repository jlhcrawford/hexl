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
#include "poly/poly-mult.hpp"

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
// state[1] is approximately the number of bits in the coefficient modulus
// We distinguish between above and below 50 bits, since optimized
// AVX512-IFMA implementations exist for < 50 bits
static void BM_PolyMult(benchmark::State& state) {  //  NOLINT
  size_t poly_size = state.range(0);

  size_t modulus_bits = state.range(1);
  size_t modulus = 1;
  if (modulus_bits < 50) {
    modulus = 0xffffee001;
  } else {
    modulus = 0xffffffffffc0001ULL;
  }

  std::vector<uint64_t> input1(poly_size, 1);
  std::vector<uint64_t> input2(poly_size, 2);

  for (auto _ : state) {
    MultiplyModInPlace(input1.data(), input2.data(), poly_size, modulus);
  }
}

BENCHMARK(BM_PolyMult)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024, 49})
    ->Args({1024, 64})
    ->Args({4096, 49})
    ->Args({4096, 64});

}  // namespace lattice
}  // namespace intel
