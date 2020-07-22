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
#include "ntt/ntt.hpp"

namespace intel {
namespace ntt {

//=================================================================

// state[0] is the degree
// state[1] is approximately the number of bits in the coefficient modulus
// We distinguish between above and below 52 bits, since optimized
// AVX512-IFMA implementations exist for < 52 bits
static void BM_NTT(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);

  size_t prime_bits = state.range(1);
  size_t prime = 1;
  if (prime_bits <= 52) {
    prime = 0xffffee001;
  } else {
    prime = 0xffffffffffc0001ULL;
  }

  std::vector<uint64_t> input(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    ntt.ForwardTransformToBitReverse(input.data());
  }
}

BENCHMARK(BM_NTT)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024, 52})
    ->Args({1024, 64})
    ->Args({4096, 52})
    ->Args({4096, 64});

}  // namespace ntt
}  // namespace intel
