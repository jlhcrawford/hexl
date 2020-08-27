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
#include "number-theory/number-theory.hpp"

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_NTTNative(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 45, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    NTT::ForwardTransformToBitReverse64(
        ntt_size, prime, ntt.GetRootOfUnityPowers().data(),
        ntt.GetPreconInvRootOfUnityPowers().data(), input.data());
  }
}

BENCHMARK(BM_NTTNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
//=================================================================

// state[0] is the degree
// state[1] is approximately the number of bits in the coefficient modulus
static void BM_NTT_AVX512(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime_bits = state.range(1);
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    ntt.ForwardTransformToBitReverse(input.data());
  }
}

BENCHMARK(BM_NTT_AVX512)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024, 49})
    ->Args({1024, 62})
    ->Args({4096, 49})
    ->Args({4096, 62})
    ->Args({16384, 49})
    ->Args({16384, 62});

//=================================================================

// state[0] is the degree
static void BM_invNTTNative(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 45, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    NTT::InverseTransformToBitReverse64(
        ntt_size, prime, ntt.GetRootOfUnityPowers().data(),
        ntt.GetPreconInvRootOfUnityPowers().data(), input.data());
  }
}

BENCHMARK(BM_invNTTNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
// state[1] is approximately the number of bits in the coefficient modulus
static void BM_invNTTAX512(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);

  size_t prime_bits = state.range(1);
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    ntt.InverseTransformToBitReverse(input.data());
  }
}

BENCHMARK(BM_invNTTAX512)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024, 49})
    ->Args({1024, 60})
    ->Args({4096, 49})
    ->Args({4096, 60})
    ->Args({16384, 49})
    ->Args({16384, 60});

}  // namespace lattice
}  // namespace intel
