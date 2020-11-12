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
#include "ntt/ntt-internal.hpp"
#include "ntt/ntt.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#endif

#include "number-theory/number-theory.hpp"

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_FwdNTTNative(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 45, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  for (auto _ : state) {
    ForwardTransformToBitReverse64(
        ntt_size, prime, ntt_impl.GetRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64InvRootOfUnityPowers().data(), input.data());
  }
}

BENCHMARK(BM_FwdNTTNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});
//=================================================================

#ifdef LATTICE_HAS_AVX512IFMA
// state[0] is the degree
// state[1] is approximately the number of bits in the coefficient modulus
static void BM_FwdNTT_AVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime_bits = state.range(1);
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const std::vector<uint64_t> root_of_unity = ntt_impl.GetRootOfUnityPowers();
  const std::vector<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52InvRootOfUnityPowers();
  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
        ntt_size, prime, root_of_unity.data(), precon_root_of_unity.data(),
        input.data());
  }
}

BENCHMARK(BM_FwdNTT_AVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024, 49})
    ->Args({4096, 49})
    ->Args({8192, 49})
    ->Args({16384, 49});
#endif

//=================================================================

#ifdef LATTICE_HAS_AVX512DQ
// state[0] is the degree
// state[1] is approximately the number of bits in the coefficient modulus
static void BM_FwdNTT_AVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime_bits = state.range(1);
  size_t prime = GeneratePrimes(1, prime_bits, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const std::vector<uint64_t> root_of_unity = ntt_impl.GetRootOfUnityPowers();
  const std::vector<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    ForwardTransformToBitReverseAVX512<NTT::NTTImpl::s_default_shift_bits>(
        ntt_size, prime, root_of_unity.data(), precon_root_of_unity.data(),
        input.data());
  }
}

BENCHMARK(BM_FwdNTT_AVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024, 62})
    ->Args({4096, 62})
    ->Args({8192, 62})
    ->Args({16384, 62});
#endif

//=================================================================

// state[0] is the degree
static void BM_InvNTTNative(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 45, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const std::vector<uint64_t> root_of_unity =
      ntt_impl.GetInvRootOfUnityPowers();
  const std::vector<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverse64(ntt_size, prime, root_of_unity.data(),
                                     precon_root_of_unity.data(), input.data());
  }
}

BENCHMARK(BM_InvNTTNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

//=================================================================

#ifdef LATTICE_HAS_AVX512IFMA
// state[0] is the degree
static void BM_InvNTT_AVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 49, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const std::vector<uint64_t> root_of_unity =
      ntt_impl.GetInvRootOfUnityPowers();
  const std::vector<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon52InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
        ntt_size, prime, root_of_unity.data(), precon_root_of_unity.data(),
        input.data());
  }
}

BENCHMARK(BM_InvNTT_AVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});
#endif

//=================================================================

#ifdef LATTICE_HAS_AVX512DQ
// state[0] is the degree
static void BM_InvNTT_AVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 62, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT::NTTImpl ntt_impl(ntt_size, prime);

  const std::vector<uint64_t> root_of_unity =
      ntt_impl.GetInvRootOfUnityPowers();
  const std::vector<uint64_t> precon_root_of_unity =
      ntt_impl.GetPrecon64InvRootOfUnityPowers();
  for (auto _ : state) {
    InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_default_shift_bits>(
        ntt_size, prime, root_of_unity.data(), precon_root_of_unity.data(),
        input.data());
  }
}

BENCHMARK(BM_InvNTT_AVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});
#endif

// state[0] is the degree
static void BM_FwdNTTInPlace(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 62, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    ntt.ComputeForward(input.data());
  }
}

BENCHMARK(BM_FwdNTTInPlace)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

// state[0] is the degree
static void BM_FwdNTTCopy(benchmark::State& state) {  //  NOLINT
  size_t ntt_size = state.range(0);
  size_t prime = GeneratePrimes(1, 62, ntt_size)[0];

  std::vector<uint64_t> input(ntt_size, 1);
  std::vector<uint64_t> output(ntt_size, 1);
  NTT ntt(ntt_size, prime);

  for (auto _ : state) {
    ntt.ComputeForward(input.data(), output.data());
  }
}

BENCHMARK(BM_FwdNTTCopy)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(5.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});

}  // namespace lattice
}  // namespace intel
