// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

#include <benchmark/benchmark.h>

#include <vector>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "intel-lattice/eltwise/eltwise-mult-mod.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-mult-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_EltwiseMultModInPlace(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 123;

  AlignedVector<uint64_t> input1(input_size, 1);
  AlignedVector<uint64_t> input2(input_size, 2);

  for (auto _ : state) {
    EltwiseMultMod(input1.data(), input2.data(), input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseMultModInPlace)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_EltwiseMultMod(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 123;

  AlignedVector<uint64_t> input1(input_size, 1);
  AlignedVector<uint64_t> input2(input_size, 2);
  AlignedVector<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    EltwiseMultMod(input1.data(), input2.data(), output.data(), input_size,
                   modulus);
  }
}

BENCHMARK(BM_EltwiseMultMod)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

// state[0] is the degree
static void BM_EltwiseMultModNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  AlignedVector<uint64_t> input1(input_size, 1);
  AlignedVector<uint64_t> input2(input_size, 2);
  AlignedVector<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    EltwiseMultModNative(input1.data(), input2.data(), output.data(),
                         input_size, modulus);
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

  AlignedVector<uint64_t> input1(input_size, 1);
  AlignedVector<uint64_t> input2(input_size, 2);
  AlignedVector<uint64_t> output(input_size, 2);

  for (auto _ : state) {
    EltwiseMultModAVX512Float(input1.data(), input2.data(), output.data(),
                              input_size, modulus);
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

  AlignedVector<uint64_t> input1(input_size, 1);
  AlignedVector<uint64_t> input2(input_size, 2);
  AlignedVector<uint64_t> output(input_size, 3);

  for (auto _ : state) {
    EltwiseMultModAVX512Int(input1.data(), input2.data(), output.data(),
                            input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseMultModAVX512Int)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({4096})
    ->Args({8192})
    ->Args({16384});
#endif

//=================================================================

}  // namespace lattice
}  // namespace intel
