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

#include "eltwise/eltwise-fma-internal.hpp"
#include "intel-lattice/eltwise/eltwise-fma.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"

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

  AlignedVector<uint64_t> op1(input_size, 1);
  uint64_t op2 = 1;
  AlignedVector<uint64_t> op3(input_size, 2);

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

  AlignedVector<uint64_t> input1(input_size, 1);
  uint64_t input2 = 3;
  AlignedVector<uint64_t> input3(input_size, 2);

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

  AlignedVector<uint64_t> input1(input_size, 1);
  uint64_t input2 = 3;
  AlignedVector<uint64_t> input3(input_size, 2);

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
