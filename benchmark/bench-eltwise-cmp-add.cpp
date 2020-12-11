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

#include <random>
#include <vector>

#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "eltwise/eltwise-cmp-add.hpp"
#include "logging/logging.hpp"
#include "util/aligned-allocator.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-cmp-add-avx512.hpp"
#endif

namespace intel {
namespace lattice {

//=================================================================

// state[0] is the degree
static void BM_EltwiseCmpAddNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<uint64_t> distrib(0, 100);

  uint64_t bound = distrib(gen);
  uint64_t diff = distrib(gen);
  AlignedVector<uint64_t> input1(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input1[i] = distrib(gen);
  }

  for (auto _ : state) {
    EltwiseCmpAddNative(input1.data(), CMPINT::NLT, bound, diff, input_size);
  }
}

BENCHMARK(BM_EltwiseCmpAddNative)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});

//=================================================================

#ifdef LATTICE_HAS_AVX512DQ
// state[0] is the degree
static void BM_EltwiseCmpAddAVX512(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<uint64_t> distrib(0, 100);

  uint64_t bound = 50;
  uint64_t diff = distrib(gen);
  AlignedVector<uint64_t> input1(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input1[i] = distrib(gen);
  }

  for (auto _ : state) {
    EltwiseCmpAddAVX512(input1.data(), CMPINT::NLT, bound, diff, input_size);
  }
}

BENCHMARK(BM_EltwiseCmpAddAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.0)
    ->Args({1024})
    ->Args({4096})
    ->Args({16384});
#endif

}  // namespace lattice
}  // namespace intel
