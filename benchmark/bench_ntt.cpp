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

#include <complex>
#include <vector>

#include "logging/logging.hpp"
#include "ntt.hpp"

namespace intel {
namespace ntt {

//=================================================================

static void BM_NTT1024(benchmark::State& state) {  //  NOLINT
  size_t N = 1024;
  size_t prime = 0xffffffffffc0001ULL;

  std::vector<uint64_t> input(1024, 1);
  NTT ntt(prime, N);

  for (auto _ : state) {
    ntt.ForwardTransformToBitReverse(&input);
  }
}

BENCHMARK(BM_NTT1024)->Unit(benchmark::kMicrosecond)->MinTime(5.0);

//=================================================================
static void BM_NTT4096(benchmark::State& state) {  //  NOLINT
  size_t N = 4096;
  size_t prime = 0xffffffffffc0001ULL;

  std::vector<uint64_t> input(4096, 1);
  NTT ntt(prime, N);

  for (auto _ : state) {
    ntt.ForwardTransformToBitReverse(&input);
  }
}

BENCHMARK(BM_NTT4096)->Unit(benchmark::kMicrosecond)->MinTime(5.0);

}  // namespace ntt
}  // namespace intel
