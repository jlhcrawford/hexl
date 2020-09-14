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

#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "poly/poly-cmp-sub-mod-internal.hpp"
#include "poly/poly-cmp-sub-mod.hpp"
#include "test/test-util.hpp"

#ifdef LATTICE_HAS_AVX512F
#include "poly/poly-cmp-sub-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

TEST(PolyCmpGtSubMod, small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7};
  uint64_t cmp = 4;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2};

  uint64_t modulus = 10;

  CmpGtSubMod(op1.data(), cmp, diff, modulus, op1.size());
  CheckEqual(op1, exp_out);
}

TEST(PolyCmpGtSubMod, small8) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t cmp = 4;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2, 3};

  uint64_t modulus = 10;

  CmpGtSubMod(op1.data(), cmp, diff, modulus, op1.size());
  CheckEqual(op1, exp_out);
}

// Checks AVX512 and native implementations match
#ifdef LATTICE_HAS_AVX512IFMA
TEST(PolyCmpGtSubMod, AVX512) {
  uint64_t length = 128;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t bits = 48; bits <= 51; ++bits) {
    uint64_t prime = GeneratePrimes(1, bits, 1024)[0];
    std::uniform_int_distribution<> distrib(0, prime - 1);

    for (size_t trial = 0; trial < 1000; ++trial) {
      std::vector<uint64_t> arg1(length, 0);
      uint64_t cmp = distrib(gen);
      uint64_t diff = distrib(gen);
      std::vector<uint64_t> arg3(length, 0);
      for (size_t i = 0; i < length; ++i) {
        arg1[i] = distrib(gen);
        arg3[i] = distrib(gen);
      }
      std::vector<uint64_t> arg1a = arg1;

      CmpGtSubModNative(arg1.data(), cmp, diff, prime, arg1.size());
      CmpGtSubModAVX512(arg1a.data(), cmp, diff, prime, arg1a.size());

      ASSERT_EQ(arg1, arg1a);
    }
  }
}
#endif

}  // namespace lattice
}  // namespace intel
