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

#include "eltwise/eltwise-add-mod-internal.hpp"
#include "eltwise/eltwise-add-mod.hpp"
#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "test/test-util.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-add-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

TEST(EltwiseAdd, native_small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 2, 4, 6};
  std::vector<uint64_t> exp_out{2, 5, 8, 1, 4, 8, 1, 4};
  uint64_t modulus = 10;

  EltwiseAddModNative(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAdd, native_big) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, modulus - 1, modulus - 2, modulus - 2,
                            modulus - 3, modulus - 3, modulus - 4, modulus - 4};
  std::vector<uint64_t> op2{modulus - 1, modulus - 2, modulus - 3, modulus - 4,
                            modulus - 5, modulus - 6, modulus - 7, modulus - 8};
  std::vector<uint64_t> exp_out{modulus - 2,  modulus - 3, modulus - 5,
                                modulus - 6,  modulus - 8, modulus - 9,
                                modulus - 11, modulus - 12};

  EltwiseAddModNative(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseAdd, avx512_small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 2, 4, 6};
  std::vector<uint64_t> exp_out{2, 5, 8, 1, 4, 8, 1, 4};
  uint64_t modulus = 10;
  EltwiseAddModAVX512(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAdd, avx512_big) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, modulus - 1, modulus - 2, modulus - 2,
                            modulus - 3, modulus - 3, modulus - 4, modulus - 4};
  std::vector<uint64_t> op2{modulus - 1, modulus - 2, modulus - 3, modulus - 4,
                            modulus - 5, modulus - 6, modulus - 7, modulus - 8};
  std::vector<uint64_t> exp_out{modulus - 2,  modulus - 3, modulus - 5,
                                modulus - 6,  modulus - 8, modulus - 9,
                                modulus - 11, modulus - 12};

  EltwiseAddModAVX512(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

#endif

// Checks AVX512 and native eltwise mult implementations match
#ifdef LATTICE_HAS_AVX512DQ
#ifndef LATTICE_DEBUG
TEST(EltwiseAdd, AVX512Big) {
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t log2N = 13; log2N <= 15; ++log2N) {
    size_t length = 1 << log2N;

    for (size_t bits = 54; bits <= 60; ++bits) {
      uint64_t prime = GeneratePrimes(1, bits, 1024)[0];
      std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

      for (size_t trial = 0; trial < 100; ++trial) {
        std::vector<uint64_t> op1(length, 0);
        std::vector<uint64_t> op2(length, 0);
        for (size_t i = 0; i < length; ++i) {
          op1[i] = distrib(gen);
          op2[i] = distrib(gen);
        }
        auto op1a = op1;

        EltwiseAddModNative(op1.data(), op2.data(), op1.size(), prime);
        EltwiseAddModAVX512(op1a.data(), op2.data(), op1.size(), prime);

        ASSERT_EQ(op1, op1a);
      }
    }
  }
}
#endif
#endif

}  // namespace lattice
}  // namespace intel
