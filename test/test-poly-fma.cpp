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
#include "poly/poly-fma-internal.hpp"
#include "poly/poly-fma.hpp"
#include "test/test-util.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "poly/poly-fma-avx512.hpp"
#endif

namespace intel {
namespace lattice {

TEST(FMAModScalar, small) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 1;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{10, 12, 14, 16, 18, 20, 22, 24};
  uint64_t modulus = 769;

  MultiplyFactor mf(arg2, 64, modulus);

  FMAModScalarNative(arg1.data(), arg2, arg3.data(), arg1.data(),
                     mf.BarrettFactor(), arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(FMAModScalar, native_null) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;

  FMAModScalarNative(arg1.data(), arg2, nullptr, arg1.data(), arg1.size(),
                     modulus);

  CheckEqual(arg1, exp_out);
}

TEST(FMAModScalar, null) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;

  FMAModScalar(arg1.data(), arg2, nullptr, arg1.data(), arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(FMAModScalar, mult2) {
  std::vector<uint64_t> arg1{1,  2,  3,  4,  5,  6,  7,  8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17};
  uint64_t arg2 = 72;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24, 25,
                             26, 27, 28, 29, 30, 31, 32, 33};
  std::vector<uint64_t> exp_out{89, 61, 33, 5,  78, 50, 22, 95, 67,
                                39, 11, 84, 56, 28, 0,  73, 45};
  uint64_t modulus = 101;

  FMAModScalar(arg1.data(), arg2, arg3.data(), arg1.data(), arg1.size(),
               modulus);

  CheckEqual(arg1, exp_out);
}

#ifdef LATTICE_HAS_AVX512DQ
TEST(FMAModScalar, avx512_small) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 2;
  std::vector<uint64_t> arg3{1, 1, 1, 1, 2, 3, 1, 0};
  std::vector<uint64_t> exp_out{3, 5, 7, 9, 12, 15, 15, 16};

  uint64_t modulus = 101;
  BarrettFactor<64> bf(modulus);

  FMAModScalarAVX512<64>(arg1.data(), arg2, arg3.data(), arg1.data(),
                         arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(FMAModScalar, avx512_small2) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{26, 44, 62, 80, 98, 15, 33, 51};

  uint64_t modulus = 101;

  FMAModScalarAVX512<64>(arg1.data(), arg2, arg3.data(), arg1.data(),
                         arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(FMAModScalar, avx512_mult2) {
  std::vector<uint64_t> arg1{1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{34, 52, 70, 88, 5,  23, 41, 59,
                                77, 95, 12, 30, 48, 66, 84, 1};

  uint64_t modulus = 101;

  FMAModScalarAVX512<64>(arg1.data(), arg2, arg3.data(), arg1.data(),
                         arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

#endif

// Checks AVX512 and native poly FMA implementations match
#ifdef LATTICE_HAS_AVX512IFMA
TEST(FMAModScalar, AVX512) {
  uint64_t length = 1024;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t bits = 48; bits <= 51; ++bits) {
    uint64_t prime = GeneratePrimes(1, bits, 1024)[0];
    std::uniform_int_distribution<> distrib(0, prime - 1);

    for (size_t trial = 0; trial < 200; ++trial) {
      std::vector<uint64_t> arg1(length, 0);
      uint64_t arg2 = distrib(gen);
      std::vector<uint64_t> arg3(length, 0);
      for (size_t i = 0; i < length; ++i) {
        arg1[i] = distrib(gen);
        arg3[i] = distrib(gen);
      }
      std::vector<uint64_t> arg1a = arg1;
      std::vector<uint64_t> arg1b = arg1;

      uint64_t* arg3_data = (trial % 2 == 0) ? arg3.data() : nullptr;

      FMAModScalarNative(arg1.data(), arg2, arg3_data, arg1.data(), arg1.size(),
                         prime);
      FMAModScalarAVX512<52>(arg1a.data(), arg2, arg3_data, arg1a.data(),
                             arg1.size(), prime);
      FMAModScalarAVX512<64>(arg1b.data(), arg2, arg3_data, arg1b.data(),
                             arg1.size(), prime);

      ASSERT_EQ(arg1, arg1a);
      ASSERT_EQ(arg1, arg1b);
    }
  }
}
#endif

}  // namespace lattice
}  // namespace intel
