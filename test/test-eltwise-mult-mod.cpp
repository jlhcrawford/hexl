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

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod.hpp"
#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "test/test-util.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-mult-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

TEST(EltwiseMult, native_mult2) {
  std::vector<uint64_t> op1{1, 2,  3,  4,  5,  6,  7,  8,
                            9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> op2{17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{17, 36, 57, 80, 4,  31, 60, 91,
                                23, 58, 95, 33, 74, 16, 61, 7};
  uint64_t modulus = 101;

  EltwiseMultModNative(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseMult, avx512_small) {
  std::vector<uint64_t> op1{1, 2, 3, 1, 1, 1, 0, 1, 0};
  std::vector<uint64_t> op2{1, 1, 1, 1, 2, 3, 1, 0, 0};
  std::vector<uint64_t> exp_out{1, 2, 3, 1, 2, 3, 0, 0, 0};

  uint64_t modulus = 769;
  EltwiseMultModAVX512Int(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseMult, avx512_int2) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModAVX512Int(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

#endif

TEST(EltwiseMult, native2_big) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseMult, 4) {
  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

  EltwiseMultMod(op1.data(), op2.data(), op1.size(), modulus);
  CheckEqual(op1, exp_out);
}

TEST(EltwiseMult, 6) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5};
  std::vector<uint64_t> op2{2, 4, 6, 8, 10, 12};
  std::vector<uint64_t> exp_out{0, 4, 12, 24, 40, 60};

  uint64_t modulus = 769;

  EltwiseMultMod(op1.data(), op2.data(), op1.size(), modulus);
  CheckEqual(op1, exp_out);
}

TEST(EltwiseMult, 8big) {
  uint64_t modulus = GeneratePrimes(1, 48, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> exp_out{1, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseMult, 8big2) {
  uint64_t p = 281474976749569;

  std::vector<uint64_t> op1{(p - 1) / 2, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{(p + 1) / 2, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> exp_out{70368744187392, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(op1.data(), op2.data(), op1.size(), p);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseMult, 8big3) {
  uint64_t p = 1125891450734593;

  std::vector<uint64_t> op1{1078888294739028, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{1114802337613200, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> exp_out{13344071208410, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(op1.data(), op2.data(), op1.size(), p);

  CheckEqual(op1, exp_out);
}

#ifdef LATTICE_DEBUG
TEST(EltwiseMult, 8_bounds) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint64_t> op2{0, 1, 2, 3, 4, 5, 6, 770};

  uint64_t modulus = 769;

  EXPECT_ANY_THROW(EltwiseMultMod(op1.data(), op2.data(), op1.size(), modulus));
}
#endif

TEST(EltwiseMult, 9) {
  uint64_t modulus = GeneratePrimes(1, 51, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{modulus - 4, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<uint64_t> exp_out{12, 8, 14, 18, 20, 20, 18, 14, 8};

  EltwiseMultMod(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseMultBig, 9) {
  uint64_t modulus = 1125891450734593;

  std::vector<uint64_t> op1{706712574074152, 943467560561867, 1115920708919443,
                            515713505356094, 525633777116309, 910766532971356,
                            757086506562426, 799841520990167};
  std::vector<uint64_t> op2{515910833966633, 96924929169117,  537587376997453,
                            41829060600750,  205864998008014, 463185427411646,
                            965818279134294, 1075778049568657};
  std::vector<uint64_t> exp_out{
      231838787758587, 618753612121218, 1116345967490421, 409735411065439,
      25680427818594,  950138933882289, 554128714280822,  1465109636753};

  EltwiseMultModAVX512Int(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}
#endif

// Checks AVX512 and native eltwise mult implementations match
#ifdef LATTICE_HAS_AVX512DQ
#ifndef LATTICE_DEBUG
TEST(EltwiseMult, AVX512Big) {
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

        EltwiseMultModNative(op1.data(), op2.data(), op1.size(), prime);
        EltwiseMultModAVX512Int(op1a.data(), op2.data(), op1.size(), prime);

        ASSERT_EQ(op1, op1a);
      }
    }
  }
}
#endif
#endif

TEST(EltwiseMultOofP, native_mult2) {
  std::vector<uint64_t> op1{1, 2,  3,  4,  5,  6,  7,  8,
                            9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> op2{17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0};
  std::vector<uint64_t> exp_out{17, 36, 57, 80, 4,  31, 60, 91,
                                23, 58, 95, 33, 74, 16, 61, 7};
  uint64_t modulus = 101;

  EltwiseMultModNative(result.data(), op1.data(), op2.data(), op1.size(),
                       modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultOofP, native2_big) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(result.data(), op1.data(), op2.data(), op1.size(),
                       modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultOofP, 8big) {
  uint64_t modulus = GeneratePrimes(1, 48, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{1, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(result.data(), op1.data(), op2.data(), op1.size(),
                       modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultOofP, 8big2) {
  uint64_t p = 281474976749569;

  std::vector<uint64_t> op1{(p - 1) / 2, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{(p + 1) / 2, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{70368744187392, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(result.data(), op1.data(), op2.data(), op1.size(), p);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultOofP, 8big3) {
  uint64_t p = 1125891450734593;

  std::vector<uint64_t> op1{1078888294739028, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{1114802337613200, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{13344071208410, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative(result.data(), op1.data(), op2.data(), op1.size(), p);

  CheckEqual(result, exp_out);
}
#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseMultOofP, avx512_small) {
  std::vector<uint64_t> op1{1, 2, 3, 1, 1, 1, 0, 1, 0};
  std::vector<uint64_t> op2{1, 1, 1, 1, 2, 3, 1, 0, 0};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{1, 2, 3, 1, 2, 3, 0, 0, 0};

  uint64_t modulus = 769;
  EltwiseMultModAVX512Int(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultOofP, avx512_int2) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModAVX512Int(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}

#endif

TEST(EltwiseMultOofP, 4) {
  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> result{0, 0, 0, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

  EltwiseMultMod(result.data(), op1.data(), op2.data(), op1.size(), modulus);
  CheckEqual(result, exp_out);
}

TEST(EltwiseMultOofP, 6) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5};
  std::vector<uint64_t> op2{2, 4, 6, 8, 10, 12};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{0, 4, 12, 24, 40, 60};

  uint64_t modulus = 769;

  EltwiseMultMod(result.data(), op1.data(), op2.data(), op1.size(), modulus);
  CheckEqual(result, exp_out);
}

#ifdef LATTICE_DEBUG
TEST(EltwiseMultOofP, 8_bounds) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint64_t> op2{0, 1, 2, 3, 4, 5, 6, 770};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 769;

  EXPECT_ANY_THROW(EltwiseMultMod(result.data(), op1.data(), op2.data(),
                                  op1.size(), modulus));
}
#endif

TEST(EltwiseMultOofP, 9) {
  uint64_t modulus = GeneratePrimes(1, 51, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{modulus - 4, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 8, 14, 18, 20, 20, 18, 14, 8};

  EltwiseMultMod(result.data(), op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(result, exp_out);
}

#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseMultBigOofP, 9) {
  uint64_t modulus = 1125891450734593;

  std::vector<uint64_t> op1{706712574074152, 943467560561867, 1115920708919443,
                            515713505356094, 525633777116309, 910766532971356,
                            757086506562426, 799841520990167};
  std::vector<uint64_t> op2{515910833966633, 96924929169117,  537587376997453,
                            41829060600750,  205864998008014, 463185427411646,
                            965818279134294, 1075778049568657};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{
      231838787758587, 618753612121218, 1116345967490421, 409735411065439,
      25680427818594,  950138933882289, 554128714280822,  1465109636753};

  EltwiseMultModAVX512Int(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}
#endif

// Checks AVX512 and native eltwise mult Out-of-Place implementations match
#ifdef LATTICE_HAS_AVX512DQ
#ifndef LATTICE_DEBUG
TEST(EltwiseMultOofP, AVX512Big) {
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
        std::vector<uint64_t> rs1(length, 0);
        std::vector<uint64_t> rs2(length, 0);
        for (size_t i = 0; i < length; ++i) {
          op1[i] = distrib(gen);
          op2[i] = distrib(gen);
        }
        auto op1a = op1;

        EltwiseMultModNative(rs1.data(), op1.data(), op2.data(), op1.size(),
                             prime);
        EltwiseMultModAVX512Int(rs2.data(), op1a.data(), op2.data(), op1.size(),
                                prime);

        ASSERT_EQ(rs1, rs2);
      }
    }
  }
}
#endif
#endif
}  // namespace lattice
}  // namespace intel
