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

#include <memory>
#include <random>
#include <vector>

#include "eltwise/eltwise-fma-internal.hpp"
#include "gtest/gtest.h"
#include "intel-lattice/eltwise/eltwise-fma.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "test-util.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-fma-avx512.hpp"
#endif

namespace intel {
namespace lattice {

#ifdef LATTICE_DEBUG
TEST(EltwiseFMAMod, null) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};

  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 1;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{10, 12, 14, 16, 18, 20, 22, 24};
  uint64_t modulus = 769;
  std::vector<uint64_t> big_input(op1.size(), modulus);

  EXPECT_ANY_THROW(EltwiseFMAMod(nullptr, arg2, arg3.data(), arg1.data(),
                                 arg1.size(), modulus));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), arg2, arg3.data(), nullptr,
                                 arg1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseFMAMod(arg1.data(), arg2, arg3.data(), arg1.data(), 0, modulus));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), arg2, arg3.data(), arg1.data(),
                                 arg1.size(), 1));
  EXPECT_ANY_THROW(EltwiseFMAMod(big_input.data(), arg2, arg3.data(),
                                 arg1.data(), arg1.size(), modulus));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), arg2, big_input.data(),
                                 arg1.data(), arg1.size(), modulus));
}
#endif

TEST(EltwiseFMAMod, small) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 1;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{10, 12, 14, 16, 18, 20, 22, 24};
  uint64_t modulus = 769;

  EltwiseFMAMod(arg1.data(), arg2, arg3.data(), arg1.data(), arg1.size(),
                modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, native_null) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;

  EltwiseFMAMod(arg1.data(), arg2, nullptr, arg1.data(), arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, mult2) {
  std::vector<uint64_t> arg1{1,  2,  3,  4,  5,  6,  7,  8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17};
  uint64_t arg2 = 72;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24, 25,
                             26, 27, 28, 29, 30, 31, 32, 33};
  std::vector<uint64_t> exp_out{89, 61, 33, 5,  78, 50, 22, 95, 67,
                                39, 11, 84, 56, 28, 0,  73, 45};
  uint64_t modulus = 101;

  EltwiseFMAMod(arg1.data(), arg2, arg3.data(), arg1.data(), arg1.size(),
                modulus);

  CheckEqual(arg1, exp_out);
}

#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseFMAMod, avx512_small) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 2;
  std::vector<uint64_t> arg3{1, 1, 1, 1, 2, 3, 1, 0};
  std::vector<uint64_t> exp_out{3, 5, 7, 9, 12, 15, 15, 16};

  uint64_t modulus = 101;
  BarrettFactor<64> bf(modulus);

  EltwiseFMAModAVX512<64>(arg1.data(), arg2, arg3.data(), arg1.data(),
                          arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, avx512_small2) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{26, 44, 62, 80, 98, 15, 33, 51};

  uint64_t modulus = 101;

  EltwiseFMAModAVX512<64>(arg1.data(), arg2, arg3.data(), arg1.data(),
                          arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, avx512_mult2) {
  std::vector<uint64_t> arg1{1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{34, 52, 70, 88, 5,  23, 41, 59,
                                77, 95, 12, 30, 48, 66, 84, 1};

  uint64_t modulus = 101;

  EltwiseFMAModAVX512<64>(arg1.data(), arg2, arg3.data(), arg1.data(),
                          arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

#endif

// Checks AVX512 and native eltwise FMA implementations match
#ifdef LATTICE_HAS_AVX512IFMA
TEST(EltwiseFMAMod, AVX512) {
  uint64_t length = 1024;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t bits = 48; bits <= 51; ++bits) {
    uint64_t prime = GeneratePrimes(1, bits, length)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

    for (size_t trial = 0; trial < 1000; ++trial) {
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

      EltwiseFMAMod(arg1.data(), arg2, arg3_data, arg1.data(), arg1.size(),
                    prime);
      EltwiseFMAModAVX512<52>(arg1a.data(), arg2, arg3_data, arg1a.data(),
                              arg1.size(), prime);
      EltwiseFMAModAVX512<64>(arg1b.data(), arg2, arg3_data, arg1b.data(),
                              arg1.size(), prime);

      ASSERT_EQ(arg1, arg1a);
      ASSERT_EQ(arg1, arg1b);
    }
  }
}
#endif

}  // namespace lattice
}  // namespace intel
