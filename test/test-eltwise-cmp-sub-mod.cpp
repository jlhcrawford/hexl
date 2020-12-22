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

#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "gtest/gtest.h"
#include "intel-lattice/eltwise/eltwise-cmp-sub-mod.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "test-util.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-cmp-sub-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

#ifdef LATTICE_DEBUG
TEST(EltwiseCmpSubMod, null) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};

  EXPECT_ANY_THROW(EltwiseCmpSubMod(nullptr, CMPINT::EQ, 1, 1, 10, op1.size()));
  EXPECT_ANY_THROW(
      EltwiseCmpSubMod(op1.data(), CMPINT::EQ, 1, 0, 10, op1.size()));
  EXPECT_ANY_THROW(
      EltwiseCmpSubMod(op1.data(), CMPINT::EQ, 1, 0, 0, op1.size()));
  EXPECT_ANY_THROW(EltwiseCmpSubMod(op1.data(), CMPINT::EQ, 1, 1, 10, 0));
}
#endif

// Parameters = (input, cmp, bound, diff, modulus, expected_output)
class EltwiseCmpSubModTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<uint64_t>, CMPINT, uint64_t, uint64_t,
                     uint64_t, std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test Native implementation
TEST_P(EltwiseCmpSubModTest, Native) {
  std::vector<uint64_t> input = std::get<0>(GetParam());
  CMPINT cmp = std::get<1>(GetParam());
  uint64_t bound = std::get<2>(GetParam());
  uint64_t diff = std::get<3>(GetParam());
  uint64_t modulus = std::get<4>(GetParam());
  std::vector<uint64_t> exp_output = std::get<5>(GetParam());

  EltwiseCmpSubModNative(input.data(), cmp, bound, diff, modulus, input.size());

  CheckEqual(input, exp_output);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseCmpSubModTest, EltwiseCmpSubModTest,
    ::testing::Values(
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::EQ,
                        4, 5, 10, std::vector<uint64_t>{1, 2, 3, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::LT,
                        4, 5, 10, std::vector<uint64_t>{6, 7, 8, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::LE,
                        4, 5, 10, std::vector<uint64_t>{6, 7, 8, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7},
                        CMPINT::FALSE, 4, 5, 10,
                        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NE,
                        4, 5, 10, std::vector<uint64_t>{6, 7, 8, 4, 0, 1, 2}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NLT,
                        4, 5, 10, std::vector<uint64_t>{1, 2, 3, 9, 0, 1, 2}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NLE,
                        4, 5, 10, std::vector<uint64_t>{1, 2, 3, 4, 0, 1, 2}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7},
                        CMPINT::TRUE, 4, 5, 10,
                        std::vector<uint64_t>{6, 7, 8, 9, 0, 1, 2})));

// Checks AVX512 and native implementations match
#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseCmpSubMod, AVX512) {
  uint64_t length = 172;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t cmp = 0; cmp < 8; ++cmp) {
    for (size_t bits = 48; bits <= 51; ++bits) {
      uint64_t prime = GeneratePrimes(1, bits, 1024)[0];
      std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

      for (size_t trial = 0; trial < 200; ++trial) {
        std::vector<uint64_t> arg1(length, 0);
        uint64_t bound = distrib(gen);
        uint64_t diff = distrib(gen);
        std::vector<uint64_t> arg3(length, 0);
        for (size_t i = 0; i < length; ++i) {
          arg1[i] = distrib(gen);
          arg3[i] = distrib(gen);
        }
        std::vector<uint64_t> arg1a = arg1;
        std::vector<uint64_t> arg1b = arg1;

        EltwiseCmpSubMod(arg1.data(), static_cast<CMPINT>(cmp), bound, diff,
                         prime, arg1.size());
        EltwiseCmpSubModNative(arg1a.data(), static_cast<CMPINT>(cmp), bound,
                               diff, prime, arg1a.size());
        EltwiseCmpSubModAVX512(arg1b.data(), static_cast<CMPINT>(cmp), bound,
                               diff, prime, arg1b.size());

        ASSERT_EQ(arg1, arg1a);
        ASSERT_EQ(arg1, arg1b);
      }
    }
  }
}
#endif

}  // namespace lattice
}  // namespace intel
