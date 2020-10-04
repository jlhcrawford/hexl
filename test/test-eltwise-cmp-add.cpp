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

#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "eltwise/eltwise-cmp-add.hpp"
#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "test/test-util.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-cmp-add-avx512.hpp"
#endif

namespace intel {
namespace lattice {

// Parameters = (input, cmp, bound, diff, expected_output)
class EltwiseCmpAddTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<uint64_t>, CMPINT, uint64_t, uint64_t,
                     std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test Native implementation
TEST_P(EltwiseCmpAddTest, Native) {
  std::vector<uint64_t> input = std::get<0>(GetParam());
  CMPINT cmp = std::get<1>(GetParam());
  uint64_t bound = std::get<2>(GetParam());
  uint64_t diff = std::get<3>(GetParam());
  std::vector<uint64_t> exp_output = std::get<4>(GetParam());

  EltwiseCmpAddNative(input.data(), cmp, bound, diff, input.size());

  CheckEqual(input, exp_output);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseCmpAddTest, EltwiseCmpAddTest,
    ::testing::Values(
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::EQ,
                        4, 5, std::vector<uint64_t>{1, 2, 3, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::LT,
                        4, 5, std::vector<uint64_t>{6, 7, 8, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::LE,
                        4, 5, std::vector<uint64_t>{6, 7, 8, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7},
                        CMPINT::FALSE, 4, 5,
                        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NE,
                        4, 5, std::vector<uint64_t>{6, 7, 8, 4, 10, 11, 12}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NLT,
                        4, 5, std::vector<uint64_t>{1, 2, 3, 9, 10, 11, 12}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NLE,
                        4, 5, std::vector<uint64_t>{1, 2, 3, 4, 10, 11, 12}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7},
                        CMPINT::TRUE, 4, 5,
                        std::vector<uint64_t>{6, 7, 8, 9, 10, 11, 12})));

// Checks AVX512 and native implementations match
#ifdef LATTICE_HAS_AVX512DQ
TEST(EltwiseCmpAdd, AVX512) {
  uint64_t length = 1025;
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<uint64_t> distrib(0, 100);

  for (size_t cmp = 0; cmp < 8; ++cmp) {
    for (size_t trial = 0; trial < 200; ++trial) {
      std::vector<uint64_t> op1(length, 0);
      uint64_t bound = distrib(gen);
      uint64_t diff = distrib(gen);
      for (size_t i = 0; i < length; ++i) {
        op1[i] = distrib(gen);
      }
      std::vector<uint64_t> op1a = op1;
      std::vector<uint64_t> op1b = op1;

      EltwiseCmpAdd(op1.data(), static_cast<CMPINT>(cmp), bound, diff,
                    op1.size());
      EltwiseCmpAddNative(op1a.data(), static_cast<CMPINT>(cmp), bound, diff,
                          op1a.size());
      EltwiseCmpAddAVX512(op1b.data(), static_cast<CMPINT>(cmp), bound, diff,
                          op1b.size());

      ASSERT_EQ(op1, op1a);
      ASSERT_EQ(op1, op1b);
    }
  }
}
#endif

}  // namespace lattice
}  // namespace intel
