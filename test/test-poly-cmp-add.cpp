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
#include "poly/poly-cmp-add-internal.hpp"
#include "poly/poly-cmp-add.hpp"
#include "test/test-util.hpp"

#ifdef LATTICE_HAS_AVX512F
#include "poly/poly-cmp-add-avx512.hpp"
#endif

namespace intel {
namespace lattice {

TEST(PolyCmpGtAdd, small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 1, 0};
  uint64_t cmp = 1;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 7, 8, 9, 10, 1, 0};

  CmpGtAdd(op1.data(), cmp, diff, op1.size());
  CheckEqual(op1, exp_out);
}

TEST(PolyCmpGtAdd, small8) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t cmp = 4;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 10, 11, 12, 13};

  CmpGtAdd(op1.data(), cmp, diff, op1.size());
  CheckEqual(op1, exp_out);
}

// Checks AVX512 and native implementations match
#ifdef LATTICE_HAS_AVX512F
TEST(PolyCmpGtAdd, AVX512) {
  uint64_t length = 1024;
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> distrib(0, 100);

  for (size_t trial = 0; trial < 1000; ++trial) {
    std::vector<uint64_t> op1(length, 0);
    uint64_t cmp = distrib(gen);
    uint64_t diff = distrib(gen);
    for (size_t i = 0; i < length; ++i) {
      op1[i] = distrib(gen);
    }
    std::vector<uint64_t> op1a = op1;

    CmpGtAddNative(op1.data(), cmp, diff, op1.size());
    CmpGtAddAVX512(op1a.data(), cmp, diff, op1a.size());

    ASSERT_EQ(op1, op1a);
  }
}
#endif

}  // namespace lattice
}  // namespace intel
