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

#include <chrono>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "ntt/ntt.hpp"
#include "number-theory/number-theory.hpp"
#include "poly/poly-mult.hpp"
#include "test/test_util.hpp"

namespace intel {
namespace lattice {

TEST(PolyMult, small) {
  std::vector<uint64_t> op1{1, 2, 3, 1, 1, 1, 0, 1};
  std::vector<uint64_t> op2{1, 1, 1, 1, 2, 3, 1, 0};
  std::vector<uint64_t> exp_out{1, 2, 3, 1, 2, 3, 0, 0};

  uint64_t modulus = 769;
  Barrett128Factor bf(modulus);

  MultiplyModInPlace(op1.data(), op2.data(), op1.size(), bf.Hi(), bf.Lo(),
                     modulus);

  CheckEqual(op1, exp_out);
}

TEST(PolyMult, mult2) {
  std::vector<uint64_t> op1{2, 4, 6, 8, 10, 12, 14};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 11, 13};
  std::vector<uint64_t> exp_out{2, 12, 30, 56, 90, 31, 81};

  uint64_t modulus = 101;
  Barrett128Factor bf(modulus);

  MultiplyModInPlace(op1.data(), op2.data(), op1.size(), bf.Hi(), bf.Lo(),
                     modulus);

  CheckEqual(op1, exp_out);
}

}  // namespace lattice
}  // namespace intel
