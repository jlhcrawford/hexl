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

  MultiplyModInPlace64(op1.data(), op2.data(), op1.size(), bf.Hi(), bf.Lo(),
                       modulus);

  CheckEqual(op1, exp_out);
}

TEST(PolyMult, mult2) {
  std::vector<uint64_t> op1{1, 2,  3,  4,  5,  6,  7,  8,
                            9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> op2{17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{17, 36, 57, 80, 4,  31, 60, 91,
                                23, 58, 95, 33, 74, 16, 61, 7};
  uint64_t modulus = 101;

  MultiplyModInPlace64(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

#ifdef LATTICE_HAS_AVX512F
TEST(PolyMult, avx512_small) {
  std::vector<uint64_t> op1{1, 2, 3, 1, 1, 1, 0, 1};
  std::vector<uint64_t> op2{1, 1, 1, 1, 2, 3, 1, 0};
  std::vector<uint64_t> exp_out{1, 2, 3, 1, 2, 3, 0, 0};

  uint64_t modulus = 769;
  Barrett128Factor bf(modulus);

  MultiplyModInPlace64AVX512(op1.data(), op2.data(), op1.size(), bf.Hi(),
                             bf.Lo(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(PolyMult, avx512_mult2) {
  std::vector<uint64_t> op1{1, 2,  3,  4,  5,  6,  7,  8,
                            9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> op2{17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{17, 36, 57, 80, 4,  31, 60, 91,
                                23, 58, 95, 33, 74, 16, 61, 7};

  uint64_t modulus = 101;
  Barrett128Factor bf(modulus);

  MultiplyModInPlace64AVX512(op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}
#endif

}  // namespace lattice
}  // namespace intel
