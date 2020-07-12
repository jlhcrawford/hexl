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
#include "number-theory.hpp"

namespace intel {
namespace ntt {

TEST(NumberTheory, Log2) {
  ASSERT_EQ(0, Log2(1));
  ASSERT_EQ(1, Log2(2));
  ASSERT_EQ(2, Log2(4));
  ASSERT_EQ(3, Log2(8));
  ASSERT_EQ(4, Log2(16));
  ASSERT_EQ(5, Log2(32));
  ASSERT_EQ(6, Log2(64));
  ASSERT_EQ(7, Log2(128));
  ASSERT_EQ(8, Log2(256));
  ASSERT_EQ(9, Log2(512));
  ASSERT_EQ(10, Log2(1024));
  ASSERT_EQ(11, Log2(2048));
  ASSERT_EQ(12, Log2(4096));
  ASSERT_EQ(13, Log2(8192));
}

TEST(NumberTheory, MultiplyUIntMod) {
  uint64_t mod(2);
  ASSERT_EQ(0ULL, MultiplyUIntMod(0, 0, mod));
  ASSERT_EQ(0ULL, MultiplyUIntMod(0, 1, mod));
  ASSERT_EQ(0ULL, MultiplyUIntMod(1, 0, mod));
  ASSERT_EQ(1ULL, MultiplyUIntMod(1, 1, mod));

  mod = 10;
  ASSERT_EQ(0ULL, MultiplyUIntMod(0, 0, mod));
  ASSERT_EQ(0ULL, MultiplyUIntMod(0, 1, mod));
  ASSERT_EQ(0ULL, MultiplyUIntMod(1, 0, mod));
  ASSERT_EQ(1ULL, MultiplyUIntMod(1, 1, mod));
  ASSERT_EQ(9ULL, MultiplyUIntMod(7, 7, mod));
  ASSERT_EQ(2ULL, MultiplyUIntMod(6, 7, mod));
  ASSERT_EQ(2ULL, MultiplyUIntMod(7, 6, mod));

  mod = 2305843009211596801ULL;
  ASSERT_EQ(0ULL, MultiplyUIntMod(0, 0, mod));
  ASSERT_EQ(0ULL, MultiplyUIntMod(0, 1, mod));
  ASSERT_EQ(0ULL, MultiplyUIntMod(1, 0, mod));
  ASSERT_EQ(1ULL, MultiplyUIntMod(1, 1, mod));
  ASSERT_EQ(
      576460752302899200ULL,
      MultiplyUIntMod(1152921504605798400ULL, 1152921504605798401ULL, mod));
  ASSERT_EQ(
      576460752302899200ULL,
      MultiplyUIntMod(1152921504605798401ULL, 1152921504605798400ULL, mod));
  ASSERT_EQ(
      1729382256908697601ULL,
      MultiplyUIntMod(1152921504605798401ULL, 1152921504605798401ULL, mod));
  ASSERT_EQ(1ULL, MultiplyUIntMod(2305843009211596800ULL,
                                  2305843009211596800ULL, mod));
}

TEST(NumberTheory, PowMod) {
  uint64_t mod = 5;
  ASSERT_EQ(1ULL, PowMod(1, 0, mod));
  ASSERT_EQ(1ULL, PowMod(1, 0xFFFFFFFFFFFFFFFFULL, mod));
  ASSERT_EQ(3ULL, PowMod(2, 0xFFFFFFFFFFFFFFFFULL, mod));

  mod = 0x1000000000000000ULL;
  ASSERT_EQ(0ULL, PowMod(2, 60, mod));
  ASSERT_EQ(0x800000000000000ULL, PowMod(2, 59, mod));

  mod = 131313131313;
  ASSERT_EQ(39418477653ULL, PowMod(2424242424, 16, mod));
}

TEST(NumberTheory, IsPowerOfTwo) {
  std::vector<uint64_t> powers_of_two{1,   2,    4,    8,    16,    32,
                                      512, 1024, 2048, 4096, 16384, 32768};
  std::vector<uint64_t> not_powers_of_two{0, 3, 5, 7, 9, 31, 33, 1025, 4095};

  for (auto power_of_two : powers_of_two) {
    EXPECT_TRUE(IsPowerOfTwo(power_of_two));
  }

  for (auto not_power_of_two : not_powers_of_two) {
    EXPECT_FALSE(IsPowerOfTwo(not_power_of_two));
  }
}

TEST(NumberTheory, IsPrimitiveRoot) {
  uint64_t mod = 11;
  ASSERT_TRUE(IsPrimitiveRoot(10, 2, mod));
  ASSERT_FALSE(IsPrimitiveRoot(9, 2, mod));
  ASSERT_FALSE(IsPrimitiveRoot(10, 4, mod));

  mod = 29;
  ASSERT_TRUE(IsPrimitiveRoot(28, 2, mod));
  ASSERT_TRUE(IsPrimitiveRoot(12, 4, mod));
  ASSERT_FALSE(IsPrimitiveRoot(12, 2, mod));
  ASSERT_FALSE(IsPrimitiveRoot(12, 8, mod));

  mod = 1234565441ULL;
  ASSERT_TRUE(IsPrimitiveRoot(1234565440ULL, 2, mod));
  ASSERT_TRUE(IsPrimitiveRoot(960907033ULL, 8, mod));
  ASSERT_TRUE(IsPrimitiveRoot(1180581915ULL, 16, mod));
  ASSERT_FALSE(IsPrimitiveRoot(1180581915ULL, 32, mod));
  ASSERT_FALSE(IsPrimitiveRoot(1180581915ULL, 8, mod));
  ASSERT_FALSE(IsPrimitiveRoot(1180581915ULL, 2, mod));
}

TEST(NumberTheory, MinimalPrimitiveRoot) {
  uint64_t mod = 11;

  ASSERT_EQ(10ULL, MinimalPrimitiveRoot(2, mod));

  mod = 29;
  ASSERT_EQ(28ULL, MinimalPrimitiveRoot(2, mod));
  ASSERT_EQ(12ULL, MinimalPrimitiveRoot(4, mod));

  mod = 1234565441;
  ASSERT_EQ(1234565440ULL, MinimalPrimitiveRoot(2, mod));
  ASSERT_EQ(249725733ULL, MinimalPrimitiveRoot(8, mod));
}

TEST(NumberTheory, InverseUIntMod) {
  uint64_t input;
  uint64_t modulus;

  input = 1, modulus = 2;
  ASSERT_EQ(1ULL, InverseUIntMod(input, modulus));

#ifndef NTT_CHECK
  input = 2, modulus = 2;
  EXPECT_ANY_THROW(InverseUIntMod(input, modulus));

  input = 0xFFFFFE, modulus = 2;
  EXPECT_ANY_THROW(InverseUIntMod(input, modulus));

  input = 12345, modulus = 3;
  EXPECT_ANY_THROW(InverseUIntMod(input, modulus));
#endif

  input = 3, modulus = 2;
  ASSERT_EQ(1ULL, InverseUIntMod(input, modulus));

  input = 0xFFFFFF, modulus = 2;
  ASSERT_EQ(1ULL, InverseUIntMod(input, modulus));

  input = 5, modulus = 19;
  ASSERT_EQ(4ULL, InverseUIntMod(input, modulus));

  input = 4, modulus = 19;
  ASSERT_EQ(5ULL, InverseUIntMod(input, modulus));
}

TEST(NumberTheory, ReverseBitsUInt64) {
  ASSERT_EQ(0ULL, ReverseBitsUInt(0ULL, 0));
  ASSERT_EQ(0ULL, ReverseBitsUInt(0ULL, 1));
  ASSERT_EQ(0ULL, ReverseBitsUInt(0ULL, 32));
  ASSERT_EQ(0ULL, ReverseBitsUInt(0ULL, 64));

  ASSERT_EQ(0ULL, ReverseBitsUInt(1ULL, 0));
  ASSERT_EQ(1ULL, ReverseBitsUInt(1ULL, 1));
  ASSERT_EQ(1ULL << 31, ReverseBitsUInt(1ULL, 32));
  ASSERT_EQ(1ULL << 63, ReverseBitsUInt(1ULL, 64));

  ASSERT_EQ(0ULL, ReverseBitsUInt(1ULL << 31, 0));
  ASSERT_EQ(0ULL, ReverseBitsUInt(1ULL << 31, 1));
  ASSERT_EQ(1ULL, ReverseBitsUInt(1ULL << 31, 32));
  ASSERT_EQ(1ULL << 32, ReverseBitsUInt(1ULL << 31, 64));

  ASSERT_EQ(0ULL, ReverseBitsUInt(0xFFFFULL << 16, 0));
  ASSERT_EQ(0ULL, ReverseBitsUInt(0xFFFFULL << 16, 1));
  ASSERT_EQ(0xFFFFULL, ReverseBitsUInt(0xFFFFULL << 16, 32));
  ASSERT_EQ(0xFFFFULL << 32, ReverseBitsUInt(0xFFFFULL << 16, 64));

  ASSERT_EQ(0ULL, ReverseBitsUInt(0x0000FFFFFFFF0000ULL, 0));
  ASSERT_EQ(0ULL, ReverseBitsUInt(0x0000FFFFFFFF0000ULL, 1));
  ASSERT_EQ(0xFFFFULL, ReverseBitsUInt(0x0000FFFFFFFF0000ULL, 32));
  ASSERT_EQ(0x0000FFFFFFFF0000ULL, ReverseBitsUInt(0x0000FFFFFFFF0000ULL, 64));

  ASSERT_EQ(0ULL, ReverseBitsUInt(0xFFFF0000FFFF0000ULL, 0));
  ASSERT_EQ(0ULL, ReverseBitsUInt(0xFFFF0000FFFF0000ULL, 1));
  ASSERT_EQ(0xFFFFULL, ReverseBitsUInt(0xFFFF0000FFFF0000ULL, 32));
  ASSERT_EQ(0x0000FFFF0000FFFFULL, ReverseBitsUInt(0xFFFF0000FFFF0000ULL, 64));
}

TEST(NumberTheory, MultiplyUIntModLazy) {
  uint64_t mod = 2;
  uint64_t y = 0;
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(0, y, mod));
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(1, y, mod));
  y = 1;
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(0, y, mod));
  ASSERT_EQ(1ULL, MultiplyUIntModLazy(1, y, mod));

  mod = 10;
  y = 0;
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(0, y, mod));
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(1, y, mod));
  y = 1;
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(0, y, mod));
  ASSERT_EQ(1ULL, MultiplyUIntModLazy(1, y, mod));
  y = 6;
  ASSERT_EQ(2ULL, MultiplyUIntModLazy(7, y, mod));
  y = 7;
  ASSERT_EQ(9ULL, MultiplyUIntModLazy(7, y, mod));
  ASSERT_EQ(2ULL, MultiplyUIntModLazy(6, y, mod));

  mod = 2305843009211596801ULL;
  y = 0;
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(0, y, mod));
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(1, y, mod));
  y = 1;
  ASSERT_EQ(0ULL, MultiplyUIntModLazy(0, y, mod));
  ASSERT_EQ(1ULL, MultiplyUIntModLazy(1, y, mod));
  y = 1152921504605798400ULL;
  ASSERT_EQ(576460752302899200ULL,
            MultiplyUIntModLazy(1152921504605798401ULL, y, mod));
  y = 1152921504605798401ULL;
  ASSERT_EQ(576460752302899200ULL,
            MultiplyUIntModLazy(1152921504605798400ULL, y, mod));
  ASSERT_EQ(1729382256908697601ULL,
            MultiplyUIntModLazy(1152921504605798401ULL, y, mod));
  y = 2305843009211596800ULL;
  ASSERT_EQ(2305843009211596802ULL,
            MultiplyUIntModLazy(2305843009211596800ULL, y, mod));
}

}  // namespace ntt
}  // namespace intel
