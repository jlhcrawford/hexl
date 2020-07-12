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

#include "number-theory.hpp"

#include <bitset>
#include <cassert>
#include <random>

#include "logging/logging.hpp"
#include "util.hpp"

namespace intel {
namespace ntt {

uint64_t InverseUIntMod(uint64_t input, uint64_t modulus) {
  uint64_t a = input % modulus;
  NTT_CHECK(a != 0,
            input << " does not have a InverseMod with modulus " << modulus)
  if (modulus == 1) {
    return 0;
  }

  int64_t m0 = modulus;
  int64_t y = 0;
  int64_t x = 1;
  while (a > 1) {
    // q is quotient
    int64_t q = a / modulus;

    int64_t t = modulus;
    modulus = a % modulus;
    a = t;

    // Update y and x
    t = y;
    y = x - q * y;
    x = t;
  }

  // Make x positive
  if (x < 0) x += m0;

  return uint64_t(x);
}

uint64_t BarrettReduce128(const uint128_t input, const uint64_t modulus) {
  NTT_CHECK(modulus != 0, "modulus == 0")
  return input % modulus;

  // TODO(fboemer): actually use barrett reduction

  // uint64_t tmp1, tmp3;

  // uint128_t const_ratio = Compute64BitConstRatio(modulus);

  // // Multiply input and const_ratio
  // // Round 1
  // uint64_t carry = MultiplyUInt64Hi(Hi64Bits(input), Hi64Bits(const_ratio));

  // uint128_t tmp2 = MultiplyUInt64(Hi64Bits(input), Low64Bits(const_ratio));
  // tmp3 = Low64Bits(tmp2) + AddUInt64NoCarry(Hi64Bits(tmp2), carry);

  // // Round 2
  // tmp2 = MultiplyUInt64(Low64Bits(input), Hi64Bits(const_ratio));
  // carry = Low64Bits(tmp2) + AddUInt64NoCarry(tmp1, Hi64Bits(tmp2));

  // // This is all we care about
  // tmp1 = Low64Bits(input) * Low64Bits(const_ratio) + tmp3 + carry;

  // // Barrett subtraction
  // tmp3 = Hi64Bits(input) - tmp1 * modulus;

  // // One more subtraction is enough
  // return static_cast<std::uint64_t>(tmp3) -
  //        (modulus & static_cast<std::uint64_t>(
  //                       -static_cast<std::int64_t>(tmp3 >= modulus)));
}

uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, const uint64_t modulus) {
  NTT_CHECK(modulus != 0, "modulus == 0");
  uint128_t z = MultiplyUInt64(x, y);
  return BarrettReduce128(z, modulus);
}

// Returns base^exp mod modulus
uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus) {
  base %= modulus;
  uint64_t result = 1;
  while (exp > 0) {
    if (exp & 1) {
      result = MultiplyUIntMod(result, base, modulus);
    }
    base = MultiplyUIntMod(base, base, modulus);
    exp >>= 1;
  }
  return result;
}

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus) {
  if (root == 0) {
    return false;
  }
  NTT_CHECK(IsPowerOfTwo(degree), degree << " not a power of 2");

  IVLOG(4, "IsPrimitiveRoot root " << root << ", degree " << degree
                                   << ", modulus " << modulus);

  // Check if root^(degree/2) == -1 mod modulus
  return PowMod(root, degree / 2, modulus) == (modulus - 1);
}

// Tries to return a primtiive degree-th root of unity
// throw std::invalid_argumentif no root is found
uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus) {
  std::default_random_engine generator;
  std::uniform_int_distribution<uint64_t> distribution(0, modulus - 1);

  // We need to divide modulus-1 by degree to get the size of the quotient group
  uint64_t size_entire_group = modulus - 1;

  // Compute size of quotient group
  uint64_t size_quotient_group = size_entire_group / degree;

  for (int trial = 0; trial < 1000; ++trial) {
    uint64_t root = distribution(generator);

    root = PowMod(root, size_quotient_group, modulus);

    if (IsPrimitiveRoot(root, degree, modulus)) {
      return root;
    }
  }
  NTT_CHECK(false, "no primitive root found for degree "
                       << degree << " modulus " << modulus);
}

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus) {
  assert(IsPowerOfTwo(degree));

  uint64_t root = GeneratePrimitiveRoot(degree, modulus);

  uint64_t generator_sq = MultiplyUIntMod(root, root, modulus);
  uint64_t current_generator = root;

  uint64_t min_root = root;

  // Check if root^(degree/2) == -1 mod modulus
  for (size_t i = 0; i < degree; ++i) {
    if (current_generator < min_root) {
      min_root = current_generator;
    }
    current_generator =
        MultiplyUIntMod(current_generator, generator_sq, modulus);
  }

  return min_root;
}

uint64_t ReverseBitsUInt(uint64_t x, uint64_t bit_width) {
  if (bit_width == 0) {
    return 0;
  }
  uint64_t rev = 0;
  for (int i = bit_width; i > 0; i--) {
    rev |= ((x & 1) << (i - 1));
    x >>= 1;
  }
  return rev;
}

}  // namespace ntt
}  // namespace intel
