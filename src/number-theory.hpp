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

#pragma once

#include "logging/logging.hpp"
#include "util.hpp"

namespace intel {
namespace ntt {

inline bool IsPowerOfTwo(uint64_t num) { return num && !(num & (num - 1)); }

// Returns log2(x) for x a power of 2
inline uint64_t Log2(uint64_t x) {
  NTT_CHECK(IsPowerOfTwo(x), x << " not a power of 2");
  uint64_t ret = 0;
  while (x >>= 1) ++ret;
  return ret;
}

// Reverses the bits
uint64_t ReverseBitsUInt(uint64_t x, uint64_t bits);

// Returns a^{-1} mod modulus
uint64_t InverseUIntMod(uint64_t a, uint64_t modulus);

inline uint64_t Compute64BitConstRatio(uint64_t modulus) {
  NTT_CHECK(false, "TODO");
  // uint128_t tmp = 1 << (2 * 30 + 3);
  // return uint64_t(tmp / static_cast<uint128_t>(modulus));
}

inline uint64_t AddUInt64NoCarry(uint64_t x, uint64_t y) { return x + y; }

// Reduces input using base 2^64 Barrett reduction
// input allocation size must be 128 bits
// modulus <= 63 bits
uint64_t BarrettReduce128(const uint128_t input, const uint64_t modulus);

inline uint64_t Hi64Bits(uint128_t x) { return (uint64_t)(x >> 64); }
inline uint64_t Low64Bits(uint128_t x) { return (uint64_t)(x); }

// Return x * y as 128-bit integer
inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
  return static_cast<uint128_t>(x) * y;
}

// Returns hi 64 bits of x*y
inline uint64_t MultiplyUInt64Hi(uint64_t x, uint64_t y) {
  uint128_t product = static_cast<uint128_t>(x) * y;
  return (uint64_t)(product >> 64);
}

uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, const uint64_t modulus);

// Returns base^exp mod modulus
uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus);

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus);

// Tries to return a primtiive degree-th root of unity
// Returns -1 if no root is found
uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus);

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus);

inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y,
                                    const uint64_t modulus) {
  NTT_CHECK(y <= modulus,
            "y " << y << " must be less than modulus " << modulus);

  const uint64_t y_quotient =
      (uint128_t(y) << 64) / modulus;  // TODO(fboemer): precompute

  const uint64_t p = modulus;
  uint64_t tmp1 = MultiplyUInt64Hi(x, y_quotient);

  return y * x - tmp1 * p;
}

}  // namespace ntt
}  // namespace intel
