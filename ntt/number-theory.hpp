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

#include <limits>
#include <vector>

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

// Returns the maximum value that can be represented using bits bits.
inline uint64_t MaximumValue(uint64_t bits) {
  NTT_CHECK(bits <= 64, "MaximumValue requires bits <= 64; got " << bits);
  if (bits == 64) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (1UL << bits) - 1;
}

// Reverses the bits
uint64_t ReverseBitsUInt(uint64_t x, uint64_t bits);

// Returns a^{-1} mod modulus
uint64_t InverseUIntMod(uint64_t a, uint64_t modulus);

// Return x * y as 128-bit integer
inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
  return static_cast<uint128_t>(x) * y;
}

// Multiply packed unsigned 52-bit integers in x and y to form a 104-bit
// intermediate result. Return the high 52-bit unsigned integer stored in an
// unsigned 64-bit integer
template <int BitShift>
inline uint64_t MultiplyUInt64Hi(uint64_t x, uint64_t y) {
  uint128_t product = static_cast<uint128_t>(x) * y;
  return (uint64_t)(product >> BitShift);
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

class MultiplyFactor {
 public:
  MultiplyFactor() = default;
  MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
      : m_operand(operand) {
    NTT_CHECK(operand <= modulus, "operand " << operand
                                             << " must be less than modulus "
                                             << modulus);
    m_barrett_factor =
        static_cast<uint64_t>((uint128_t(operand) << bit_shift) / modulus);
  }

  inline uint64_t BarrettFactor() const { return m_barrett_factor; }
  inline uint64_t Operand() const { return m_operand; }

 private:
  uint64_t m_operand;
  uint64_t m_barrett_factor;
};

// Computes (x * y) mod modulus
// @param modulus_precon Pre-computed Barrett reduction factor
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(const uint64_t x, const uint64_t y_operand,
                                    uint64_t const y_barrett_factor,
                                    const uint64_t mod) {
  NTT_CHECK(y_operand <= mod,
            "y_operand " << y_operand << " must be less than modulus " << mod);
  NTT_CHECK(mod <= MaximumValue(BitShift),
            "Modulus " << mod << " exceeds bound " << MaximumValue(BitShift));
  NTT_CHECK(x <= MaximumValue(BitShift),
            "Operand " << x << " exceeds bound " << MaximumValue(BitShift));

  uint64_t tmp1 = MultiplyUInt64Hi<BitShift>(x, y_barrett_factor);
  return y_operand * x - tmp1 * mod;
}

template <int BitShift>
inline uint64_t MultiplyUIntModLazy(const uint64_t x, const uint64_t y,
                                    const uint64_t modulus) {
  const uint64_t y_barrett = (uint128_t(y) << BitShift) / modulus;
  return MultiplyUIntModLazy<BitShift>(x, y, y_barrett, modulus);
}

// Returns whether or not the input is prime
inline bool IsPrime(const uint64_t n) {
  static const std::vector<uint64_t> as{2,  3,  5,  7,  11, 13,
                                        17, 19, 23, 29, 31, 37};

  for (const uint64_t a : as) {
    if (n == a) return true;
    if (n % a == 0) return false;
  }

  // Miller-Rabin primality test
  // n < 2^64, so it is enough to test a=2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
  // and 37. See
  // https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Testing_against_small_sets_of_bases

  // Write n == 2**r * d + 1 with d odd.
  uint64_t r = 63;
  while (r > 0) {
    uint64_t two_pow_r = (1UL << r);
    if ((n - 1) % two_pow_r == 0) {
      break;
    }
    --r;
  }
  NTT_CHECK(r != 0, "Error factoring n " << n);
  uint64_t d = (n - 1) / (1UL << r);

  NTT_CHECK(n == (1UL << r) * d + 1, "Error factoring n " << n);
  NTT_CHECK(d % 2 == 1, "d is even");

  for (const uint64_t a : as) {
    uint64_t x = PowMod(a, d, n);
    if ((x == 1) || (x == n - 1)) {
      continue;
    }

    bool prime = false;
    for (uint64_t i = 1; i < r; ++i) {
      x = PowMod(x, 2, n);
      if (x == n - 1) {
        prime = true;
      }
    }
    if (!prime) {
      return false;
    }
  }
  return true;
}

// Generates a list of num_primes primes in the range [2^bit_size,
// 2^(bit_size+1)]. Ensures each prime p satisfies
// p % (2*ntt_size+1)) == 1
// @param num_primes Number of primes to generate
// @param bit_size Bit size of each prime
// @param ntt_size N such that each prime p satisfies p % (2N) == 1. N must be
// a power of two
inline std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
                                            size_t ntt_size = 1) {
  NTT_CHECK(num_primes > 0, "num_primes == 0");
  NTT_CHECK(IsPowerOfTwo(ntt_size),
            "ntt_size " << ntt_size << " is not a power of two");
  NTT_CHECK(Log2(ntt_size) < bit_size,
            "log2(ntt_size) " << Log2(ntt_size)
                              << " should be less than bit_size " << bit_size);

  uint64_t value = (1UL << bit_size) + 1;

  std::vector<uint64_t> ret;

  while (value < (1UL << (bit_size + 1))) {
    if (IsPrime(value)) {
      ret.emplace_back(value);
      if (ret.size() == num_primes) {
        return ret;
      }
    }
    value += 2 * ntt_size;
  }

  NTT_CHECK(false, "Failed to find enough primes");
  return ret;
}

}  // namespace ntt
}  // namespace intel
