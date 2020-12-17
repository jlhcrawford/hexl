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

#pragma once

#include <stdint.h>

#include <limits>
#include <vector>

#include "util/check.hpp"

namespace intel {
namespace lattice {

// Computes floor(2^(2 * BitShift) / modulus)
template <int BitShift>
class BarrettFactor {
 public:
  BarrettFactor() = delete;

  explicit BarrettFactor(uint64_t modulus) {
    LATTICE_CHECK(BitShift == 64 || BitShift == 52,
                  "Unsupport BitShift " << BitShift);
    constexpr uint128_t two_pow_bitshift = uint128_t(1) << BitShift;
    constexpr uint128_t two_pow_twice_bitshift_minus_1 =
        (BitShift == 64) ? uint128_t(-1) : (uint128_t(1) << (2 * BitShift)) - 1;

    // The Barrett factor is actually floor(2^(2 * BitShift) / modulus)
    // But since modulus should be prime,
    // modulus does not divide 2^(2 * BitShift), hence
    // floor(2^(2 * BitShift)/modulus) = floor((2^(2 * BitShift) - 1) / modulus)
    uint128_t barrett_factor = (two_pow_twice_bitshift_minus_1 / modulus);
    m_barrett_hi = barrett_factor >> BitShift;
    m_barrett_lo = barrett_factor % two_pow_bitshift;
  }

  uint64_t Hi() const { return m_barrett_hi; }
  uint64_t Lo() const { return m_barrett_lo; }

 private:
  uint64_t m_barrett_hi;
  uint64_t m_barrett_lo;
};

// Stores an integer on which modular multiplication can be performed more
// efficiently, at the cost of some precomputation.
class MultiplyFactor {
 public:
  MultiplyFactor() = default;

  // Computes and stores the Barrett factor (operand << bit_shift) / modulus
  MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
      : m_operand(operand) {
    LATTICE_CHECK(
        operand <= modulus,
        "operand " << operand << " must be less than modulus " << modulus);
    m_barrett_factor =
        static_cast<uint64_t>((uint128_t(operand) << bit_shift) / modulus);
  }

  inline uint64_t BarrettFactor() const { return m_barrett_factor; }
  inline uint64_t Operand() const { return m_operand; }

 private:
  uint64_t m_operand;
  uint64_t m_barrett_factor;
};

// Returns whether or not num is a power of two
inline bool IsPowerOfTwo(uint64_t num) { return num && !(num & (num - 1)); }

// Returns log2(x) for x a power of 2
inline uint64_t Log2(uint64_t x) {
  LATTICE_CHECK(IsPowerOfTwo(x), x << " not a power of 2");
  uint64_t ret = 0;
  while (x >>= 1) ++ret;
  return ret;
}

// Returns the maximum value that can be represented using bits bits
inline uint64_t MaximumValue(uint64_t bits) {
  LATTICE_CHECK(bits <= 64, "MaximumValue requires bits <= 64; got " << bits);
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

// Returns low 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Lo(uint64_t x0, uint64_t x1, uint64_t y) {
  uint128_t n =
      (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
  uint128_t q = n / y;

  return static_cast<uint64_t>(q);
}

// Returns high 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Hi(uint64_t x0, uint64_t x1, uint64_t y) {
  uint128_t n =
      (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
  uint128_t q = n / y;

  return static_cast<uint64_t>(q >> 64);
}

// Multiplies x * y as 128-bit integer.
// @param prod_hi Stores high 64 bits of product
// @param prod_lo Stores low 64 bits of product
inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
  uint128_t prod = MultiplyUInt64(x, y);
  *prod_hi = static_cast<uint64_t>(prod >> 64);
  *prod_lo = static_cast<uint64_t>(prod);
}

// Return the high 128 minus BitShift bits of the 128-bit product x * y
template <int BitShift>
inline uint64_t MultiplyUInt64Hi(uint64_t x, uint64_t y) {
  uint128_t product = static_cast<uint128_t>(x) * y;
  return (uint64_t)(product >> BitShift);
}

// Returns (x * y) mod modulus
// Assumes x, y < modulus
uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns (x * y) mod modulus
// @param y_precon floor(2**64 / modulus)
uint64_t MultiplyMod(uint64_t x, uint64_t y, uint64_t y_precon,
                     uint64_t modulus);

// Returns (x + y) mod modulus
// Assumes x, y < modulus
uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns (x - y) mod modulus
// Assumes x, y < modulus
uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

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

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
// @param modulus_precon Pre-computed Barrett reduction factor
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y_operand,
                                    uint64_t y_barrett_factor, uint64_t mod) {
  LATTICE_CHECK(y_operand <= mod, "y_operand " << y_operand
                                               << " must be less than modulus "
                                               << mod);
  LATTICE_CHECK(mod <= MaximumValue(BitShift), "Modulus "
                                                   << mod << " exceeds bound "
                                                   << MaximumValue(BitShift));
  LATTICE_CHECK(x <= MaximumValue(BitShift),
                "Operand " << x << " exceeds bound " << MaximumValue(BitShift));

  uint64_t Q = MultiplyUInt64Hi<BitShift>(x, y_barrett_factor);
  return y_operand * x - Q * mod;
}

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y, uint64_t modulus) {
  uint64_t y_barrett = (uint128_t(y) << BitShift) / modulus;
  return MultiplyUIntModLazy<BitShift>(x, y, y_barrett, modulus);
}

// Adds two unsigned 64-bit integers
// @param operand1 Number to add
// @param operand2 Number to add
// @param result Stores the sum
// @return The carry bit
inline unsigned char AddUInt64(uint64_t operand1, uint64_t operand2,
                               uint64_t* result) {
  *result = operand1 + operand2;
  return static_cast<unsigned char>(*result < operand1);
}

// Returns whether or not the input is prime
bool IsPrime(uint64_t n);

// Generates a list of num_primes primes in the range [2^(bit_size,
// 2^(bit_size+1)]. Ensures each prime p satisfies
// p % (2*ntt_size+1)) == 1
// @param num_primes Number of primes to generate
// @param bit_size Bit size of each prime
// @param ntt_size N such that each prime p satisfies p % (2N) == 1. N must be
// a power of two
std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
                                     size_t ntt_size = 1);

}  // namespace lattice
}  // namespace intel
