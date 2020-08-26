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

#include "poly/poly-mult.hpp"

#include "number-theory/number-theory.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

void MultiplyModInPlace64(uint64_t* operand1, const uint64_t* operand2,
                          const uint64_t n, const uint64_t barrett_hi,
                          const uint64_t barrett_lo, const uint64_t modulus) {
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = 0; i < n; ++i) {
    // Reduces z using base 2^64 Barrett reduction
    uint64_t tmp1;
    uint64_t prod_hi;
    uint64_t prod_lo;
    uint64_t tmp2_hi;
    uint64_t tmp2_lo;

    // Multiply inputs
    MultiplyUInt64(*operand1, *operand2, &prod_hi, &prod_lo);

    // Round 1
    uint64_t carry = MultiplyUInt64Hi<64>(prod_lo, barrett_lo);
    MultiplyUInt64(prod_lo, barrett_hi, &tmp2_hi, &tmp2_lo);
    uint64_t tmp3 = tmp2_hi + AddUInt64(tmp2_lo, carry, &tmp1);

    // Round 2
    MultiplyUInt64(prod_hi, barrett_lo, &tmp2_hi, &tmp2_lo);
    carry = tmp2_hi + AddUInt64(tmp1, tmp2_lo, &tmp1);
    tmp1 = prod_hi * barrett_hi + tmp3 + carry;

    // Barrett subtraction
    tmp3 = prod_lo - tmp1 * modulus;

    // Conditional subtraction
    *operand1 = tmp3 - (modulus & static_cast<uint64_t>(
                                      -static_cast<int64_t>(tmp3 >= modulus)));

    ++operand1;
    ++operand2;
  }
}

void MultiplyModInPlace(uint64_t* operand1, const uint64_t* operand2,
                        const uint64_t n, const uint64_t modulus) {
#ifdef LATTICE_HAS_AVX512IFMA
  // TODO(fboemer): check behavior around 50-52 bits
  if (modulus < (1UL << 50) && (n % 8 == 0)) {
    IVLOG(3, "Calling 52-bit AVX512 MultiplyMod");
    MultiplyModInPlaceAVX512<52>(operand1, operand2, n, modulus);
    return;
  }
#endif
#ifdef LATTICE_HAS_AVX512F
  if (n % 8 == 0) {
    IVLOG(3, "Calling 64-bit AVX512 MultiplyMod");

    MultiplyModInPlaceAVX512<64>(operand1, operand2, n, modulus);
    return;
  }
#endif

  IVLOG(3, "Calling 64-bit default MultiplyMod");
  MultiplyModInPlace64(operand1, operand2, n, modulus);
}

}  // namespace lattice
}  // namespace intel
