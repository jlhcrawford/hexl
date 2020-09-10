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

#include <immintrin.h>
#include <stdint.h>

#include "number-theory/number-theory.hpp"
#include "poly/poly-mult.hpp"
#include "util/avx512_util.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

template <int BitShift>
void MultiplyModInPlaceAVX512(uint64_t* operand1, const uint64_t* operand2,
                              const uint64_t n, const uint64_t barr_hi,
                              const uint64_t barr_lo, const uint64_t modulus) {
  // TODO(fboemer): Support n % 8 != 0
  LATTICE_CHECK(n % 8 == 0,
                "MultiplyModInPlaceAVX512 supports n % 8 == 0; got n = " << n);
  LATTICE_CHECK((modulus) < MaximumValue(BitShift),
                "Modulus " << (modulus) << " exceeds bit shift bound "
                           << MaximumValue(BitShift));

  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "pre-mult value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus,
                       "Value in operand2 exceeds bound " << modulus);
  LATTICE_CHECK(BitShift == 52 || BitShift == 64,
                "Invalid bitshift " << BitShift << "; need 52 or 64");

  __m512i vbarr_hi = _mm512_set1_epi64(barr_hi);
  __m512i vbarr_lo = _mm512_set1_epi64(barr_lo);

  __m512i vmodulus = _mm512_set1_epi64(modulus);
  __m512i* vp_operand1 = reinterpret_cast<__m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    __m512i vprod_hi, vprod_lo, vrnd1_hi, vrnd2_hi, vrnd2_lo, vrnd3_hi,
        vrnd3_lo, vrnd4_lo, vfloor_lo, vfloor_hi, vresult, vcarry;

    // Compute product
    vprod_hi = _mm512_il_mulhi_epi<BitShift>(v_operand1, v_operand2);
    vprod_lo = _mm512_il_mullo_epi<BitShift>(v_operand1, v_operand2);

    // Reduces product using base 2^BitShift Barrett reduction
    // Each | indicates BitShift-bit chunks
    //
    //                        | barr_hi | barr_lo |
    //      X                 | prod_hi | prod_lo |
    // --------------------------------------------
    //                        | prod_lo x barr_lo | // Round 1
    // +            | prod_lo x barr_hi |           // Round 2
    // +            | prod_hi x barr_lo |           // Round 3
    // +  | barr_hi x prod_hi |                     // Round 4
    // --------------------------------------------
    //              |floor_hi | floor_lo|
    //               \-------/
    //                   \- The only BitShift-bit chunk we care about: vfloor_hi

    // Round 1
    vrnd1_hi = _mm512_il_mulhi_epi<BitShift>(vprod_lo, vbarr_lo);
    // Round 2
    vrnd2_hi = _mm512_il_mulhi_epi<BitShift>(vprod_lo, vbarr_hi);
    vrnd2_lo = _mm512_il_mullo_epi<BitShift>(vprod_lo, vbarr_hi);

    vfloor_hi = _mm512_il_add_epu<BitShift>(vrnd2_lo, vrnd1_hi, &vfloor_lo);
    vfloor_hi = _mm512_add_epi64(vrnd2_hi, vfloor_hi);

    // Round 3
    vrnd3_hi = _mm512_il_mulhi_epi<BitShift>(vprod_hi, vbarr_lo);
    vrnd3_lo = _mm512_il_mullo_epi<BitShift>(vprod_hi, vbarr_lo);

    vfloor_hi = _mm512_add_epi64(vrnd3_hi, vfloor_hi);
    vcarry = _mm512_il_add_epu<BitShift>(vfloor_lo, vrnd3_lo, &vfloor_lo);
    vfloor_hi = _mm512_add_epi64(vfloor_hi, vcarry);

    // Round 4
    vrnd4_lo = _mm512_il_mullo_epi<BitShift>(vprod_hi, vbarr_hi);
    vfloor_hi = _mm512_add_epi64(vrnd4_lo, vfloor_hi);

    // Barrett subtraction
    // result = prod_lo - vfloor_hi * modulus;
    vresult = _mm512_il_mullo_epi<64>(vfloor_hi, vmodulus);
    if (BitShift == 52) {
      vprod_lo = _mm512_il_mullo_epi<64>(v_operand1, v_operand2);
    }
    vresult = _mm512_sub_epi64(vprod_lo, vresult);

    // Conditional subtraction
    // result = (result >= modulus) ? result - modulus : result
    vresult = _mm512_il_small_mod_epi64(vresult, vmodulus);
    _mm512_storeu_si512(vp_operand1, vresult);

    ++vp_operand1;
    ++vp_operand2;
  }
  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "post-mult value in operand1 exceeds bound " << modulus);
}

}  // namespace lattice
}  // namespace intel
