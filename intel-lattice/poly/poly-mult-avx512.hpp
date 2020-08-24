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

#include "number-theory/number-theory.hpp"
#include "poly/poly-mult.hpp"
#include "util/avx512_util.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

template <int BitShift>
void MultiplyModInPlaceAVX512(uint64_t* operand1, const uint64_t* operand2,
                              const uint64_t n, const uint64_t barrett_hi,
                              const uint64_t barrett_lo,
                              const uint64_t modulus) {
  // TODO(fboemer): Support n % 8 != 0
  LATTICE_CHECK(n % 8 == 0,
                "MultiplyModInPlaceAVX512 supports n % 8 == 0; got n = " << n);
  LATTICE_CHECK((modulus) < MaximumValue(BitShift),
                "Modulus " << (modulus) << " exceeds bit shift bound "
                           << MaximumValue(BitShift));

  auto check_bounds = [](const uint64_t* operand, uint64_t op_len,
                         uint64_t bound) -> bool {
    for (size_t i = 0; i < op_len; ++i) {
      if (operand[i] >= bound) {
        LOG(INFO) << "Operand[ " << i << "] = " << operand[i]
                  << " exceeds bound " << bound;
        return false;
      }
    }
    return true;
  };

  LATTICE_CHECK(check_bounds(operand1, n, modulus),
                "pre-mult value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK(check_bounds(operand2, n, modulus),
                "Value in operand2 exceeds bound " << modulus);
  LATTICE_CHECK(BitShift == 52 || BitShift == 64,
                "Invalid bitshift " << BitShift << "; need 52 or 64");

  __m512i vtwo_pow_52 = _mm512_set1_epi64(1UL << 52);
  __m512i vbarrett_hi = _mm512_set1_epi64(barrett_hi);
  __m512i vbarrett_lo = _mm512_set1_epi64(barrett_lo);

  __m512i vmodulus = _mm512_set1_epi64(modulus);
  __m512i* vp_operand1 = reinterpret_cast<__m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

  for (size_t i = 0; i < n; i += 8) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    __m512i vprod_hi, vprod_lo, vcarry, vtmp2_hi, vtmp2_lo, vtmp3, vtmp1;

    vprod_hi = avx512_multiply_uint64_hi<BitShift>(v_operand1, v_operand2);
    vprod_lo = avx512_multiply_uint64_lo<BitShift>(v_operand1, v_operand2);

    // Reduces product using base 2^BitShift Barrett reduction

    // Multiply input and barrett
    // Round 1
    vcarry = avx512_multiply_uint64_hi<BitShift>(vprod_lo, vbarrett_lo);
    vtmp2_hi = avx512_multiply_uint64_hi<BitShift>(vprod_lo, vbarrett_hi);
    vtmp2_lo = avx512_multiply_uint64_lo<BitShift>(vprod_lo, vbarrett_hi);

    // uint64_t tmp3 = tmp2_hi + AddUInt64(tmp2_lo, carry, &tmp1);
    __m512i vtt;
    if (BitShift == 52) {
      vtmp1 = _mm512_add_epi64(vtmp2_lo, vcarry);
      // Conditional subtraction
      // if (vtmp1 >= 2**52) {
      //   vtmp1 -= 2**52;
      //   vtt = 1;
      //   tmp3 = tmp2_hi + vtt;
      // } else {
      //   vtt = 0;
      //   tmp3 = tmp2_hi;
      // }
      vtt = avx512_cmpgteq_epu64(vtmp1, vtwo_pow_52, 1);
      vtmp1 = avx512_mod_epu64(vtmp1, vtwo_pow_52);
      vtmp3 = _mm512_add_epi64(vtmp2_hi, vtt);
    } else {
      vtt = avx512_add_uint64(vtmp2_lo, vcarry, &vtmp1);
      vtmp3 = _mm512_add_epi64(vtmp2_hi, vtt);
    }

    // Round 2
    vtmp2_hi = avx512_multiply_uint64_hi<BitShift>(vprod_hi, vbarrett_lo);
    vtmp2_lo = avx512_multiply_uint64_lo<BitShift>(vprod_hi, vbarrett_lo);

    // carry = tmp2_hi + AddUInt64(tmp1, tmp2_lo, &tmp1);
    if (BitShift == 52) {
      vtmp1 = _mm512_add_epi64(vtmp1, vtmp2_lo);
      // Conditional subtraction
      // if (vtmp1 >= 2**52) {
      //   vtmp1 -= 2**52;
      //   vtt = 1;
      //   vcarry = tmp2_hi + vtt;
      // } else {
      //   vtt = 0;
      //   vcarry = tmp2_hi;
      // }
      vtt = avx512_cmpgteq_epu64(vtmp1, vtwo_pow_52, 1);
      vtmp1 = avx512_mod_epu64(vtmp1, vtwo_pow_52);
      vcarry = _mm512_add_epi64(vtmp2_hi, vtt);
    } else {
      vtt = avx512_add_uint64(vtmp1, vtmp2_lo, &vtmp1);
      vcarry = _mm512_add_epi64(vtmp2_hi, vtt);
    }

    // This is all we care about
    // tmp1 = prod_hi * barrett_hi + tmp3 + carry;
    vtt = avx512_multiply_uint64_lo<BitShift>(vprod_hi, vbarrett_hi);
    __m512i vtt1 = _mm512_add_epi64(vtmp3, vcarry);

    vtmp1 = _mm512_add_epi64(vtt, vtt1);

    // Barrett subtraction
    // tmp3 = prod_lo - tmp1 * modulus;
    vtt = avx512_multiply_uint64_lo<64>(vtmp1, vmodulus);
    if (BitShift == 52) {
      vprod_lo = avx512_multiply_uint64_lo<64>(v_operand1, v_operand2);
    }
    vtmp3 = _mm512_sub_epi64(vprod_lo, vtt);

    // Conditional subtraction
    __m512i exceeded = avx512_cmpgteq_epu64(vtmp3, vmodulus, modulus);
    __m512i vr = _mm512_sub_epi64(vtmp3, exceeded);

    _mm512_storeu_si512(vp_operand1, vr);

    ++vp_operand1;
    ++vp_operand2;
  }
  LATTICE_CHECK(check_bounds(operand1, n, modulus),
                "post-mult value in operand1 exceeds bound " << modulus);
}

}  // namespace lattice
}  // namespace intel
