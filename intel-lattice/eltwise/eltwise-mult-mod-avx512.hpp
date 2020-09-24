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

#include <functional>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

// @brief Multiplies two vectors elementwise with modular reduction
// @param operand1 Vector of elements to multiply; stores result
// @param operand2 Vector of elements to multiply
// @param n Number of elements in each vector
// @param barr_hi High 64 bits of Barrett precomputation floor(2^128 /
// modulus)
// @param barr_lo Low 64 bits of Barrett precomputation floor(2^128 /
// modulus)
// @param modulus Modulus with which to perform modular reduction
template <int BitShift>
inline void EltwiseMultModAVX512Int(uint64_t* operand1,
                                    const uint64_t* operand2, uint64_t n,
                                    const uint64_t barr_hi,
                                    const uint64_t barr_lo,
                                    const uint64_t modulus) {
  LATTICE_CHECK((modulus) < MaximumValue(BitShift),
                "Modulus " << (modulus) << " exceeds bit shift bound "
                           << MaximumValue(BitShift));

  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "pre-mult value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus,
                       "Value in operand2 exceeds bound " << modulus);
  LATTICE_CHECK(BitShift == 52 || BitShift == 64,
                "Invalid bitshift " << BitShift << "; need 52 or 64");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative(operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    n -= n_mod_8;
  }

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

// From Function 18, page 19 of https://arxiv.org/pdf/1407.3383.pdf
// See also Algorithm 2/3 of
// https://hal.archives-ouvertes.fr/hal-02552673/document
inline void EltwiseMultModAVX512Float(uint64_t* operand1,
                                      const uint64_t* operand2, uint64_t n,
                                      const uint64_t modulus) {
  LATTICE_CHECK((modulus) < MaximumValue(50),
                "Modulus " << (modulus) << " exceeds bit shift bound "
                           << MaximumValue(50));

  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "pre-mult value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus,
                       "Value in operand2 exceeds bound " << modulus);

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative(operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    n -= n_mod_8;
  }
  __m512d p = _mm512_set1_pd(static_cast<double>(modulus));
  // TODO(fboemer): investigate more closely. Use round toward infinity in
  // computing u. See Proposition 13 at https://arxiv.org/pdf/1407.3383.pdf
  __m512d u = _mm512_set1_pd(static_cast<double>((1.0 / modulus)));

  __m512i* vp_operand1 = reinterpret_cast<__m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

  __m512d zero = _mm512_setzero_pd();

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    // TODO(fboemer): investigate more closely. Use round toward infinity?
    // see Proposition 13 at https://arxiv.org/pdf/1407.3383.pdf
    __m512d x = _mm512_cvt_roundepu64_pd(
        v_operand1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m512d y = _mm512_cvt_roundepu64_pd(
        v_operand2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    __m512d h = _mm512_mul_pd(x, y);
    __m512d l =
        _mm512_fmsub_pd(x, y, h);     // rounding error; h + l == x * y exactly
    __m512d b = _mm512_mul_pd(h, u);  // ~ (x * y) / p
    __m512d c = _mm512_floor_pd(b);
    __m512d d = _mm512_fnmadd_pd(c, p, h);
    __m512d g = _mm512_add_pd(d, l);
    // if g >= p, return g - p;

    // Omit g < 0.0 condition -- see Proposition 13 at
    // https://arxiv.org/pdf/1407.3383.pdf
    // // if g < 0.0, return g + p
    // return g;
    __mmask8 m = _mm512_cmp_pd_mask(g, zero, _CMP_LT_OQ);
    // __mmask8 mm = _mm512_cmp_pd_mask(p, g, _CMP_LE_OQ);
    g = _mm512_mask_add_pd(g, m, g, p);
    // g = _mm512_mask_sub_pd(g, mm, g, p);

    v_operand1 = _mm512_cvt_roundpd_epu64(
        g, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm512_storeu_si512(vp_operand1, v_operand1);

    ++vp_operand1;
    ++vp_operand2;
  }
  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "post-mult value in operand1 exceeds bound " << modulus);
}

template <int BitShift>
inline void EltwiseMultModAVX512Int(uint64_t* operand1,
                                    const uint64_t* operand2, const uint64_t n,
                                    const uint64_t modulus) {
  BarrettFactor<BitShift> bf(modulus);

  EltwiseMultModAVX512Int<BitShift>(operand1, operand2, n, bf.Hi(), bf.Lo(),
                                    modulus);
}

}  // namespace lattice
}  // namespace intel
