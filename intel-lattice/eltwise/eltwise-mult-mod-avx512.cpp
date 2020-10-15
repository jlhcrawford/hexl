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

#include "eltwise/eltwise-mult-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include <limits>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

// Helper function
template <int BitShift>
void EltwiseMultModAVX512IntLoop(__m512i* vp_operand1,
                                 const __m512i* vp_operand2, __m512i vbarr_lo,
                                 __m512i vmodulus, uint64_t n) {
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);
    __m512i vprod_hi = _mm512_il_mulhi_epi<64>(v_operand1, v_operand2);
    __m512i vprod_lo = _mm512_il_mullo_epi<64>(v_operand1, v_operand2);
    __m512i c1 = _mm512_il_shrdi_epi64<BitShift - 1>(vprod_lo, vprod_hi);
    __m512i c3 = _mm512_il_mulhi_epi<64>(c1, vbarr_lo);
    __m512i vresult = _mm512_il_mullo_epi<64>(c3, vmodulus);
    vresult = _mm512_sub_epi64(vprod_lo, vresult);
    vresult = _mm512_il_small_mod_epu64(vresult, vmodulus);
    _mm512_storeu_si512(vp_operand1, vresult);

    ++vp_operand1;
    ++vp_operand2;
  }
}

void EltwiseMultModAVX512Int(uint64_t* operand1, const uint64_t* operand2,
                             uint64_t n, const uint64_t modulus) {
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

  const uint64_t logmod = std::log2l(modulus);
  // modulus < 2**N
  const uint64_t N = logmod + 1;
  uint64_t L = 63 + N;  // Ensures L-N+1 == 64
  uint64_t barr_lo = (uint128_t(1) << L) / modulus;

  __m512i vbarr_lo = _mm512_set1_epi64(barr_lo);
  __m512i vmodulus = _mm512_set1_epi64(modulus);
  __m512i* vp_operand1 = reinterpret_cast<__m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

  // For N < 50, we should prefer EltwiseMultModAVX512Float, so we don't
  // generate a special case for it here
  switch (N) {
    case 50: {
      EltwiseMultModAVX512IntLoop<50>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 51: {
      EltwiseMultModAVX512IntLoop<51>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 52: {
      EltwiseMultModAVX512IntLoop<52>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 53: {
      EltwiseMultModAVX512IntLoop<53>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 54: {
      EltwiseMultModAVX512IntLoop<54>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 55: {
      EltwiseMultModAVX512IntLoop<55>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 56: {
      EltwiseMultModAVX512IntLoop<56>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 57: {
      EltwiseMultModAVX512IntLoop<57>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 58: {
      EltwiseMultModAVX512IntLoop<58>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 59: {
      EltwiseMultModAVX512IntLoop<59>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    case 60: {
      EltwiseMultModAVX512IntLoop<60>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    default: {
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
      for (size_t i = n / 8; i > 0; --i) {
        __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
        __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

        // Compute product
        __m512i vprod_hi = _mm512_il_mulhi_epi<64>(v_operand1, v_operand2);
        __m512i vprod_lo = _mm512_il_mullo_epi<64>(v_operand1, v_operand2);

        __m512i c1 = _mm512_il_shrdi_epi64(vprod_lo, vprod_hi, N - 1);

        // L - N + 1 == 64, so we only need high 64 bits
        __m512i c3 = _mm512_il_mulhi_epi<64>(c1, vbarr_lo);

        // C4 = prod_lo - (p * c3)_lo
        __m512i vresult = _mm512_il_mullo_epi<64>(c3, vmodulus);
        vresult = _mm512_sub_epi64(vprod_lo, vresult);

        // Conditional subtraction
        vresult = _mm512_il_small_mod_epu64(vresult, vmodulus);
        _mm512_storeu_si512(vp_operand1, vresult);

        ++vp_operand1;
        ++vp_operand2;
      }
    }
  }

  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "post-mult value in operand1 exceeds bound " << modulus);
}

void EltwiseMultModAVX512Float(uint64_t* operand1, const uint64_t* operand2,
                               uint64_t n, const uint64_t modulus) {
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

  // Add epsilon to ensure u * p >= 1.0
  // See Proposition 13 of https://arxiv.org/pdf/1407.3383.pdf
  double ubar = (1.0 + std::numeric_limits<double>::epsilon()) / modulus;
  __m512d u = _mm512_set1_pd(ubar);
  __m512d zero = _mm512_setzero_pd();

  __m512i* vp_operand1 = reinterpret_cast<__m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    __m512d x = _mm512_cvt_roundepu64_pd(
        v_operand1, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
    __m512d y = _mm512_cvt_roundepu64_pd(
        v_operand2, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));

    __m512d h = _mm512_mul_pd(x, y);
    __m512d l = _mm512_fmsub_pd(x, y, h);  // rounding error; h + l == x * y
    __m512d b = _mm512_mul_pd(h, u);       // ~ (x * y) / p
    __m512d c = _mm512_floor_pd(b);        // ~ floor(x * y / p)
    __m512d d = _mm512_fnmadd_pd(c, p, h);
    __m512d g = _mm512_add_pd(d, l);
    __mmask8 m = _mm512_cmp_pd_mask(g, zero, _CMP_LT_OQ);
    g = _mm512_mask_add_pd(g, m, g, p);

    v_operand1 = _mm512_cvt_roundpd_epu64(
        g, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));

    _mm512_storeu_si512(vp_operand1, v_operand1);

    ++vp_operand1;
    ++vp_operand2;
  }
  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "post-mult value in operand1 exceeds bound " << modulus);
}

}  // namespace lattice
}  // namespace intel
