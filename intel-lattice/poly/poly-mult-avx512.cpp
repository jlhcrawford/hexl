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

#include <immintrin.h>

#include "number-theory/number-theory.hpp"
#include "poly/poly-mult.hpp"
#include "util/avx512_util.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

void MultiplyModInPlace64AVX512(uint64_t *operand1, const uint64_t *operand2,
                                const uint64_t n, const uint64_t barrett_hi,
                                const uint64_t barrett_lo,
                                const uint64_t modulus) {
  LATTICE_CHECK(
      n % 8 == 0,
      "MultiplyModInPlace64AVX512 supports n % 8 == 0; got n = " << n);

  __m512i vconst_ratio_hi = _mm512_set1_epi64(barrett_hi);
  __m512i vconst_ratio_lo = _mm512_set1_epi64(barrett_lo);
  __m512i vmodulus = _mm512_set1_epi64(modulus);
  __m512i *vp_operand1 = reinterpret_cast<__m512i *>(operand1);
  const __m512i *vp_operand2 = reinterpret_cast<const __m512i *>(operand2);

  for (size_t i = 0; i < n; i += 8) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    __m512i vz1, vz0, vcarry, vtmp21, vtmp20, vtmp3, vtmp1;

    avx512_multiply_uint64(v_operand1, v_operand2, &vz1, &vz0);
    // Reduces z using base 2^64 Barrett reduction

    // Multiply input and const_ratio
    // Round 1
    vcarry = avx512_multiply_uint64_hi<64>(vz0, vconst_ratio_lo);
    avx512_multiply_uint64(vz0, vconst_ratio_hi, &vtmp21, &vtmp20);
    __m512i vtt = avx512_add_uint64(vtmp20, vcarry, &vtmp1);
    vtmp3 = _mm512_add_epi64(vtmp21, vtt);

    // Round 2
    avx512_multiply_uint64(vz1, vconst_ratio_lo, &vtmp21, &vtmp20);
    vtt = avx512_add_uint64(vtmp1, vtmp20, &vtmp1);
    vcarry = _mm512_add_epi64(vtmp21, vtt);

    // This is all we care about
    vtt = _mm512_mullo_epi64(vz1, vconst_ratio_hi);
    __m512i vtt1 = _mm512_add_epi64(vtmp3, vcarry);
    vtmp1 = _mm512_add_epi64(vtt, vtt1);

    // Barrett subtraction
    vtt = _mm512_mullo_epi64(vtmp1, vmodulus);
    vtmp3 = _mm512_sub_epi64(vz0, vtt);

    // Conditional subtraction
    __m512i exceeded = avx512_cmpgteq_epu64(vtmp3, vmodulus, modulus);
    __m512i vr = _mm512_sub_epi64(vtmp3, exceeded);

    _mm512_storeu_si512(vp_operand1, vr);

    ++vp_operand1;
    ++vp_operand2;
  }
}

}  // namespace lattice
}  // namespace intel
