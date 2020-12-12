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
void EltwiseMultModAVX512IntLoop8192(__m512i* vp_operand1,
                                     const __m512i* vp_operand2,
                                     __m512i vbarr_lo, __m512i vmodulus) {
  __m512i* vp_out = vp_operand1;
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = 64; i > 0; --i) {
    __m512i x1 = _mm512_loadu_si512(vp_operand1++);
    __m512i y1 = _mm512_loadu_si512(vp_operand2++);
    __m512i x2 = _mm512_loadu_si512(vp_operand1++);
    __m512i y2 = _mm512_loadu_si512(vp_operand2++);
    __m512i x3 = _mm512_loadu_si512(vp_operand1++);
    __m512i y3 = _mm512_loadu_si512(vp_operand2++);
    __m512i x4 = _mm512_loadu_si512(vp_operand1++);
    __m512i y4 = _mm512_loadu_si512(vp_operand2++);
    __m512i x5 = _mm512_loadu_si512(vp_operand1++);
    __m512i y5 = _mm512_loadu_si512(vp_operand2++);
    __m512i x6 = _mm512_loadu_si512(vp_operand1++);
    __m512i y6 = _mm512_loadu_si512(vp_operand2++);
    __m512i x7 = _mm512_loadu_si512(vp_operand1++);
    __m512i y7 = _mm512_loadu_si512(vp_operand2++);
    __m512i x8 = _mm512_loadu_si512(vp_operand1++);
    __m512i y8 = _mm512_loadu_si512(vp_operand2++);
    __m512i x9 = _mm512_loadu_si512(vp_operand1++);
    __m512i y9 = _mm512_loadu_si512(vp_operand2++);
    __m512i x10 = _mm512_loadu_si512(vp_operand1++);
    __m512i y10 = _mm512_loadu_si512(vp_operand2++);
    __m512i x11 = _mm512_loadu_si512(vp_operand1++);
    __m512i y11 = _mm512_loadu_si512(vp_operand2++);
    __m512i x12 = _mm512_loadu_si512(vp_operand1++);
    __m512i y12 = _mm512_loadu_si512(vp_operand2++);
    __m512i x13 = _mm512_loadu_si512(vp_operand1++);
    __m512i y13 = _mm512_loadu_si512(vp_operand2++);
    __m512i x14 = _mm512_loadu_si512(vp_operand1++);
    __m512i y14 = _mm512_loadu_si512(vp_operand2++);
    __m512i x15 = _mm512_loadu_si512(vp_operand1++);
    __m512i y15 = _mm512_loadu_si512(vp_operand2++);
    __m512i x16 = _mm512_loadu_si512(vp_operand1++);
    __m512i y16 = _mm512_loadu_si512(vp_operand2++);

    __m512i zhi1 = _mm512_il_mulhi_epi<64>(x1, y1);
    __m512i zhi2 = _mm512_il_mulhi_epi<64>(x2, y2);
    __m512i zhi3 = _mm512_il_mulhi_epi<64>(x3, y3);
    __m512i zhi4 = _mm512_il_mulhi_epi<64>(x4, y4);
    __m512i zhi5 = _mm512_il_mulhi_epi<64>(x5, y5);
    __m512i zhi6 = _mm512_il_mulhi_epi<64>(x6, y6);
    __m512i zhi7 = _mm512_il_mulhi_epi<64>(x7, y7);
    __m512i zhi8 = _mm512_il_mulhi_epi<64>(x8, y8);
    __m512i zhi9 = _mm512_il_mulhi_epi<64>(x9, y9);
    __m512i zhi10 = _mm512_il_mulhi_epi<64>(x10, y10);
    __m512i zhi11 = _mm512_il_mulhi_epi<64>(x11, y11);
    __m512i zhi12 = _mm512_il_mulhi_epi<64>(x12, y12);
    __m512i zhi13 = _mm512_il_mulhi_epi<64>(x13, y13);
    __m512i zhi14 = _mm512_il_mulhi_epi<64>(x14, y14);
    __m512i zhi15 = _mm512_il_mulhi_epi<64>(x15, y15);
    __m512i zhi16 = _mm512_il_mulhi_epi<64>(x16, y16);

    __m512i zlo1 = _mm512_il_mullo_epi<64>(x1, y1);
    __m512i zlo2 = _mm512_il_mullo_epi<64>(x2, y2);
    __m512i zlo3 = _mm512_il_mullo_epi<64>(x3, y3);
    __m512i zlo4 = _mm512_il_mullo_epi<64>(x4, y4);
    __m512i zlo5 = _mm512_il_mullo_epi<64>(x5, y5);
    __m512i zlo6 = _mm512_il_mullo_epi<64>(x6, y6);
    __m512i zlo7 = _mm512_il_mullo_epi<64>(x7, y7);
    __m512i zlo8 = _mm512_il_mullo_epi<64>(x8, y8);
    __m512i zlo9 = _mm512_il_mullo_epi<64>(x9, y9);
    __m512i zlo10 = _mm512_il_mullo_epi<64>(x10, y10);
    __m512i zlo11 = _mm512_il_mullo_epi<64>(x11, y11);
    __m512i zlo12 = _mm512_il_mullo_epi<64>(x12, y12);
    __m512i zlo13 = _mm512_il_mullo_epi<64>(x13, y13);
    __m512i zlo14 = _mm512_il_mullo_epi<64>(x14, y14);
    __m512i zlo15 = _mm512_il_mullo_epi<64>(x15, y15);
    __m512i zlo16 = _mm512_il_mullo_epi<64>(x16, y16);

    __m512i c1 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo1, zhi1);
    __m512i c2 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo2, zhi2);
    __m512i c3 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo3, zhi3);
    __m512i c4 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo4, zhi4);
    __m512i c5 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo5, zhi5);
    __m512i c6 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo6, zhi6);
    __m512i c7 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo7, zhi7);
    __m512i c8 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo8, zhi8);
    __m512i c9 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo9, zhi9);
    __m512i c10 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo10, zhi10);
    __m512i c11 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo11, zhi11);
    __m512i c12 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo12, zhi12);
    __m512i c13 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo13, zhi13);
    __m512i c14 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo14, zhi14);
    __m512i c15 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo15, zhi15);
    __m512i c16 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo16, zhi16);

    c1 = _mm512_il_mulhi_epi<64>(c1, vbarr_lo);
    c2 = _mm512_il_mulhi_epi<64>(c2, vbarr_lo);
    c3 = _mm512_il_mulhi_epi<64>(c3, vbarr_lo);
    c4 = _mm512_il_mulhi_epi<64>(c4, vbarr_lo);
    c5 = _mm512_il_mulhi_epi<64>(c5, vbarr_lo);
    c6 = _mm512_il_mulhi_epi<64>(c6, vbarr_lo);
    c7 = _mm512_il_mulhi_epi<64>(c7, vbarr_lo);
    c8 = _mm512_il_mulhi_epi<64>(c8, vbarr_lo);
    c9 = _mm512_il_mulhi_epi<64>(c9, vbarr_lo);
    c10 = _mm512_il_mulhi_epi<64>(c10, vbarr_lo);
    c11 = _mm512_il_mulhi_epi<64>(c11, vbarr_lo);
    c12 = _mm512_il_mulhi_epi<64>(c12, vbarr_lo);
    c13 = _mm512_il_mulhi_epi<64>(c13, vbarr_lo);
    c14 = _mm512_il_mulhi_epi<64>(c14, vbarr_lo);
    c15 = _mm512_il_mulhi_epi<64>(c15, vbarr_lo);
    c16 = _mm512_il_mulhi_epi<64>(c16, vbarr_lo);

    __m512i vr1 = _mm512_il_mullo_epi<64>(c1, vmodulus);
    __m512i vr2 = _mm512_il_mullo_epi<64>(c2, vmodulus);
    __m512i vr3 = _mm512_il_mullo_epi<64>(c3, vmodulus);
    __m512i vr4 = _mm512_il_mullo_epi<64>(c4, vmodulus);
    __m512i vr5 = _mm512_il_mullo_epi<64>(c5, vmodulus);
    __m512i vr6 = _mm512_il_mullo_epi<64>(c6, vmodulus);
    __m512i vr7 = _mm512_il_mullo_epi<64>(c7, vmodulus);
    __m512i vr8 = _mm512_il_mullo_epi<64>(c8, vmodulus);
    __m512i vr9 = _mm512_il_mullo_epi<64>(c9, vmodulus);
    __m512i vr10 = _mm512_il_mullo_epi<64>(c10, vmodulus);
    __m512i vr11 = _mm512_il_mullo_epi<64>(c11, vmodulus);
    __m512i vr12 = _mm512_il_mullo_epi<64>(c12, vmodulus);
    __m512i vr13 = _mm512_il_mullo_epi<64>(c13, vmodulus);
    __m512i vr14 = _mm512_il_mullo_epi<64>(c14, vmodulus);
    __m512i vr15 = _mm512_il_mullo_epi<64>(c15, vmodulus);
    __m512i vr16 = _mm512_il_mullo_epi<64>(c16, vmodulus);

    vr1 = _mm512_sub_epi64(zlo1, vr1);
    vr2 = _mm512_sub_epi64(zlo2, vr2);
    vr3 = _mm512_sub_epi64(zlo3, vr3);
    vr4 = _mm512_sub_epi64(zlo4, vr4);
    vr5 = _mm512_sub_epi64(zlo5, vr5);
    vr6 = _mm512_sub_epi64(zlo6, vr6);
    vr7 = _mm512_sub_epi64(zlo7, vr7);
    vr8 = _mm512_sub_epi64(zlo8, vr8);
    vr9 = _mm512_sub_epi64(zlo9, vr9);
    vr10 = _mm512_sub_epi64(zlo10, vr10);
    vr11 = _mm512_sub_epi64(zlo11, vr11);
    vr12 = _mm512_sub_epi64(zlo12, vr12);
    vr13 = _mm512_sub_epi64(zlo13, vr13);
    vr14 = _mm512_sub_epi64(zlo14, vr14);
    vr15 = _mm512_sub_epi64(zlo15, vr15);
    vr16 = _mm512_sub_epi64(zlo16, vr16);

    vr1 = _mm512_il_small_mod_epu64(vr1, vmodulus);
    vr2 = _mm512_il_small_mod_epu64(vr2, vmodulus);
    vr3 = _mm512_il_small_mod_epu64(vr3, vmodulus);
    vr4 = _mm512_il_small_mod_epu64(vr4, vmodulus);
    vr5 = _mm512_il_small_mod_epu64(vr5, vmodulus);
    vr6 = _mm512_il_small_mod_epu64(vr6, vmodulus);
    vr7 = _mm512_il_small_mod_epu64(vr7, vmodulus);
    vr8 = _mm512_il_small_mod_epu64(vr8, vmodulus);
    vr9 = _mm512_il_small_mod_epu64(vr9, vmodulus);
    vr10 = _mm512_il_small_mod_epu64(vr10, vmodulus);
    vr11 = _mm512_il_small_mod_epu64(vr11, vmodulus);
    vr12 = _mm512_il_small_mod_epu64(vr12, vmodulus);
    vr13 = _mm512_il_small_mod_epu64(vr13, vmodulus);
    vr14 = _mm512_il_small_mod_epu64(vr14, vmodulus);
    vr15 = _mm512_il_small_mod_epu64(vr15, vmodulus);
    vr16 = _mm512_il_small_mod_epu64(vr16, vmodulus);

    _mm512_storeu_si512(vp_out++, vr1);
    _mm512_storeu_si512(vp_out++, vr2);
    _mm512_storeu_si512(vp_out++, vr3);
    _mm512_storeu_si512(vp_out++, vr4);
    _mm512_storeu_si512(vp_out++, vr5);
    _mm512_storeu_si512(vp_out++, vr6);
    _mm512_storeu_si512(vp_out++, vr7);
    _mm512_storeu_si512(vp_out++, vr8);
    _mm512_storeu_si512(vp_out++, vr9);
    _mm512_storeu_si512(vp_out++, vr10);
    _mm512_storeu_si512(vp_out++, vr11);
    _mm512_storeu_si512(vp_out++, vr12);
    _mm512_storeu_si512(vp_out++, vr13);
    _mm512_storeu_si512(vp_out++, vr14);
    _mm512_storeu_si512(vp_out++, vr15);
    _mm512_storeu_si512(vp_out++, vr16);
  }
}

// TODO(fboemer): More optimal implementation
template <int BitShift>
void EltwiseMultModAVX512IntLoop16384(__m512i* vp_operand1,
                                      const __m512i* vp_operand2,
                                      __m512i vbarr_lo, __m512i vmodulus) {
  EltwiseMultModAVX512IntLoop8192<BitShift>(vp_operand1, vp_operand2, vbarr_lo,
                                            vmodulus);
  vp_operand1 += 1024;
  vp_operand2 += 1024;
  EltwiseMultModAVX512IntLoop8192<BitShift>(vp_operand1, vp_operand2, vbarr_lo,
                                            vmodulus);
}

// Helper function
template <int BitShift>
void EltwiseMultModAVX512IntLoopDefault(__m512i* vp_operand1,
                                        const __m512i* vp_operand2,
                                        __m512i vbarr_lo, __m512i vmodulus,
                                        uint64_t n) {
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

// Helper function
template <int BitShift>
void EltwiseMultModAVX512IntLoop(__m512i* vp_operand1,
                                 const __m512i* vp_operand2, __m512i vbarr_lo,
                                 __m512i vmodulus, uint64_t n) {
  if (n == 8192) {
    EltwiseMultModAVX512IntLoop8192<BitShift>(vp_operand1, vp_operand2,
                                              vbarr_lo, vmodulus);
  } else if (n == 16384) {
    EltwiseMultModAVX512IntLoop16384<BitShift>(vp_operand1, vp_operand2,
                                               vbarr_lo, vmodulus);
  } else {
    EltwiseMultModAVX512IntLoopDefault<BitShift>(vp_operand1, vp_operand2,
                                                 vbarr_lo, vmodulus, n);
  }
}

template <int BitShift>
void EltwiseMultModAVX512IntLoop8192(__m512i* vp_result,
                                     const __m512i* vp_operand1,
                                     const __m512i* vp_operand2,
                                     __m512i vbarr_lo, __m512i vmodulus) {
  __m512i* vp_out = vp_result;
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = 64; i > 0; --i) {
    __m512i x1 = _mm512_loadu_si512(vp_operand1++);
    __m512i y1 = _mm512_loadu_si512(vp_operand2++);
    __m512i x2 = _mm512_loadu_si512(vp_operand1++);
    __m512i y2 = _mm512_loadu_si512(vp_operand2++);
    __m512i x3 = _mm512_loadu_si512(vp_operand1++);
    __m512i y3 = _mm512_loadu_si512(vp_operand2++);
    __m512i x4 = _mm512_loadu_si512(vp_operand1++);
    __m512i y4 = _mm512_loadu_si512(vp_operand2++);
    __m512i x5 = _mm512_loadu_si512(vp_operand1++);
    __m512i y5 = _mm512_loadu_si512(vp_operand2++);
    __m512i x6 = _mm512_loadu_si512(vp_operand1++);
    __m512i y6 = _mm512_loadu_si512(vp_operand2++);
    __m512i x7 = _mm512_loadu_si512(vp_operand1++);
    __m512i y7 = _mm512_loadu_si512(vp_operand2++);
    __m512i x8 = _mm512_loadu_si512(vp_operand1++);
    __m512i y8 = _mm512_loadu_si512(vp_operand2++);
    __m512i x9 = _mm512_loadu_si512(vp_operand1++);
    __m512i y9 = _mm512_loadu_si512(vp_operand2++);
    __m512i x10 = _mm512_loadu_si512(vp_operand1++);
    __m512i y10 = _mm512_loadu_si512(vp_operand2++);
    __m512i x11 = _mm512_loadu_si512(vp_operand1++);
    __m512i y11 = _mm512_loadu_si512(vp_operand2++);
    __m512i x12 = _mm512_loadu_si512(vp_operand1++);
    __m512i y12 = _mm512_loadu_si512(vp_operand2++);
    __m512i x13 = _mm512_loadu_si512(vp_operand1++);
    __m512i y13 = _mm512_loadu_si512(vp_operand2++);
    __m512i x14 = _mm512_loadu_si512(vp_operand1++);
    __m512i y14 = _mm512_loadu_si512(vp_operand2++);
    __m512i x15 = _mm512_loadu_si512(vp_operand1++);
    __m512i y15 = _mm512_loadu_si512(vp_operand2++);
    __m512i x16 = _mm512_loadu_si512(vp_operand1++);
    __m512i y16 = _mm512_loadu_si512(vp_operand2++);

    __m512i zhi1 = _mm512_il_mulhi_epi<64>(x1, y1);
    __m512i zhi2 = _mm512_il_mulhi_epi<64>(x2, y2);
    __m512i zhi3 = _mm512_il_mulhi_epi<64>(x3, y3);
    __m512i zhi4 = _mm512_il_mulhi_epi<64>(x4, y4);
    __m512i zhi5 = _mm512_il_mulhi_epi<64>(x5, y5);
    __m512i zhi6 = _mm512_il_mulhi_epi<64>(x6, y6);
    __m512i zhi7 = _mm512_il_mulhi_epi<64>(x7, y7);
    __m512i zhi8 = _mm512_il_mulhi_epi<64>(x8, y8);
    __m512i zhi9 = _mm512_il_mulhi_epi<64>(x9, y9);
    __m512i zhi10 = _mm512_il_mulhi_epi<64>(x10, y10);
    __m512i zhi11 = _mm512_il_mulhi_epi<64>(x11, y11);
    __m512i zhi12 = _mm512_il_mulhi_epi<64>(x12, y12);
    __m512i zhi13 = _mm512_il_mulhi_epi<64>(x13, y13);
    __m512i zhi14 = _mm512_il_mulhi_epi<64>(x14, y14);
    __m512i zhi15 = _mm512_il_mulhi_epi<64>(x15, y15);
    __m512i zhi16 = _mm512_il_mulhi_epi<64>(x16, y16);

    __m512i zlo1 = _mm512_il_mullo_epi<64>(x1, y1);
    __m512i zlo2 = _mm512_il_mullo_epi<64>(x2, y2);
    __m512i zlo3 = _mm512_il_mullo_epi<64>(x3, y3);
    __m512i zlo4 = _mm512_il_mullo_epi<64>(x4, y4);
    __m512i zlo5 = _mm512_il_mullo_epi<64>(x5, y5);
    __m512i zlo6 = _mm512_il_mullo_epi<64>(x6, y6);
    __m512i zlo7 = _mm512_il_mullo_epi<64>(x7, y7);
    __m512i zlo8 = _mm512_il_mullo_epi<64>(x8, y8);
    __m512i zlo9 = _mm512_il_mullo_epi<64>(x9, y9);
    __m512i zlo10 = _mm512_il_mullo_epi<64>(x10, y10);
    __m512i zlo11 = _mm512_il_mullo_epi<64>(x11, y11);
    __m512i zlo12 = _mm512_il_mullo_epi<64>(x12, y12);
    __m512i zlo13 = _mm512_il_mullo_epi<64>(x13, y13);
    __m512i zlo14 = _mm512_il_mullo_epi<64>(x14, y14);
    __m512i zlo15 = _mm512_il_mullo_epi<64>(x15, y15);
    __m512i zlo16 = _mm512_il_mullo_epi<64>(x16, y16);

    __m512i c1 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo1, zhi1);
    __m512i c2 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo2, zhi2);
    __m512i c3 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo3, zhi3);
    __m512i c4 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo4, zhi4);
    __m512i c5 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo5, zhi5);
    __m512i c6 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo6, zhi6);
    __m512i c7 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo7, zhi7);
    __m512i c8 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo8, zhi8);
    __m512i c9 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo9, zhi9);
    __m512i c10 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo10, zhi10);
    __m512i c11 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo11, zhi11);
    __m512i c12 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo12, zhi12);
    __m512i c13 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo13, zhi13);
    __m512i c14 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo14, zhi14);
    __m512i c15 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo15, zhi15);
    __m512i c16 = _mm512_il_shrdi_epi64<BitShift - 1>(zlo16, zhi16);

    c1 = _mm512_il_mulhi_epi<64>(c1, vbarr_lo);
    c2 = _mm512_il_mulhi_epi<64>(c2, vbarr_lo);
    c3 = _mm512_il_mulhi_epi<64>(c3, vbarr_lo);
    c4 = _mm512_il_mulhi_epi<64>(c4, vbarr_lo);
    c5 = _mm512_il_mulhi_epi<64>(c5, vbarr_lo);
    c6 = _mm512_il_mulhi_epi<64>(c6, vbarr_lo);
    c7 = _mm512_il_mulhi_epi<64>(c7, vbarr_lo);
    c8 = _mm512_il_mulhi_epi<64>(c8, vbarr_lo);
    c9 = _mm512_il_mulhi_epi<64>(c9, vbarr_lo);
    c10 = _mm512_il_mulhi_epi<64>(c10, vbarr_lo);
    c11 = _mm512_il_mulhi_epi<64>(c11, vbarr_lo);
    c12 = _mm512_il_mulhi_epi<64>(c12, vbarr_lo);
    c13 = _mm512_il_mulhi_epi<64>(c13, vbarr_lo);
    c14 = _mm512_il_mulhi_epi<64>(c14, vbarr_lo);
    c15 = _mm512_il_mulhi_epi<64>(c15, vbarr_lo);
    c16 = _mm512_il_mulhi_epi<64>(c16, vbarr_lo);

    __m512i vr1 = _mm512_il_mullo_epi<64>(c1, vmodulus);
    __m512i vr2 = _mm512_il_mullo_epi<64>(c2, vmodulus);
    __m512i vr3 = _mm512_il_mullo_epi<64>(c3, vmodulus);
    __m512i vr4 = _mm512_il_mullo_epi<64>(c4, vmodulus);
    __m512i vr5 = _mm512_il_mullo_epi<64>(c5, vmodulus);
    __m512i vr6 = _mm512_il_mullo_epi<64>(c6, vmodulus);
    __m512i vr7 = _mm512_il_mullo_epi<64>(c7, vmodulus);
    __m512i vr8 = _mm512_il_mullo_epi<64>(c8, vmodulus);
    __m512i vr9 = _mm512_il_mullo_epi<64>(c9, vmodulus);
    __m512i vr10 = _mm512_il_mullo_epi<64>(c10, vmodulus);
    __m512i vr11 = _mm512_il_mullo_epi<64>(c11, vmodulus);
    __m512i vr12 = _mm512_il_mullo_epi<64>(c12, vmodulus);
    __m512i vr13 = _mm512_il_mullo_epi<64>(c13, vmodulus);
    __m512i vr14 = _mm512_il_mullo_epi<64>(c14, vmodulus);
    __m512i vr15 = _mm512_il_mullo_epi<64>(c15, vmodulus);
    __m512i vr16 = _mm512_il_mullo_epi<64>(c16, vmodulus);

    vr1 = _mm512_sub_epi64(zlo1, vr1);
    vr2 = _mm512_sub_epi64(zlo2, vr2);
    vr3 = _mm512_sub_epi64(zlo3, vr3);
    vr4 = _mm512_sub_epi64(zlo4, vr4);
    vr5 = _mm512_sub_epi64(zlo5, vr5);
    vr6 = _mm512_sub_epi64(zlo6, vr6);
    vr7 = _mm512_sub_epi64(zlo7, vr7);
    vr8 = _mm512_sub_epi64(zlo8, vr8);
    vr9 = _mm512_sub_epi64(zlo9, vr9);
    vr10 = _mm512_sub_epi64(zlo10, vr10);
    vr11 = _mm512_sub_epi64(zlo11, vr11);
    vr12 = _mm512_sub_epi64(zlo12, vr12);
    vr13 = _mm512_sub_epi64(zlo13, vr13);
    vr14 = _mm512_sub_epi64(zlo14, vr14);
    vr15 = _mm512_sub_epi64(zlo15, vr15);
    vr16 = _mm512_sub_epi64(zlo16, vr16);

    vr1 = _mm512_il_small_mod_epu64(vr1, vmodulus);
    vr2 = _mm512_il_small_mod_epu64(vr2, vmodulus);
    vr3 = _mm512_il_small_mod_epu64(vr3, vmodulus);
    vr4 = _mm512_il_small_mod_epu64(vr4, vmodulus);
    vr5 = _mm512_il_small_mod_epu64(vr5, vmodulus);
    vr6 = _mm512_il_small_mod_epu64(vr6, vmodulus);
    vr7 = _mm512_il_small_mod_epu64(vr7, vmodulus);
    vr8 = _mm512_il_small_mod_epu64(vr8, vmodulus);
    vr9 = _mm512_il_small_mod_epu64(vr9, vmodulus);
    vr10 = _mm512_il_small_mod_epu64(vr10, vmodulus);
    vr11 = _mm512_il_small_mod_epu64(vr11, vmodulus);
    vr12 = _mm512_il_small_mod_epu64(vr12, vmodulus);
    vr13 = _mm512_il_small_mod_epu64(vr13, vmodulus);
    vr14 = _mm512_il_small_mod_epu64(vr14, vmodulus);
    vr15 = _mm512_il_small_mod_epu64(vr15, vmodulus);
    vr16 = _mm512_il_small_mod_epu64(vr16, vmodulus);

    _mm512_storeu_si512(vp_out++, vr1);
    _mm512_storeu_si512(vp_out++, vr2);
    _mm512_storeu_si512(vp_out++, vr3);
    _mm512_storeu_si512(vp_out++, vr4);
    _mm512_storeu_si512(vp_out++, vr5);
    _mm512_storeu_si512(vp_out++, vr6);
    _mm512_storeu_si512(vp_out++, vr7);
    _mm512_storeu_si512(vp_out++, vr8);
    _mm512_storeu_si512(vp_out++, vr9);
    _mm512_storeu_si512(vp_out++, vr10);
    _mm512_storeu_si512(vp_out++, vr11);
    _mm512_storeu_si512(vp_out++, vr12);
    _mm512_storeu_si512(vp_out++, vr13);
    _mm512_storeu_si512(vp_out++, vr14);
    _mm512_storeu_si512(vp_out++, vr15);
    _mm512_storeu_si512(vp_out++, vr16);
  }
}

template <int BitShift>
void EltwiseMultModAVX512IntLoop16384(__m512i* vp_result,
                                      const __m512i* vp_operand1,
                                      const __m512i* vp_operand2,
                                      __m512i vbarr_lo, __m512i vmodulus) {
  EltwiseMultModAVX512IntLoop8192<BitShift>(vp_result, vp_operand1, vp_operand2,
                                            vbarr_lo, vmodulus);
  vp_operand1 += 1024;
  vp_operand2 += 1024;
  vp_result += 1024;
  EltwiseMultModAVX512IntLoop8192<BitShift>(vp_result, vp_operand1, vp_operand2,
                                            vbarr_lo, vmodulus);
}

template <int BitShift>
void EltwiseMultModAVX512IntLoopDefault(__m512i* vp_result,
                                        const __m512i* vp_operand1,
                                        const __m512i* vp_operand2,
                                        __m512i vbarr_lo, __m512i vmodulus,
                                        uint64_t n) {
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
    _mm512_storeu_si512(vp_result, vresult);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

template <int BitShift>
void EltwiseMultModAVX512IntLoop(__m512i* vp_result, const __m512i* vp_operand1,
                                 const __m512i* vp_operand2, __m512i vbarr_lo,
                                 __m512i vmodulus, uint64_t n) {
  if (n == 8192) {
    EltwiseMultModAVX512IntLoop8192<BitShift>(vp_result, vp_operand1,
                                              vp_operand2, vbarr_lo, vmodulus);
  } else if (n == 16384) {
    EltwiseMultModAVX512IntLoop16384<BitShift>(vp_result, vp_operand1,
                                               vp_operand2, vbarr_lo, vmodulus);
  } else {
    EltwiseMultModAVX512IntLoopDefault<BitShift>(
        vp_result, vp_operand1, vp_operand2, vbarr_lo, vmodulus, n);
  }
}

void EltwiseMultModAVX512Int(uint64_t* operand1, const uint64_t* operand2,
                             uint64_t n, const uint64_t modulus) {
  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "pre-mult value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus,
                       "Value in operand2 exceeds bound " << modulus);
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");

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
    case 61: {
      EltwiseMultModAVX512IntLoop<61>(vp_operand1, vp_operand2, vbarr_lo,
                                      vmodulus, n);
      break;
    }
    default: {
      // Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document
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
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");

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

void EltwiseMultModAVX512Float(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               const uint64_t modulus) {
  LATTICE_CHECK((modulus) < MaximumValue(50),
                "Modulus " << (modulus) << " exceeds bit shift bound "
                           << MaximumValue(50));
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");

  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "pre-mult value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus,
                       "Value in operand2 exceeds bound " << modulus);
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }
  __m512d p = _mm512_set1_pd(static_cast<double>(modulus));

  // Add epsilon to ensure u * p >= 1.0
  // See Proposition 13 of https://arxiv.org/pdf/1407.3383.pdf
  double ubar = (1.0 + std::numeric_limits<double>::epsilon()) / modulus;
  __m512d u = _mm512_set1_pd(ubar);
  __m512d zero = _mm512_setzero_pd();

  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);
    __m512i v_result;
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

    v_result = _mm512_cvt_roundpd_epu64(
        g, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));

    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
  LATTICE_CHECK_BOUNDS(result, n, modulus,
                       "post-mult value in operand1 exceeds bound " << modulus);
}
void EltwiseMultModAVX512Int(uint64_t* result, const uint64_t* operand1,
                             const uint64_t* operand2, uint64_t n,
                             const uint64_t modulus) {
  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "pre-mult value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus,
                       "Value in operand2 exceeds bound " << modulus);
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  const uint64_t logmod = std::log2l(modulus);
  // modulus < 2**N
  const uint64_t N = logmod + 1;
  uint64_t L = 63 + N;  // Ensures L-N+1 == 64
  uint64_t barr_lo = (uint128_t(1) << L) / modulus;

  __m512i vbarr_lo = _mm512_set1_epi64(barr_lo);
  __m512i vmodulus = _mm512_set1_epi64(modulus);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  // For N < 50, we should prefer EltwiseMultModAVX512Float, so we don't
  // generate a special case for it here
  switch (N) {
    case 50: {
      EltwiseMultModAVX512IntLoop<50>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 51: {
      EltwiseMultModAVX512IntLoop<51>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 52: {
      EltwiseMultModAVX512IntLoop<52>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 53: {
      EltwiseMultModAVX512IntLoop<53>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 54: {
      EltwiseMultModAVX512IntLoop<54>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 55: {
      EltwiseMultModAVX512IntLoop<55>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 56: {
      EltwiseMultModAVX512IntLoop<56>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 57: {
      EltwiseMultModAVX512IntLoop<57>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 58: {
      EltwiseMultModAVX512IntLoop<58>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 59: {
      EltwiseMultModAVX512IntLoop<59>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 60: {
      EltwiseMultModAVX512IntLoop<60>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    case 61: {
      EltwiseMultModAVX512IntLoop<61>(vp_result, vp_operand1, vp_operand2,
                                      vbarr_lo, vmodulus, n);
      break;
    }
    default: {
      // Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document
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
        _mm512_storeu_si512(vp_result, vresult);

        ++vp_operand1;
        ++vp_operand2;
        ++vp_result;
      }
    }
  }

  LATTICE_CHECK_BOUNDS(result, n, modulus,
                       "post-mult value in result exceeds bound " << modulus);
}

}  // namespace lattice
}  // namespace intel
