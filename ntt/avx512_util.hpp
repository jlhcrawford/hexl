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

#include <iostream>

#include "logging/logging.hpp"
#include "ntt.hpp"
#include "number-theory.hpp"

namespace intel {
namespace ntt {

// returns x mod p; assumes x < 2p
// == x >= p ? x - p : x
// == min(x-p, x)
inline __m512i avx512_mod_epu64(__m512i x, __m512i p) {
  return _mm512_min_epu64(x, _mm512_sub_epi64(x, p));
}

// Returns c[i] = hi 64 bits of x[i] * y[i]
inline __m512i avx512_multiply_uint64_hi(__m512i x, __m512i y) {
  // https://stackoverflow.com/questions/28807341/simd-signed-with-unsigned-multiplication-for-64-bit-64-bit-to-128-bit
  __m512i lomask = _mm512_set1_epi64(0x00000000ffffffff);
  __m512i xh =
      _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);  // x0l, x0h, x1l, x1h
  __m512i yh =
      _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);  // y0l, y0h, y1l, y1h
  __m512i w0 = _mm512_mul_epu32(x, y);               // x0l*y0l, x1l*y1l
  __m512i w1 = _mm512_mul_epu32(x, yh);              // x0l*y0h, x1l*y1h
  __m512i w2 = _mm512_mul_epu32(xh, y);              // x0h*y0l, x1h*y0l
  __m512i w3 = _mm512_mul_epu32(xh, yh);             // x0h*y0h, x1h*y1h
  __m512i w0h = _mm512_srli_epi64(w0, 32);
  __m512i s1 = _mm512_add_epi64(w1, w0h);
  __m512i s1l = _mm512_and_si512(s1, lomask);
  __m512i s1h = _mm512_srli_epi64(s1, 32);
  __m512i s2 = _mm512_add_epi64(w2, s1l);
  __m512i s2h = _mm512_srli_epi64(s2, 32);
  __m512i hi1 = _mm512_add_epi64(w3, s1h);
  return _mm512_add_epi64(hi1, s2h);
}

}  // namespace ntt
}  // namespace intel
