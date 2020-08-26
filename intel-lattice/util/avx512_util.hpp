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
#include <vector>

#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

// Returns the unsigned 64-bit integer values in x as a vector
inline std::vector<uint64_t> ExtractValues(__m512i x) {
  __m256i x0 = _mm512_extracti64x4_epi64(x, 0);
  __m256i x1 = _mm512_extracti64x4_epi64(x, 1);

  std::vector<uint64_t> xs{static_cast<uint64_t>(_mm256_extract_epi64(x0, 0)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x0, 1)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x0, 2)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x0, 3)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 0)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 1)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 2)),
                           static_cast<uint64_t>(_mm256_extract_epi64(x1, 3))};

  return xs;
}

// Checks all values in a vector are strictly less than bound
// Returns true
template <typename T>
inline bool CheckBounds(const T* values, size_t num_values, T bound) {
  // Avoid unused variable warnings
  (void)values;
  (void)num_values;
  (void)bound;
  LATTICE_CHECK(
      [&]() {
        for (size_t i = 0; i < num_values; ++i) {
          if (values[i] >= bound) return false;
        }
        return true;
      }(),
      "Value in " << std::vector<T>(values, values + num_values)
                  << " exceeds bound " << bound);
  return true;
}

// Checks all 64-bit values in x are less than bound
// Returns true
inline bool CheckBounds(__m512i x, uint64_t bound) {
  return CheckBounds(ExtractValues(x).data(), 512 / 64, bound);
}

// Multiply packed unsigned BitShift-bit integers in each 64-bit element of x
// and y to form a 2*BitShift-bit intermediate result.
// Returns the high BitShift-bit unsigned integer from the intermediate result
template <int BitShift>
inline __m512i _mm512_il_mulhi_epi(__m512i x, __m512i y);

template <>
inline __m512i _mm512_il_mulhi_epi<64>(__m512i x, __m512i y) {
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

#ifdef LATTICE_HAS_AVX512IFMA
template <>
inline __m512i _mm512_il_mulhi_epi<52>(__m512i x, __m512i y) {
  LATTICE_CHECK(CheckBounds(x, MaximumValue(52)), "");
  LATTICE_CHECK(CheckBounds(y, MaximumValue(52)), "");
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52hi_epu64(zero, x, y);
}
#endif

// Multiply packed unsigned BitShift-bit integers in each 64-bit element of x
// and y to form a 104-bit intermediate result.
// Returns the low BitShift-bit unsigned integer from the intermediate result
template <int BitShift>
inline __m512i _mm512_il_mullo_epi(__m512i x, __m512i y);

template <>
inline __m512i _mm512_il_mullo_epi<64>(__m512i x, __m512i y) {
  return _mm512_mullo_epi64(x, y);
}

#ifdef LATTICE_HAS_AVX512IFMA
template <>
inline __m512i _mm512_il_mullo_epi<52>(__m512i x, __m512i y) {
  LATTICE_CHECK(CheckBounds(x, MaximumValue(52)), "");
  LATTICE_CHECK(CheckBounds(y, MaximumValue(52)), "");
  __m512i zero = _mm512_set1_epi64(0);
  return _mm512_madd52lo_epu64(zero, x, y);
}
#endif

// Returns x mod p; assumes x < 2p
// x mod p == x >= p ? x - p : x
//         == min(x - p, x)
inline __m512i _mm512_il_mod_epi64(__m512i x, __m512i p) {
  return _mm512_min_epu64(x, _mm512_sub_epi64(x, p));
}

// Returns c[i] = a[i] >= b[i] ? match_value : 0
inline __m512i _mm512_il_cmpge_epu64(__m512i a, __m512i b,
                                     uint64_t match_value) {
  __mmask8 mask = _mm512_cmpge_epu64_mask(a, b);
  return _mm512_maskz_broadcastq_epi64(mask, _mm_set1_epi64x(match_value));
}

// Returns c[i] = a[i] < b[i] ? match_value : 0
inline __m512i _mm512_il_cmplt_epu64(__m512i a, __m512i b,
                                     uint64_t match_value) {
  __mmask8 mask = _mm512_cmplt_epu64_mask(a, b);
  return _mm512_maskz_broadcastq_epi64(mask, _mm_set1_epi64x(match_value));
}

// Computes x + y mod 2^BitShift and stores the result in c.
// Returns the overflow bit
template <int BitShift>
inline __m512i _mm512_il_add_epu(__m512i x, __m512i y, __m512i* c);

template <>
inline __m512i _mm512_il_add_epu<64>(__m512i x, __m512i y, __m512i* c) {
  *c = _mm512_add_epi64(x, y);
  return _mm512_il_cmplt_epu64(*c, x, 1);
}

template <>
inline __m512i _mm512_il_add_epu<52>(__m512i x, __m512i y, __m512i* c) {
  __m512i vtwo_pow_52 = _mm512_set1_epi64(1UL << 52);
  __m512i sum = _mm512_add_epi64(x, y);
  __m512i carry = _mm512_il_cmpge_epu64(sum, vtwo_pow_52, 1);
  *c = _mm512_il_mod_epi64(sum, vtwo_pow_52);
  return carry;
}

}  // namespace lattice
}  // namespace intel
