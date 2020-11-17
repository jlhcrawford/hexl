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

#include <functional>
#include <vector>

#include "logging/logging.hpp"
#include "ntt/ntt-internal.hpp"
#include "ntt/ntt.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace lattice {

// Returns
// *out1 =  _mm512_set_epi64(arg[15], arg[13], arg[11], arg[9],
//                           arg[7], arg[5], arg[3], arg[1], arg[0]);
// *out2 =  _mm512_set_epi64(arg[14], arg[12], arg[10], arg[8],
//                           arg[6], arg[4], arg[2], arg[0])
inline void LoadInterleavedT1(const uint64_t* arg, __m512i* out1,
                              __m512i* out2) {
  const __m512i vperm_hi_idx = _mm512_set_epi64(6, 4, 2, 0, 7, 5, 3, 1);
  const __m512i vperm_lo_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  // 7, 6, 5, 4, 3, 2, 1, 0
  __m512i v_7to0 = _mm512_loadu_si512(arg_512++);
  // 15, 14, 13, 12, 11, 10, 9, 8
  __m512i v_15to8 = _mm512_loadu_si512(arg_512);
  // 7, 5, 3, 1, 6, 4, 2, 0
  __m512i perm_lo = _mm512_permutexvar_epi64(vperm_lo_idx, v_7to0);
  // 14, 12, 10, 8, 15, 13, 11, 9
  __m512i perm_hi = _mm512_permutexvar_epi64(vperm_hi_idx, v_15to8);

  *out1 = _mm512_mask_blend_epi64(0b00001111, perm_hi, perm_lo);
  *out2 = _mm512_mask_blend_epi64(0b11110000, perm_hi, perm_lo);
  *out2 = _mm512_permutexvar_epi64(vperm2_idx, *out2);
}

// Given inputs
// @param arg1 = (15, 13, 11, 9, 7, 5, 3, 1)
// @param arg2 = (14, 12, 10, 8, 6, 4, 2, 0)
// Writes out[i] = i for i=0, ..., 15
inline void WriteInterleavedT1(__m512i arg1, __m512i arg2, __m512i* out) {
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i v_X_out_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  const __m512i v_Y_out_idx = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);

  // Reverse the permutation
  // v_X (15, 13, 11, 9, 7, 5, 3, 1) => (15, 14, 13, 12, 11, 10, 9, 8)
  // v_Y (14, 12, 10, 8, 6, 4, 2, 0) => (7, 6, 5, 4, 3, 2, 1, 0)

  // v_Y => (6, 4, 2, 0, 14, 12, 10, 8)
  arg2 = _mm512_permutexvar_epi64(vperm2_idx, arg2);
  // 6, 4, 2, 0, 7, 5, 3, 1
  __m512i perm_lo = _mm512_mask_blend_epi64(0b00001111, arg1, arg2);
  // 15, 13, 11, 9, 14, 12, 10, 8
  __m512i perm_hi = _mm512_mask_blend_epi64(0b11110000, arg1, arg2);
  arg1 = _mm512_permutexvar_epi64(v_X_out_idx, perm_hi);
  arg2 = _mm512_permutexvar_epi64(v_Y_out_idx, perm_lo);

  _mm512_storeu_si512(out++, arg1);
  _mm512_storeu_si512(out, arg2);
}

// Returns
// *out1 =  _mm512_set_epi64(arg[13], arg[12], arg[9], arg[8],
//                           arg[5], arg[4], arg[1], arg[0]);
// *out2 =  _mm512_set_epi64(arg[15], arg[14], arg[11], arg[10],
//                           arg[7], arg[6], arg[3], arg[2]);
inline void LoadInterleavedT2(const uint64_t* arg, __m512i* out1,
                              __m512i* out2) {
  const __m512i vperm_hi_idx = _mm512_set_epi64(5, 4, 1, 0, 7, 6, 3, 2);
  const __m512i vperm_lo_idx = _mm512_set_epi64(7, 6, 3, 2, 5, 4, 1, 0);
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);

  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  // 7, 6, 5, 4, 3, 2, 1, 0
  __m512i v_7to0 = _mm512_loadu_si512(arg_512++);
  // 15, 14, 13, 12, 11, 10, 9, 8
  __m512i v_15to8 = _mm512_loadu_si512(arg_512);
  // 7, 5, 3, 1, 6, 4, 2, 0
  __m512i perm_lo = _mm512_permutexvar_epi64(vperm_lo_idx, v_7to0);
  // 14, 12, 10, 8, 15, 13, 11, 9
  __m512i perm_hi = _mm512_permutexvar_epi64(vperm_hi_idx, v_15to8);
  *out1 = _mm512_mask_blend_epi64(0b00001111, perm_hi, perm_lo);
  *out2 = _mm512_mask_blend_epi64(0b11110000, perm_hi, perm_lo);
  *out2 = _mm512_permutexvar_epi64(vperm2_idx, *out2);
}

// Given inputs
// @param arg1 = (15, 14, 11, 10, 7, 6, 3, 2)
// @param arg2 = (13, 12, 9,  8,  5, 4, 1, 0)
// Writes out[i] = i for i=0, ..., 15
inline void WriteInterleavedT2(__m512i arg1, __m512i arg2, __m512i* out) {
  const __m512i vperm_lo_idx = _mm512_set_epi64(7, 6, 3, 2, 5, 4, 1, 0);
  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  const __m512i v_Y_out_idx = _mm512_set_epi64(3, 2, 7, 6, 1, 0, 5, 4);
  // Perform reverse permutation
  // v_X (13, 12, 9, 8, 5, 4, 1, 0) => (15, 14, 13, 12, 11, 10, 9, 8)
  // v_Y (15, 14, 11, 10, 7, 6, 3, 2) => (7, 6, 5, 4, 3, 2, 1, 0)

  // v_Y => (10, 11, 14, 15, 2, 3, 6, 7)
  arg2 = _mm512_permutexvar_epi64(vperm2_idx, arg2);
  // 0, 1, 4, 5, 2, 3, 6, 7
  __m512i perm_lo = _mm512_mask_blend_epi64(0b11110000, arg1, arg2);
  // 10, 11, 14, 15, 8, 9, 12, 13
  __m512i perm_hi = _mm512_mask_blend_epi64(0b00001111, arg1, arg2);
  // 15, 14, 13, 12, 11, 10, 9, 8
  arg1 = _mm512_permutexvar_epi64(vperm_lo_idx, perm_lo);
  // 7, 6, 5, 4, 3, 2, 1, 0
  arg2 = _mm512_permutexvar_epi64(v_Y_out_idx, perm_hi);

  _mm512_storeu_si512(out++, arg1);
  _mm512_storeu_si512(out, arg2);
}

// Returns
// *out1 =  _mm512_set_epi64(arg[11], arg[10], arg[9], arg[8],
//                           arg[3], arg[2], arg[1], arg[0]);
// *out2 =  _mm512_set_epi64(arg[15], arg[14], arg[13], arg[12],
//                           arg[7], arg[6], arg[4], arg[5]);
inline void LoadInterleavedT4(const uint64_t* arg, __m512i* out1,
                              __m512i* out2) {
  const __m512i* arg_512 = reinterpret_cast<const __m512i*>(arg);

  const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
  __m512i v_7to0 = _mm512_loadu_si512(arg_512++);
  __m512i v_15to8 = _mm512_loadu_si512(arg_512);
  __m512i perm_hi = _mm512_permutexvar_epi64(vperm2_idx, v_15to8);
  *out1 = _mm512_mask_blend_epi64(0b0001111, perm_hi, v_7to0);
  *out2 = _mm512_mask_blend_epi64(0b11110000, perm_hi, v_7to0);
  *out2 = _mm512_permutexvar_epi64(vperm2_idx, *out2);
}

// Given inputs
// @param arg1 = (15, 14, 13, 12, 11, 10, 9, 8)
// @param arg2 = (7, 6, 5, 4, 3, 2, 1, 0)
// Writes out[i] = i for i=0, ..., 15
inline void WriteInterleavedT4(__m512i arg1, __m512i arg2, __m512i* out) {
  __m256i x0 = _mm512_extracti64x4_epi64(arg1, 0);
  __m256i x1 = _mm512_extracti64x4_epi64(arg1, 1);
  __m256i y0 = _mm512_extracti64x4_epi64(arg2, 0);
  __m256i y1 = _mm512_extracti64x4_epi64(arg2, 1);
  __m256i* out_256 = reinterpret_cast<__m256i*>(out);
  _mm256_storeu_si256(out_256++, x0);
  _mm256_storeu_si256(out_256++, y0);
  _mm256_storeu_si256(out_256++, x1);
  _mm256_storeu_si256(out_256++, y1);
}

// Returns _mm512_set_epi64(arg[3], arg[3], arg[2], arg[2],
//                          arg[1], arg[1], arg[0], arg[0]);
inline __m512i LoadWOpT2(const void* arg) {
  const __m512i vperm_w_idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);

  __m256i v_W_op_256 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arg));
  __m512i v_W_op = _mm512_broadcast_i64x4(v_W_op_256);
  v_W_op = _mm512_permutexvar_epi64(vperm_w_idx, v_W_op);

  return v_W_op;
}

// Returns _mm512_set_epi64(arg[1], arg[1], arg[1], arg[1],
//                          arg[0], arg[0], arg[0], arg[0]);
inline __m512i LoadWOpT4(const void* arg) {
  const __m512i vperm_w_idx = _mm512_set_epi64(1, 1, 1, 1, 0, 0, 0, 0);

  __m128i v_W_op_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(arg));
  __m512i v_W_op = _mm512_broadcast_i64x2(v_W_op_128);
  v_W_op = _mm512_permutexvar_epi64(vperm_w_idx, v_W_op);

  return v_W_op;
}

}  // namespace lattice
}  // namespace intel
