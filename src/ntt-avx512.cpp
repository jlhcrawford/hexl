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

#include <iostream>

#include "avx512_util.hpp"
#include "logging/logging.hpp"
#include "ntt.hpp"
#include "number-theory.hpp"
namespace intel {
namespace ntt {

// based on
// https://github.com/microsoft/SEAL/blob/master/native/src/seal/util/ntt.cpp#L200
void NTT::ForwardTransformToBitReverse(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements) {
  uint64_t twice_mod = mod << 1;

  __m512i v_modulus = _mm512_set1_epi64(mod);
  __m512i v_twice_mod = _mm512_set1_epi64(twice_mod);

  size_t n = degree;
  size_t t = (n >> 1);

  uint64_t* input = elements;

  for (size_t m = 1; m < n; m <<= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = root_of_unity_powers[m + i];
      const uint64_t W_precon = precon_root_of_unity_powers[m + i];

      uint64_t* X = input + j1;
      uint64_t* Y = X + t;
      uint64_t tx;
      uint64_t Q;

      if (j2 - j1 < 8) {
#pragma unroll 4
        for (size_t j = j1; j < j2; j++) {
          // The Harvey butterfly: assume X, Y in [0, 2p), and return X', Y' in
          // [0, 4p).
          // X', Y' = X + WY, X - WY (mod p).
          tx = *X - (twice_mod & static_cast<uint64_t>(
                                     -static_cast<int64_t>(*X >= twice_mod)));
          Q = MultiplyUIntModLazy(*Y, W_op, W_precon, mod);
          *X++ = tx + Q;
          *Y++ = tx + twice_mod - Q;
        }
      } else {
        __m512i v_W_operand = _mm512_set1_epi64(W_op);
        __m512i v_W_barrett = _mm512_set1_epi64(W_precon);

        __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
        __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

        for (size_t j = j1; j < j2; j += 8) {
          __m512i v_X = _mm512_loadu_si512(v_X_pt);
          __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

          // tx = *X - (two_times_modulus &
          // static_cast<uint64_t>(-static_cast<int64_t>(*X >=
          // two_times_modulus)));
          __m512i v_subX = avx512_cmpgteq_epu64(v_X, v_twice_mod);
          __m512i v_and = _mm512_and_si512(v_twice_mod, v_subX);
          __m512i v_tx = _mm512_sub_epi64(v_X, v_and);

          // multiply_uint64_hw64(Wprime, *Y, &Q);
          __m512i v_Q = avx512_multiply_uint64_hi(v_W_barrett, v_Y);

          // Q = *Y * W - Q * modulus;
          __m512i tmp1 = _mm512_mullo_epi64(v_Y, v_W_operand);
          __m512i tmp2 = _mm512_mullo_epi64(v_Q, v_modulus);
          v_Q = _mm512_sub_epi64(tmp1, tmp2);

          // *X++ = tx + Q;
          v_X = _mm512_add_epi64(v_tx, v_Q);

          // *Y++ = tx + (two_times_modulus - Q);
          __m512i sub = _mm512_sub_epi64(v_twice_mod, v_Q);
          v_Y = _mm512_add_epi64(v_tx, sub);

          _mm512_storeu_si512(v_X_pt, v_X);
          _mm512_storeu_si512(v_Y_pt, v_Y);

          ++v_X_pt;
          ++v_Y_pt;
        }
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }

  // TODO(fboemer) AVX512
  for (size_t i = 0; i < n; ++i) {
    if (input[i] >= twice_mod) {
      input[i] -= twice_mod;
    }
    if (input[i] >= mod) {
      input[i] -= mod;
    }
  }
}

}  // namespace ntt
}  // namespace intel
