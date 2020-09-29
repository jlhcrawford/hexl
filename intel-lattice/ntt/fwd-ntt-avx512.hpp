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

template <int BitShift>
void FwdT1(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  const __m512i* v_W_op_pt = reinterpret_cast<const __m512i*>(W_op);
  const __m512i* v_W_precon_pt = reinterpret_cast<const __m512i*>(W_precon);
  size_t j1 = 0;

  // 8 | m guaranteed by n >= 16
  for (size_t i = m / 8; i > 0; --i) {
    uint64_t* X = elements + j1;
    uint64_t* Y = X + 1;

    __m512i v_X =
        _mm512_set_epi64(X[14], X[12], X[10], X[8], X[6], X[4], X[2], X[0]);

    __m512i v_Y =
        _mm512_set_epi64(Y[14], Y[12], Y[10], Y[8], Y[6], Y[4], Y[2], Y[0]);

    __m512i v_W_op = _mm512_loadu_si512(v_W_op_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);

    __m512i v_tx = _mm512_il_small_mod_epi64(v_X, v_twice_mod);
    __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_Y);
    __m512i tmp1 = _mm512_mullo_epi64(v_Y, v_W_op);
    __m512i tmp2 = _mm512_mullo_epi64(v_Q, v_modulus);
    v_Q = _mm512_sub_epi64(tmp1, tmp2);
    v_X = _mm512_add_epi64(v_tx, v_Q);
    __m512i sub = _mm512_sub_epi64(v_twice_mod, v_Q);
    v_Y = _mm512_add_epi64(v_tx, sub);

    uint64_t* X_out = reinterpret_cast<uint64_t*>(&v_X);
    uint64_t* Y_out = reinterpret_cast<uint64_t*>(&v_Y);
    for (size_t cpy = 0; cpy < 8; ++cpy) {
      *X++ = *X_out++;
      *X++ = *Y_out++;
    }
    j1 += 16;
  }
}

template <int BitShift>
void FwdT2(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  size_t j1 = 0;

  // 4 | m guaranteed by n >= 16
  for (size_t i = m / 4; i > 0; --i) {
    uint64_t* X = elements + j1;
    uint64_t* Y = X + 2;

    __m512i v_X =
        _mm512_set_epi64(X[13], X[12], X[9], X[8], X[5], X[4], X[1], X[0]);
    __m512i v_Y =
        _mm512_set_epi64(Y[13], Y[12], Y[9], Y[8], Y[5], Y[4], Y[1], Y[0]);

    __m512i v_W_op = _mm512_set_epi64(W_op[3], W_op[3], W_op[2], W_op[2],
                                      W_op[1], W_op[1], W_op[0], W_op[0]);
    __m512i v_W_precon =
        _mm512_set_epi64(W_precon[3], W_precon[3], W_precon[2], W_precon[2],
                         W_precon[1], W_precon[1], W_precon[0], W_precon[0]);

    __m512i v_tx = _mm512_il_small_mod_epi64(v_X, v_twice_mod);
    __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_Y);
    __m512i tmp1 = _mm512_mullo_epi64(v_Y, v_W_op);
    __m512i tmp2 = _mm512_mullo_epi64(v_Q, v_modulus);
    v_Q = _mm512_sub_epi64(tmp1, tmp2);
    v_X = _mm512_add_epi64(v_tx, v_Q);
    __m512i sub = _mm512_sub_epi64(v_twice_mod, v_Q);
    v_Y = _mm512_add_epi64(v_tx, sub);

    uint64_t* X_out = reinterpret_cast<uint64_t*>(&v_X);
    uint64_t* Y_out = reinterpret_cast<uint64_t*>(&v_Y);

    *X++ = X_out[0];
    *X++ = X_out[1];
    *X++ = Y_out[0];
    *X++ = Y_out[1];
    *X++ = X_out[2];
    *X++ = X_out[3];
    *X++ = Y_out[2];
    *X++ = Y_out[3];
    *X++ = X_out[4];
    *X++ = X_out[5];
    *X++ = Y_out[4];
    *X++ = Y_out[5];
    *X++ = X_out[6];
    *X++ = X_out[7];
    *X++ = Y_out[6];
    *X++ = Y_out[7];

    W_op += 4;
    W_precon += 4;

    j1 += 16;
  }
}

template <int BitShift>
void FwdT4(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  size_t j1 = 0;

  // 2 | m guaranteed by n >= 16
  for (size_t i = m / 2; i > 0; --i) {
    uint64_t* X = elements + j1;
    uint64_t* Y = X + 4;

    __m512i v_X =
        _mm512_set_epi64(X[11], X[10], X[9], X[8], X[3], X[2], X[1], X[0]);
    __m512i v_Y =
        _mm512_set_epi64(Y[11], Y[10], Y[9], Y[8], Y[3], Y[2], Y[1], Y[0]);

    __m512i v_W_op = _mm512_set_epi64(W_op[1], W_op[1], W_op[1], W_op[1],
                                      W_op[0], W_op[0], W_op[0], W_op[0]);
    __m512i v_W_precon =
        _mm512_set_epi64(W_precon[1], W_precon[1], W_precon[1], W_precon[1],
                         W_precon[0], W_precon[0], W_precon[0], W_precon[0]);

    __m512i v_tx = _mm512_il_small_mod_epi64(v_X, v_twice_mod);
    __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_Y);
    __m512i tmp1 = _mm512_mullo_epi64(v_Y, v_W_op);
    __m512i tmp2 = _mm512_mullo_epi64(v_Q, v_modulus);
    v_Q = _mm512_sub_epi64(tmp1, tmp2);
    v_X = _mm512_add_epi64(v_tx, v_Q);
    __m512i sub = _mm512_sub_epi64(v_twice_mod, v_Q);
    v_Y = _mm512_add_epi64(v_tx, sub);

    uint64_t* X_out = reinterpret_cast<uint64_t*>(&v_X);
    uint64_t* Y_out = reinterpret_cast<uint64_t*>(&v_Y);

    *X++ = X_out[0];
    *X++ = X_out[1];
    *X++ = X_out[2];
    *X++ = X_out[3];
    *X++ = Y_out[0];
    *X++ = Y_out[1];
    *X++ = Y_out[2];
    *X++ = Y_out[3];
    *X++ = X_out[4];
    *X++ = X_out[5];
    *X++ = X_out[6];
    *X++ = X_out[7];
    *X++ = Y_out[4];
    *X++ = Y_out[5];
    *X++ = Y_out[6];
    *X++ = Y_out[7];

    j1 += 16;
    W_op += 2;
    W_precon += 2;
  }
}

template <int BitShift>
void FwdT8(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
           uint64_t t, uint64_t m, const uint64_t* W_op,
           const uint64_t* W_precon) {
  size_t j1 = 0;

  for (size_t i = 0; i < m; i++) {
    uint64_t* X = elements + j1;
    uint64_t* Y = X + t;

    __m512i v_W_op = _mm512_set1_epi64(*W_op++);
    __m512i v_W_precon = _mm512_set1_epi64(*W_precon++);

    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
    __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

    // assume 8 | t
    for (size_t j = t / 8; j > 0; --j) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

      // tx = X >= twice_mod ? X - twice_mod : X
      __m512i v_tx = _mm512_il_small_mod_epi64(v_X, v_twice_mod);

      // multiply_uint64_hw64(Wprime, *Y, &Q);
      __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_Y);

      // Q = *Y * W - Q * modulus;
      // Use 64-bit multiply low, even when BitShift ==
      // s_ifma_shift_bits
      __m512i tmp1 = _mm512_mullo_epi64(v_Y, v_W_op);
      __m512i tmp2 = _mm512_mullo_epi64(v_Q, v_modulus);
      v_Q = _mm512_sub_epi64(tmp1, tmp2);

      // *X++ = tx + Q;
      v_X = _mm512_add_epi64(v_tx, v_Q);

      // *Y++ = tx + (two_times_modulus - Q);
      __m512i sub = _mm512_sub_epi64(v_twice_mod, v_Q);
      v_Y = _mm512_add_epi64(v_tx, sub);

      _mm512_storeu_si512(v_X_pt++, v_X);
      _mm512_storeu_si512(v_Y_pt++, v_Y);
    }
    j1 += (t << 1);
  }
}

template <int BitShift>
void ForwardTransformToBitReverseAVX512(
    const uint64_t n, const uint64_t mod, const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");
  LATTICE_CHECK_BOUNDS(precon_root_of_unity_powers, n, MaximumValue(BitShift),
                       "precon_root_of_unity_powers too large");
  LATTICE_CHECK_BOUNDS(elements, n, MaximumValue(BitShift),
                       "elements too large");
  LATTICE_CHECK(n >= 16,
                "Don't support small transforms. Need n > 16, got n = " << n);

  uint64_t twice_mod = mod << 1;

  __m512i v_modulus = _mm512_set1_epi64(mod);
  __m512i v_twice_mod = _mm512_set1_epi64(twice_mod);

  IVLOG(5, "root_of_unity_powers " << std::vector<uint64_t>(
               root_of_unity_powers, root_of_unity_powers + n))
  IVLOG(5, "precon_root_of_unity_powers " << std::vector<uint64_t>(
               precon_root_of_unity_powers, precon_root_of_unity_powers + n));

  IVLOG(5, "elements " << std::vector<uint64_t>(elements, elements + n));

  size_t t = (n >> 1);
  size_t m = 1;
  for (; m < (n >> 3); m <<= 1) {
    const uint64_t* W_op = &root_of_unity_powers[m];
    const uint64_t* W_precon = &precon_root_of_unity_powers[m];
    FwdT8<BitShift>(elements, v_modulus, v_twice_mod, t, m, W_op, W_precon);
    t >>= 1;
  }

  // Do T=1, T=2, T=4 separately
  {
    const uint64_t* W_op = &root_of_unity_powers[m];
    const uint64_t* W_precon = &precon_root_of_unity_powers[m];

    FwdT4<BitShift>(elements, v_modulus, v_twice_mod, m, W_op, W_precon);
    m <<= 1;
    W_op = &root_of_unity_powers[m];
    W_precon = &precon_root_of_unity_powers[m];
    FwdT2<BitShift>(elements, v_modulus, v_twice_mod, m, W_op, W_precon);
    m <<= 1;
    W_op = &root_of_unity_powers[m];
    W_precon = &precon_root_of_unity_powers[m];
    FwdT1<BitShift>(elements, v_modulus, v_twice_mod, m, W_op, W_precon);
  }

  // n power of two at least 8 => n divisible by 8
  LATTICE_CHECK(n % 8 == 0, "n " << n << " not a power of 2");
  __m512i* v_X_pt = reinterpret_cast<__m512i*>(elements);
  for (size_t i = 0; i < n; i += 8) {
    __m512i v_X = _mm512_loadu_si512(v_X_pt);

    // Reduce from [0, 4p) to [0, p)
    v_X = _mm512_il_small_mod_epi64(v_X, v_twice_mod);
    v_X = _mm512_il_small_mod_epi64(v_X, v_modulus);

    LATTICE_CHECK_BOUNDS(ExtractValues(v_X).data(), 8, mod);

    _mm512_storeu_si512(v_X_pt, v_X);

    ++v_X_pt;
  }
}

}  // namespace lattice
}  // namespace intel
