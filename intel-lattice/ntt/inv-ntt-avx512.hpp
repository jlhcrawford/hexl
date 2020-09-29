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
void InvT8(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
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

      // tx = *X + *Y
      __m512i v_tx = _mm512_add_epi64(v_X, v_Y);

      // ty = *X + twice_mod - *Y
      __m512i tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
      __m512i v_ty = _mm512_sub_epi64(tmp_ty, v_Y);

      // *X++ = tx >= twice_mod ? tx - twice_mod : tx
      v_X = _mm512_il_small_mod_epi64(v_tx, v_twice_mod);

      // *Y++ = MultiplyUIntModLazy<64>(ty, W_operand, mod)
      // multiply_uint64_hw64(W_precon, *Y, &Q);
      __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_ty);

      // *Y++ = ty * W_op - Q * modulus;
      // Use 64-bit multiply low, even when BitShift == s_ifma_shift_bits
      __m512i tmp_y1 = _mm512_mullo_epi64(v_ty, v_W_op);
      __m512i tmp_y2 = _mm512_mullo_epi64(v_Q, v_modulus);
      v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);

      _mm512_storeu_si512(v_X_pt++, v_X);
      _mm512_storeu_si512(v_Y_pt++, v_Y);
    }
    j1 += (t << 1);
  }
}

template <int BitShift>
void InvT1(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
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

    __m512i v_tx = _mm512_add_epi64(v_X, v_Y);
    __m512i tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
    __m512i v_ty = _mm512_sub_epi64(tmp_ty, v_Y);
    v_X = _mm512_il_small_mod_epi64(v_tx, v_twice_mod);
    __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_ty);
    __m512i tmp_y1 = _mm512_mullo_epi64(v_ty, v_W_op);
    __m512i tmp_y2 = _mm512_mullo_epi64(v_Q, v_modulus);
    v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);

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
void InverseTransformFromBitReverseAVX512(
    const uint64_t n, const uint64_t mod,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");
  LATTICE_CHECK_BOUNDS(precon_inv_root_of_unity_powers, n,
                       MaximumValue(BitShift));
  LATTICE_CHECK_BOUNDS(elements, n, MaximumValue(BitShift));

  uint64_t twice_mod = mod << 1;

  __m512i v_modulus = _mm512_set1_epi64(mod);
  __m512i v_twice_mod = _mm512_set1_epi64(twice_mod);
  __m256i v256_modulus = _mm256_set1_epi64x(mod);
  __m256i v256_twice_mod = _mm256_set1_epi64x(twice_mod);

  size_t t = 1;
  size_t root_index = 1;

  for (size_t m = (n >> 1); m > 1; m >>= 1) {
    size_t j1 = 0;

    if (t >= 8) {
      const uint64_t* W_op = &inv_root_of_unity_powers[root_index];
      const uint64_t* W_precon = &precon_inv_root_of_unity_powers[root_index];
      InvT8<BitShift>(elements, v_modulus, v_twice_mod, t, m, W_op, W_precon);
      t <<= 1;
      root_index += m;
      continue;
    }

    if (t == 1) {
      const uint64_t* W_op = &inv_root_of_unity_powers[root_index];
      const uint64_t* W_precon = &precon_inv_root_of_unity_powers[root_index];
      InvT1<BitShift>(elements, v_modulus, v_twice_mod, m, W_op, W_precon);
      t <<= 1;
      root_index += m;
      continue;
    }

    for (size_t i = 0; i < m; i++, root_index++) {
      const uint64_t W_op = inv_root_of_unity_powers[root_index];
      const uint64_t W_precon = precon_inv_root_of_unity_powers[root_index];

      uint64_t* X = elements + j1;
      uint64_t* Y = X + t;

      uint64_t tx;
      uint64_t ty;
      if (t == 4) {
        __m256i v_W_op = _mm256_set1_epi64x(W_op);
        __m256i v_W_precon = _mm256_set1_epi64x(W_precon);

        __m256i* v_X_pt = reinterpret_cast<__m256i*>(X);
        __m256i* v_Y_pt = reinterpret_cast<__m256i*>(Y);

        __m256i v_X = _mm256_loadu_si256(v_X_pt);
        __m256i v_Y = _mm256_loadu_si256(v_Y_pt);
        // tx = *X + *Y
        __m256i v_tx = _mm256_add_epi64(v_X, v_Y);

        // ty = *X + twice_mod - *Y
        __m256i tmp_ty = _mm256_add_epi64(v_X, v256_twice_mod);
        __m256i v_ty = _mm256_sub_epi64(tmp_ty, v_Y);

        // *X++ = tx >= twice_mod ? tx - twice_mod : tx
        v_X = _mm256_il_small_mod_epi64(v_tx, v256_twice_mod);

        // *Y++ = MultiplyUIntModLazy<64>(ty, W_operand, mod)
        // multiply_uint64_hw64(W_precon, *Y, &Q);
        __m256i v_Q = _mm256_il_mulhi_epi<BitShift>(v_W_precon, v_ty);

        // *Y++ = ty * W_op - Q * modulus;
        // Use 64-bit multiply low, even when BitShift == s_ifma_shift_bits
        __m256i tmp_y1 = _mm256_mullo_epi64(v_ty, v_W_op);
        __m256i tmp_y2 = _mm256_mullo_epi64(v_Q, v256_modulus);
        v_Y = _mm256_sub_epi64(tmp_y1, tmp_y2);

        _mm256_storeu_si256(v_X_pt, v_X);
        _mm256_storeu_si256(v_Y_pt, v_Y);

      } else if (t == 2) {
        tx = *X + *Y;
        ty = *X + twice_mod - *Y;
        *X++ = tx - (twice_mod & static_cast<uint64_t>(
                                     (-static_cast<int64_t>(tx >= twice_mod))));
        *Y++ = MultiplyUIntModLazy<BitShift>(ty, W_op, W_precon, mod);

        tx = *X + *Y;
        ty = *X + twice_mod - *Y;
        *X = tx - (twice_mod & static_cast<uint64_t>(
                                   (-static_cast<int64_t>(tx >= twice_mod))));
        *Y = MultiplyUIntModLazy<BitShift>(ty, W_op, W_precon, mod);
      }
      j1 += (t << 1);
    }
    t <<= 1;
  }

  IVLOG(4, "AVX512 intermediate elements "
               << std::vector<uint64_t>(elements, elements + n));

  const uint64_t W_op = inv_root_of_unity_powers[root_index];
  MultiplyFactor mf_inv_n(InverseUIntMod(n, mod), BitShift, mod);
  const uint64_t inv_n = mf_inv_n.Operand();
  const uint64_t inv_n_prime = mf_inv_n.BarrettFactor();

  MultiplyFactor mf_inv_n_w(MultiplyUIntMod(inv_n, W_op, mod), BitShift, mod);
  const uint64_t inv_n_w = mf_inv_n_w.Operand();
  const uint64_t inv_n_w_prime = mf_inv_n_w.BarrettFactor();

  IVLOG(4, "inv_n_w " << inv_n_w);

  uint64_t* X = elements;
  uint64_t* Y = X + (n >> 1);

  __m512i v_inv_n = _mm512_set1_epi64(inv_n);
  __m512i v_inv_n_prime = _mm512_set1_epi64(inv_n_prime);
  __m512i v_inv_n_w = _mm512_set1_epi64(inv_n_w);
  __m512i v_inv_n_w_prime = _mm512_set1_epi64(inv_n_w_prime);

  __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
  __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t j = n / 16; j > 0; --j) {
    __m512i v_X = _mm512_loadu_si512(v_X_pt);
    __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

    // tx = tx >= twice_mod ? tx - twice_mod : tx
    __m512i tmp_tx = _mm512_add_epi64(v_X, v_Y);
    __m512i v_tx = _mm512_il_small_mod_epi64(tmp_tx, v_twice_mod);

    // ty = *X + twice_mod - *Y
    __m512i v_tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
    __m512i v_ty = _mm512_sub_epi64(v_tmp_ty, v_Y);

    // multiply_uint64_hw64(inv_Nprime, tx, &Q);
    __m512i v_Q1 = _mm512_il_mulhi_epi<BitShift>(v_inv_n_prime, v_tx);
    // *X++ = inv_N * tx - Q * modulus;
    __m512i tmp_x1 = _mm512_mullo_epi64(v_inv_n, v_tx);
    __m512i tmp_x2 = _mm512_mullo_epi64(v_Q1, v_modulus);
    v_X = _mm512_sub_epi64(tmp_x1, tmp_x2);

    // multiply_uint64_hw64(inv_N_Wprime, ty, &Q);
    __m512i v_Q2 = _mm512_il_mulhi_epi<BitShift>(v_inv_n_w_prime, v_ty);
    // *Y++ = inv_N_W * ty - Q * modulus;
    __m512i tmp_y1 = _mm512_mullo_epi64(v_inv_n_w, v_ty);
    __m512i tmp_y2 = _mm512_mullo_epi64(v_Q2, v_modulus);
    v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);

    v_X = _mm512_il_small_mod_epi64(v_X, v_twice_mod);
    v_X = _mm512_il_small_mod_epi64(v_X, v_modulus);

    v_Y = _mm512_il_small_mod_epi64(v_Y, v_twice_mod);
    v_Y = _mm512_il_small_mod_epi64(v_Y, v_modulus);

    _mm512_storeu_si512(v_X_pt, v_X);
    _mm512_storeu_si512(v_Y_pt, v_Y);

    ++v_X_pt;
    ++v_Y_pt;
  }

  IVLOG(5, "AVX512 returning elements "
               << std::vector<uint64_t>(elements, elements + n));
}

}  // namespace lattice
}  // namespace intel
