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
#include "ntt/ntt.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512_util.hpp"

namespace intel {
namespace lattice {

template <int BitShift>
void NTT::ForwardTransformToBitReverseAVX512(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements) {
  uint64_t twice_mod = mod << 1;

  __m512i v_modulus = _mm512_set1_epi64(mod);
  __m512i v_twice_mod = _mm512_set1_epi64(twice_mod);

  IVLOG(5, "root_of_unity_powers " << std::vector<uint64_t>(
               root_of_unity_powers, root_of_unity_powers + degree))
  IVLOG(5,
        "precon_root_of_unity_powers " << std::vector<uint64_t>(
            precon_root_of_unity_powers, precon_root_of_unity_powers + degree));

  IVLOG(5, "elements " << std::vector<uint64_t>(elements, elements + degree));

  size_t n = degree;

  CheckArguments(degree, mod);
  LATTICE_CHECK(
      CheckBounds(precon_root_of_unity_powers, n, MaximumValue(BitShift)),
      "precon_root_of_unity_powers too large");
  LATTICE_CHECK(CheckBounds(elements, n, MaximumValue(BitShift)),
                "elements too large");

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
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
        for (size_t j = j1; j < j2; j++) {
          // The Harvey butterfly: assume X, Y in [0, 4p), and return X', Y' in
          // [0, 4p).
          // See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
          // X', Y' = X + WY, X - WY (mod p).
          tx = *X - (twice_mod & static_cast<uint64_t>(
                                     -static_cast<int64_t>(*X >= twice_mod)));
          Q = MultiplyUIntModLazy<BitShift>(*Y, W_op, W_precon, mod);

          LATTICE_CHECK(tx + Q <= MaximumValue(BitShift),
                        "tx " << tx << " + Q " << Q << " excceds");
          LATTICE_CHECK(tx + twice_mod - Q <= MaximumValue(BitShift),
                        "tx " << tx << " + twice_mod " << twice_mod << " + Q "
                              << Q << " excceds");
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

          // tx = X >= twice_mod ? X - twice_mod : X
          __m512i v_tx = avx512_mod_epu64(v_X, v_twice_mod);

          // multiply_uint64_hw64(Wprime, *Y, &Q);
          __m512i v_Q = avx512_multiply_uint64_hi<BitShift>(v_W_barrett, v_Y);

          // Q = *Y * W - Q * modulus;
          // Use 64-bit multiply low, even when BitShift == s_ifma_shift_bits
          __m512i tmp1 = avx512_multiply_uint64_lo<64>(v_Y, v_W_operand);
          __m512i tmp2 = avx512_multiply_uint64_lo<64>(v_Q, v_modulus);
          v_Q = _mm512_sub_epi64(tmp1, tmp2);

          // *X++ = tx + Q;
          v_X = _mm512_add_epi64(v_tx, v_Q);

          // *Y++ = tx + (two_times_modulus - Q);
          __m512i sub = _mm512_sub_epi64(v_twice_mod, v_Q);
          v_Y = _mm512_add_epi64(v_tx, sub);

          _mm512_storeu_si512(v_X_pt, v_X);
          _mm512_storeu_si512(v_Y_pt, v_Y);

          LATTICE_CHECK(CheckBounds(v_X, MaximumValue(BitShift)), "");
          LATTICE_CHECK(CheckBounds(v_Y, MaximumValue(BitShift)), "");

          ++v_X_pt;
          ++v_Y_pt;
        }
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }

  if (n < 8) {
    for (size_t i = 0; i < n; ++i) {
      if (input[i] >= twice_mod) {
        input[i] -= twice_mod;
      }
      if (input[i] >= mod) {
        input[i] -= mod;
      }
    }
  } else {
    // n power of two at least 8 => n divisible by 8
    LATTICE_CHECK(n % 8 == 0, "degree " << degree << " not a power of 2");
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(elements);
    for (size_t i = 0; i < n; i += 8) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);

      v_X = avx512_mod_epu64(v_X, v_twice_mod);
      v_X = avx512_mod_epu64(v_X, v_modulus);

      _mm512_storeu_si512(v_X_pt, v_X);

      ++v_X_pt;
    }
  }
}

template <int BitShift>
void NTT::InverseTransformToBitReverseAVX512(
    const IntType degree, const IntType mod,
    const IntType* inv_root_of_unity_powers,
    const IntType* inv_scaled_root_of_unity_powers, IntType* elements) {
  uint64_t twice_mod = mod << 1;

  __m512i v_modulus = _mm512_set1_epi64(mod);
  __m512i v_twice_mod = _mm512_set1_epi64(twice_mod);

  IVLOG(5, "inv_root_of_unity_powers " << std::vector<uint64_t>(
               inv_root_of_unity_powers, inv_root_of_unity_powers + degree))
  IVLOG(5, "inv_scaled_root_of_unity_powers " << std::vector<uint64_t>(
               inv_scaled_root_of_unity_powers,
               inv_scaled_root_of_unity_powers + degree));

  IVLOG(5, "elements " << std::vector<uint64_t>(elements, elements + degree));

  size_t n = degree;

  CheckArguments(degree, mod);
  LATTICE_CHECK(
      CheckBounds(inv_scaled_root_of_unity_powers, n, MaximumValue(BitShift)),
      "");
  LATTICE_CHECK(CheckBounds(elements, n, MaximumValue(BitShift)), "");

  size_t t = 1;
  size_t root_index = 1;
  uint64_t* input = elements;

  for (size_t m = (n >> 1); m > 1; m >>= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++, root_index++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = inv_root_of_unity_powers[root_index];
      const uint64_t W_precon = inv_scaled_root_of_unity_powers[root_index];

      uint64_t* X = input + j1;
      uint64_t* Y = X + t;

      IVLOG(3, "---m,i : " << m << "," << i);
      IVLOG(3, "*X        : " << *X);
      IVLOG(3, "*Y        : " << *Y);
      IVLOG(3, "W_operand : " << W_op << " " << (W_op > 0x000fffffffffffff));
      IVLOG(3, "W_precon  : " << W_precon << " "
                              << (W_precon > 0x000fffffffffffff));

      uint64_t tx;
      uint64_t ty;

      if (j2 - j1 < 8) {
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
        for (size_t j = j1; j < j2; j++) {
          // The Harvey butterfly: assume X, Y in [0, 4p), and return X', Y' in
          // [0, 4p).
          // X', Y' = X + Y (mod p), W(X - Y) (mod p).
          tx = *X + *Y;
          ty = *X + twice_mod - *Y;
          *X++ =
              tx - (twice_mod & static_cast<uint64_t>(
                                    (-static_cast<int64_t>(tx >= twice_mod))));
          *Y++ = MultiplyUIntModLazy<BitShift>(ty, W_op, mod);
        }
      } else {
        __m512i v_W_operand = _mm512_set1_epi64(W_op);
        __m512i v_W_barrett = _mm512_set1_epi64(W_precon);

        __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
        __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

        for (size_t j = j1; j < j2; j += 8) {
          IVLOG(3, "----j " << j);

          __m512i v_X = _mm512_loadu_si512(v_X_pt);
          __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

          IVLOG(3, "v_X : " << ExtractValues(v_X));
          IVLOG(3, "v_Y : " << ExtractValues(v_Y));

          // tx = *X + *Y
          __m512i v_tx = _mm512_add_epi64(v_X, v_Y);

          // ty = *X + twice_mod - *Y
          __m512i tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
          __m512i v_ty = _mm512_sub_epi64(tmp_ty, v_Y);

          IVLOG(3, "v_tx : " << ExtractValues(v_tx));
          IVLOG(3, "v_ty : " << ExtractValues(v_ty));

          // *X++ = tx >= twice_mod ? tx - twice_mod : tx
          v_X = avx512_mod_epu64(v_tx, v_twice_mod);
          IVLOG(3, "v_X : " << ExtractValues(v_X));

          // *Y++ = MultiplyUIntModLazy<64>(ty, W_operand, mod)
          // multiply_uint64_hw64(W_barrett, *Y, &Q);
          __m512i v_Q = avx512_multiply_uint64_hi<BitShift>(v_W_barrett, v_Y);
          IVLOG(3, "v_Q : " << ExtractValues(v_Q));

          // *Y++ = ty * W_op - Q * modulus;
          __m512i tmp_y1 =
              avx512_multiply_uint64_lo<BitShift>(v_ty, v_W_operand);
          IVLOG(3, "v_tmp_y1 : " << ExtractValues(tmp_y1));
          __m512i tmp_y2 = avx512_multiply_uint64_lo<BitShift>(v_Q, v_modulus);
          IVLOG(3, "v_tmp_y2 : " << ExtractValues(tmp_y2));
          v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);
          IVLOG(3, "v_Y : " << ExtractValues(v_Y));

          _mm512_storeu_si512(v_X_pt, v_X);
          _mm512_storeu_si512(v_Y_pt, v_Y);

          LATTICE_CHECK(CheckBounds(v_X, MaximumValue(BitShift)), "");
          LATTICE_CHECK(CheckBounds(v_Y, MaximumValue(BitShift)), "");

          ++v_X_pt;
          ++v_Y_pt;
        }
      }
      j1 += (t << 1);
    }
    t <<= 1;
  }

  const uint64_t W_op = inv_root_of_unity_powers[root_index];
  const uint64_t inv_n = InverseUIntMod(n, mod);
  const uint64_t inv_n_prime = DivideUInt128UInt64Lo(0, inv_n, mod);

  const uint64_t inv_n_w = MultiplyUIntMod(inv_n, W_op, mod);
  const uint64_t inv_n_w_prime = DivideUInt128UInt64Lo(0, inv_n_w, mod);

  uint64_t* X = input;
  uint64_t* Y = X + (n >> 1);

  if (n < 8) {
    for (size_t j = (n >> 1); j < n; j++) {
      uint64_t tx;
      uint64_t ty;

      tx = *X + *Y;
      tx -= twice_mod &
            static_cast<uint64_t>(-static_cast<int64_t>(tx >= twice_mod));
      ty = *X + twice_mod - *Y;
      *X++ = MultiplyUIntModLazy<BitShift>(tx, inv_n, mod);
      *Y++ = MultiplyUIntModLazy<BitShift>(ty, inv_n_w, mod);
    }
  } else {
    __m512i v_inv_n = _mm512_set1_epi64(inv_n);
    __m512i v_inv_n_prime = _mm512_set1_epi64(inv_n_prime);
    __m512i v_inv_n_w = _mm512_set1_epi64(inv_n_w);
    __m512i v_inv_n_w_prime = _mm512_set1_epi64(inv_n_w_prime);

    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
    __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
    for (size_t j = (n >> 1); j < n; j += 8) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

      // tx = tx >= twice_mod ? tx - twice_mod : tx
      __m512i tmp_tx = _mm512_add_epi64(v_X, v_Y);
      __m512i v_tx = avx512_mod_epu64(tmp_tx, v_twice_mod);

      // ty = *X + twice_mod - *Y
      __m512i v_tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
      __m512i v_ty = _mm512_sub_epi64(v_tmp_ty, v_Y);

      // multiply_uint64_hw64(inv_Nprime, tx, &Q);
      __m512i v_Q1 = avx512_multiply_uint64_hi<BitShift>(v_inv_n_prime, v_tx);
      // *X++ = inv_N * tx - Q * modulus;
      __m512i tmp_x1 = avx512_multiply_uint64_lo<BitShift>(v_inv_n, v_tx);
      __m512i tmp_x2 = avx512_multiply_uint64_lo<BitShift>(v_Q1, v_modulus);
      v_X = _mm512_sub_epi64(tmp_x1, tmp_x2);

      // multiply_uint64_hw64(inv_N_Wprime, ty, &Q);
      __m512i v_Q2 = avx512_multiply_uint64_hi<BitShift>(v_inv_n_w_prime, v_ty);
      // *Y++ = inv_N_W * ty - Q * modulus;
      __m512i tmp_y1 = avx512_multiply_uint64_lo<BitShift>(v_inv_n_w, v_ty);
      __m512i tmp_y2 = avx512_multiply_uint64_lo<BitShift>(v_Q2, v_modulus);
      v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);

      _mm512_storeu_si512(v_X_pt, v_X);
      _mm512_storeu_si512(v_Y_pt, v_Y);

      LATTICE_CHECK(CheckBounds(v_X, MaximumValue(BitShift)), "");
      LATTICE_CHECK(CheckBounds(v_Y, MaximumValue(BitShift)), "");

      ++v_X_pt;
      ++v_Y_pt;
    }
  }

  // Reduce from [0,4p) to [0,p)
  if (n < 8) {
    for (size_t i = 0; i < n; ++i) {
      if (input[i] >= twice_mod) {
        input[i] -= twice_mod;
      }
      if (input[i] >= mod) {
        input[i] -= mod;
      }
    }
  } else {
    // n power of two at least 8 => n divisible by 8
    LATTICE_CHECK(n % 8 == 0, "degree " << degree << " not a power of 2");
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(elements);
    for (size_t i = 0; i < n; i += 8) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);

      v_X = avx512_mod_epu64(v_X, v_twice_mod);
      v_X = avx512_mod_epu64(v_X, v_modulus);

      _mm512_storeu_si512(v_X_pt, v_X);

      ++v_X_pt;
    }
  }
}

}  // namespace lattice
}  // namespace intel
