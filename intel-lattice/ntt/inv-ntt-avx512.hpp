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
#include "ntt/ntt-avx512-util.hpp"
#include "ntt/ntt-internal.hpp"
#include "ntt/ntt.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace lattice {

template <int BitShift>
void InvT1(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  const __m512i* v_W_op_pt = reinterpret_cast<const __m512i*>(W_op);
  const __m512i* v_W_precon_pt = reinterpret_cast<const __m512i*>(W_precon);
  size_t j1 = 0;

  // 8 | m guaranteed by n >= 16
  for (size_t i = m / 8; i > 0; --i) {
    uint64_t* X = elements + j1;
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadInterleavedT1(X, &v_X, &v_Y);

    LATTICE_CHECK_BOUNDS(ExtractValues(v_X).data(), 8,
                         ExtractValues(v_twice_mod)[0]);
    LATTICE_CHECK_BOUNDS(ExtractValues(v_Y).data(), 8,
                         ExtractValues(v_twice_mod)[0]);

    __m512i v_W_op = _mm512_loadu_si512(v_W_op_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);

    __m512i tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
    __m512i v_ty = _mm512_sub_epi64(tmp_ty, v_Y);

    // No need for modulus reduction, since inputs are in [0,p)
    // v_X = _mm512_il_small_add_mod_epi64(v_X, v_Y, v_twice_mod);
    v_X = _mm512_add_epi64(v_X, v_Y);

    __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_ty);
    __m512i tmp_y1 = _mm512_mullo_epi64(v_ty, v_W_op);
    __m512i tmp_y2 = _mm512_mullo_epi64(v_Q, v_modulus);
    v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);

    WriteInterleavedT1(v_X, v_Y, v_X_pt);

    j1 += 16;
  }
}

template <int BitShift>
void InvT2(uint64_t* X, __m512i v_modulus, __m512i v_twice_mod, uint64_t m,
           const uint64_t* W_op, const uint64_t* W_precon) {
  // 4 | m guaranteed by n >= 16
  for (size_t i = m / 4; i > 0; --i) {
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadInterleavedT2(X, &v_X, &v_Y);

    __m512i v_W_op = LoadWOpT2(static_cast<const void*>(W_op));
    __m512i v_W_precon = LoadWOpT2(static_cast<const void*>(W_precon));

    __m512i tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
    __m512i v_ty = _mm512_sub_epi64(tmp_ty, v_Y);

    v_X = _mm512_il_small_add_mod_epi64(v_X, v_Y, v_twice_mod);

    __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_ty);
    __m512i tmp_y1 = _mm512_mullo_epi64(v_ty, v_W_op);
    __m512i tmp_y2 = _mm512_mullo_epi64(v_Q, v_modulus);
    v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);

    WriteInterleavedT2(v_X, v_Y, v_X_pt);
    X += 16;

    W_op += 4;
    W_precon += 4;
  }
}

template <int BitShift>
void InvT4(uint64_t* elements, __m512i v_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  uint64_t* X = elements;

  // 2 | m guaranteed by n >= 16
  for (size_t i = m / 2; i > 0; --i) {
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadInterleavedT4(X, &v_X, &v_Y);

    __m512i v_W_op = LoadWOpT4(static_cast<const void*>(W_op));
    __m512i v_W_precon = LoadWOpT4(static_cast<const void*>(W_precon));

    __m512i tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
    __m512i v_ty = _mm512_sub_epi64(tmp_ty, v_Y);

    v_X = _mm512_il_small_add_mod_epi64(v_X, v_Y, v_twice_mod);

    __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_ty);
    __m512i tmp_y1 = _mm512_mullo_epi64(v_ty, v_W_op);
    __m512i tmp_y2 = _mm512_mullo_epi64(v_Q, v_modulus);
    v_Y = _mm512_sub_epi64(tmp_y1, tmp_y2);

    WriteInterleavedT4(v_X, v_Y, v_X_pt);
    X += 16;

    W_op += 2;
    W_precon += 2;
  }
}

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

      // ty = *X + twice_mod - *Y
      __m512i tmp_ty = _mm512_add_epi64(v_X, v_twice_mod);
      __m512i v_ty = _mm512_sub_epi64(tmp_ty, v_Y);

      // tx = *X + *Y mod modulus
      v_X = _mm512_il_small_add_mod_epi64(v_X, v_Y, v_twice_mod);

      // *Y++ = MultiplyUIntModLazy<64>(ty, W_operand, mod)
      __m512i v_Q = _mm512_il_mulhi_epi<BitShift>(v_W_precon, v_ty);
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
void InverseTransformFromBitReverseAVX512(
    const uint64_t n, const uint64_t mod,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");
  LATTICE_CHECK_BOUNDS(precon_inv_root_of_unity_powers, n,
                       MaximumValue(BitShift));
  LATTICE_CHECK_BOUNDS(elements, n, MaximumValue(BitShift),
                       "elements too large");
  LATTICE_CHECK_BOUNDS(elements, n, mod,
                       "elements larger than modulus " << mod);

  uint64_t twice_mod = mod << 1;
  __m512i v_modulus = _mm512_set1_epi64(mod);
  __m512i v_twice_mod = _mm512_set1_epi64(twice_mod);

  size_t t = 1;
  size_t root_index = 1;
  size_t m = (n >> 1);

  // Extract t=1, t=2, t=4 loops separately
  {
    // t = 1
    const uint64_t* W_op = &inv_root_of_unity_powers[root_index];
    const uint64_t* W_precon = &precon_inv_root_of_unity_powers[root_index];
    InvT1<BitShift>(elements, v_modulus, v_twice_mod, m, W_op, W_precon);
    t <<= 1;
    root_index += m;
    m >>= 1;

    // t = 2
    W_op = &inv_root_of_unity_powers[root_index];
    W_precon = &precon_inv_root_of_unity_powers[root_index];
    InvT2<BitShift>(elements, v_modulus, v_twice_mod, m, W_op, W_precon);
    t <<= 1;
    root_index += m;
    m >>= 1;

    // t = 4
    W_op = &inv_root_of_unity_powers[root_index];
    W_precon = &precon_inv_root_of_unity_powers[root_index];
    InvT4<BitShift>(elements, v_modulus, v_twice_mod, m, W_op, W_precon);
    t <<= 1;
    root_index += m;
    m >>= 1;
  }

  // t >= 8
  for (; m > 1; m >>= 1) {
    const uint64_t* W_op = &inv_root_of_unity_powers[root_index];
    const uint64_t* W_precon = &precon_inv_root_of_unity_powers[root_index];
    InvT8<BitShift>(elements, v_modulus, v_twice_mod, t, m, W_op, W_precon);
    t <<= 1;
    root_index += m;
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

    __m512i v_tx = _mm512_il_small_add_mod_epi64(v_X, v_Y, v_twice_mod);

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

    // Modulus reduction from [0,2p), to [0,p)
    v_X = _mm512_il_small_mod_epu64(v_X, v_modulus);
    v_Y = _mm512_il_small_mod_epu64(v_Y, v_modulus);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_Y_pt++, v_Y);
  }

  IVLOG(5, "AVX512 returning elements "
               << std::vector<uint64_t>(elements, elements + n));
}

}  // namespace lattice
}  // namespace intel
