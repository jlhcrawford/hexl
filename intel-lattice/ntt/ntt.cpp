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

#include "ntt.hpp"

#include <immintrin.h>

#include <iostream>
#include <utility>

#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"

namespace intel {
namespace lattice {

NTT::NTT(IntType degree, IntType p, IntType root_of_unity)
    : m_degree(degree), m_p(p), m_w(root_of_unity) {
  LATTICE_CHECK(CheckArguments(degree, p), "");
  LATTICE_CHECK(
      IsPrimitiveRoot(m_w, 2 * degree, p),
      m_w << " is not a primitive 2*" << degree << "'th root of unity");

#ifdef LATTICE_HAS_AVX512IFMA
  if (m_p < s_max_ifma_modulus) {
    IVLOG(3, "Setting m_bit_shift to " << s_ifma_shift_bits);
    m_bit_shift = s_ifma_shift_bits;
  }
#endif

  m_degree_bits = Log2(m_degree);

  m_winv = InverseUIntMod(m_w, m_p);
  ComputeRootOfUnityPowers();
}

void NTT::ComputeRootOfUnityPowers() {
  {
    std::pair<IntType, IntType> key = std::make_pair(m_p, m_w);
    std::pair<IntType, IntType> key_inv = std::make_pair(m_p, m_winv);

    auto it = NTT::GetStaticRootOfUnityPowers().find(key);
    if (it != NTT::GetStaticRootOfUnityPowers().end()) {
      return;
    }

    auto it_inv = NTT::GetStaticInvRootOfUnityPowers().find(key_inv);
    if (it_inv != NTT::GetStaticInvRootOfUnityPowers().end()) {
      return;
    }

    std::vector<IntType> root_of_unity_powers(m_degree);
    std::vector<IntType> precon_root_of_unity_powers(m_degree);
    std::vector<IntType> inv_root_of_unity_powers(m_degree);
    std::vector<IntType> precon_inv_root_of_unity_powers(m_degree);

    MultiplyFactor first(1, m_bit_shift, m_p);
    root_of_unity_powers[0] = first.Operand();
    precon_root_of_unity_powers[0] = first.BarrettFactor();

    MultiplyFactor first_inv(InverseUIntMod(first.Operand(), m_p), m_bit_shift,
                             m_p);
    inv_root_of_unity_powers[0] = first_inv.Operand();
    precon_inv_root_of_unity_powers[0] = first_inv.BarrettFactor();
    int idx = 0;
    int prev_idx = idx;
    for (size_t i = 1; i < m_degree; i++) {
      idx = ReverseBitsUInt(i, m_degree_bits);
      MultiplyFactor mf(
          MultiplyUIntMod(root_of_unity_powers[prev_idx], m_w, m_p),
          m_bit_shift, m_p);
      root_of_unity_powers[idx] = mf.Operand();
      precon_root_of_unity_powers[idx] = mf.BarrettFactor();

      MultiplyFactor mf_inv(InverseUIntMod(mf.Operand(), m_p), m_bit_shift,
                            m_p);
      inv_root_of_unity_powers[idx] = mf_inv.Operand();
      precon_inv_root_of_unity_powers[idx] = mf_inv.BarrettFactor();

      prev_idx = idx;
    }

    // Reordering inv_root_of_powers
    std::vector<IntType> temp(m_degree);
    temp[0] = inv_root_of_unity_powers[0];
    idx = 1;
    for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
      for (size_t i = 0; i < m; i++) {
        temp[idx] = inv_root_of_unity_powers[m + i];
        idx++;
      }
    }
    inv_root_of_unity_powers = temp;

    // Reordering precon_inv_root_of_unity_powers
    temp[0] = precon_inv_root_of_unity_powers[0];
    idx = 1;
    for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
      for (size_t i = 0; i < m; i++) {
        temp[idx] = precon_inv_root_of_unity_powers[m + i];
        idx++;
      }
    }
    precon_inv_root_of_unity_powers = std::move(temp);

    NTT::GetStaticRootOfUnityPowers()[key] = std::move(root_of_unity_powers);
    NTT::GetStaticPreconRootOfUnityPowers()[key] =
        std::move(precon_root_of_unity_powers);

    NTT::GetStaticInvRootOfUnityPowers()[key_inv] =
        std::move(inv_root_of_unity_powers);
    NTT::GetStaticPreconInvRootOfUnityPowers()[key_inv] =
        std::move(precon_inv_root_of_unity_powers);
  }
}

void NTT::ForwardTransformToBitReverse64(
    IntType n, IntType mod, const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");

  uint64_t twice_mod = mod << 1;
  size_t t = (n >> 1);

  for (size_t m = 1; m < n; m <<= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = root_of_unity_powers[m + i];
      const uint64_t W_precon = precon_root_of_unity_powers[m + i];

      uint64_t* X = elements + j1;
      uint64_t* Y = X + t;

      uint64_t tx;
      uint64_t Q;
#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
      for (size_t j = j1; j < j2; j++) {
        // The Harvey butterfly: assume X, Y in [0, 4p), and return X', Y' in
        // [0, 4p).
        // See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
        // X', Y' = X + WY, X - WY (mod p).
        LATTICE_CHECK(*X <= (mod * 4), "input X " << (tx + Q) << " too large");
        LATTICE_CHECK(*Y <= (mod * 4), "input Y " << (tx + Q) << " too large");

        tx = *X - (twice_mod & static_cast<uint64_t>(
                                   -static_cast<int64_t>(*X >= twice_mod)));
        Q = MultiplyUIntModLazy<64>(*Y, W_op, W_precon, mod);

        *X++ = tx + Q;
        *Y++ = tx + twice_mod - Q;

        LATTICE_CHECK(tx + Q <= (mod * 4),
                      "ouput X " << (tx + Q) << " too large");
        LATTICE_CHECK(tx + twice_mod - Q <= (mod * 4),
                      "output Y " << (tx + twice_mod - Q) << " too large");
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }
  for (size_t i = 0; i < n; ++i) {
    if (elements[i] >= twice_mod) {
      elements[i] -= twice_mod;
    }
    if (elements[i] >= mod) {
      elements[i] -= mod;
    }
    LATTICE_CHECK(elements[i] < mod, "Incorrect modulus reduction in NTT "
                                         << elements[i] << " >= " << mod);
  }
}

void NTT::ReferenceForwardTransformToBitReverse(
    IntType n, IntType mod, const IntType* root_of_unity_powers,
    IntType* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");

  size_t t = (n >> 1);
  for (size_t m = 1; m < n; m <<= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = root_of_unity_powers[m + i];

      uint64_t* X = elements + j1;
      uint64_t* Y = X + t;
      for (size_t j = j1; j < j2; j++) {
        uint64_t tx = *X;
        // X', Y' = X + WY, X - WY (mod p).
        uint64_t W_x_Y = MultiplyUIntMod(*Y, W_op, mod);
        *X++ = AddUIntMod(tx, W_x_Y, mod);
        *Y++ = SubUIntMod(tx, W_x_Y, mod);
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }
}

void NTT::ForwardTransformToBitReverse(
    IntType n, IntType mod, const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements,
    IntType bit_shift) {
  LATTICE_CHECK(
      bit_shift == s_ifma_shift_bits || bit_shift == s_default_shift_bits,
      "Bit shift " << bit_shift << " should be either " << s_ifma_shift_bits
                   << " or " << s_default_shift_bits);

#ifdef LATTICE_HAS_AVX512IFMA
  if (bit_shift == s_ifma_shift_bits && (mod < s_max_ifma_modulus)) {
    IVLOG(3, "Calling 52-bit AVX512-IFMA NTT");
    NTT::ForwardTransformToBitReverseAVX512<s_ifma_shift_bits>(
        n, mod, root_of_unity_powers, precon_root_of_unity_powers, elements);
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512F
  IVLOG(3, "Calling 64-bit AVX512 NTT");
  NTT::ForwardTransformToBitReverseAVX512<s_default_shift_bits>(
      n, mod, root_of_unity_powers, precon_root_of_unity_powers, elements);
  return;
#endif

  IVLOG(3, "Calling 64-bit default NTT");
  NTT::ForwardTransformToBitReverse64(n, mod, root_of_unity_powers,
                                      precon_root_of_unity_powers, elements);
}

void NTT::InverseTransformToBitReverse64(
    const IntType n, const IntType mod, const IntType* inv_root_of_unity_powers,
    const IntType* precon_inv_root_of_unity_powers, IntType* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");

  uint64_t twice_mod = mod << 1;
  size_t t = 1;
  size_t root_index = 1;

  for (size_t m = (n >> 1); m > 1; m >>= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++, root_index++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = inv_root_of_unity_powers[root_index];
      const uint64_t W_op_precon = precon_inv_root_of_unity_powers[root_index];

      IVLOG(4, "m = " << i << ", i = " << i);
      IVLOG(4, "j1 = " << j1 << ", j2 = " << j2);

      uint64_t* X = elements + j1;
      uint64_t* Y = X + t;

      uint64_t tx;
      uint64_t ty;

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
      for (size_t j = j1; j < j2; j++) {
        IVLOG(4, "Loaded *X " << *X);
        IVLOG(4, "Loaded *Y " << *Y);
        // The Harvey butterfly: assume X, Y in [0, 4p), and return X', Y' in
        // [0, 4p).
        // X', Y' = X + Y (mod p), W(X - Y) (mod p).
        tx = *X + *Y;
        ty = *X + twice_mod - *Y;

        *X++ = tx - (twice_mod & static_cast<uint64_t>(
                                     (-static_cast<int64_t>(tx >= twice_mod))));
        *Y++ = MultiplyUIntModLazy<64>(ty, W_op, W_op_precon, mod);
      }
      j1 += (t << 1);
    }
    t <<= 1;
  }

  const uint64_t W_op = inv_root_of_unity_powers[root_index];
  const uint64_t inv_n = InverseUIntMod(n, mod);
  const uint64_t inv_n_w = MultiplyUIntMod(inv_n, W_op, mod);

  uint64_t* X = elements;
  uint64_t* Y = X + (n >> 1);
  uint64_t tx;
  uint64_t ty;

  for (size_t j = (n >> 1); j < n; j++) {
    tx = *X + *Y;
    tx -= twice_mod &
          static_cast<uint64_t>(-static_cast<int64_t>(tx >= twice_mod));
    ty = *X + twice_mod - *Y;
    *X++ = MultiplyUIntModLazy<64>(tx, inv_n, mod);
    *Y++ = MultiplyUIntModLazy<64>(ty, inv_n_w, mod);
  }

  // Reduce from [0, 4p) to [0,p)
  for (size_t i = 0; i < n; ++i) {
    if (elements[i] >= twice_mod) {
      elements[i] -= twice_mod;
    }
    if (elements[i] >= mod) {
      elements[i] -= mod;
    }
    LATTICE_CHECK(elements[i] < mod, "Incorrect modulus reduction in InvNTT"
                                         << elements[i] << " >= " << mod);
  }
}

void NTT::InverseTransformToBitReverse(
    const IntType n, const IntType mod, const IntType* inv_root_of_unity_powers,
    const IntType* precon_inv_root_of_unity_powers, IntType* elements,
    IntType bit_shift) {
  LATTICE_CHECK(
      bit_shift == s_ifma_shift_bits || bit_shift == s_default_shift_bits,
      "Bit shift " << bit_shift << " should be either " << s_ifma_shift_bits
                   << " or " << s_default_shift_bits);

#ifdef LATTICE_HAS_AVX512IFMA
  if (bit_shift == s_ifma_shift_bits && (mod < s_max_ifma_modulus)) {
    IVLOG(3, "Calling 52-bit AVX512-IFMA InvNTT");
    NTT::InverseTransformToBitReverseAVX512<s_ifma_shift_bits>(
        n, mod, inv_root_of_unity_powers, precon_inv_root_of_unity_powers,
        elements);
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512F
  IVLOG(3, "Calling 64-bit AVX512 InvNTT");
  NTT::InverseTransformToBitReverseAVX512<s_default_shift_bits>(
      n, mod, inv_root_of_unity_powers, precon_inv_root_of_unity_powers,
      elements);
  return;
#endif

  IVLOG(3, "Calling 64-bit default InvNTT");
  NTT::InverseTransformToBitReverse64(n, mod, inv_root_of_unity_powers,
                                      precon_inv_root_of_unity_powers,
                                      elements);
}

}  // namespace lattice
}  // namespace intel
