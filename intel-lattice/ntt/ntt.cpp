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
        tx = *X - (twice_mod & static_cast<uint64_t>(
                                   -static_cast<int64_t>(*X >= twice_mod)));
        Q = MultiplyUIntModLazy<64>(*Y, W_op, W_precon, mod);

        *X++ = tx + Q;
        *Y++ = tx + twice_mod - Q;
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
    LATTICE_CHECK(elements[i] < mod, "Incorrect modulus reduction "
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
  // TODO(fboemer): Check 50-bit limit more
  // carefully
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
    const IntType* scaled_inv_root_of_unity_powers, IntType* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");

  uint64_t twice_mod = mod << 1;
  size_t t = 1;
  size_t root_index = 1;

  for (size_t m = (n >> 1); m > 1; m >>= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++, root_index++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = inv_root_of_unity_powers[root_index];
      const uint64_t W_op_precon = scaled_inv_root_of_unity_powers[root_index];

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
        uint64_t write_x =
            tx - (twice_mod & static_cast<uint64_t>(
                                  (-static_cast<int64_t>(tx >= twice_mod))));
        IVLOG(4, "tx " << tx);
        uint64_t write_y = MultiplyUIntModLazy<64>(ty, W_op, W_op_precon, mod);

        *X++ = write_x;
        *Y++ = MultiplyUIntModLazy<64>(ty, W_op, W_op_precon, mod);

        IVLOG(4, "Wrote X " << write_x);
        IVLOG(4, "Wrote Y " << write_y << "\n");
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
    LATTICE_CHECK(elements[i] < mod, "Incorrect modulus reduction "
                                         << elements[i] << " >= " << mod);
  }
}

void NTT::InverseTransformToBitReverse(
    const IntType n, const IntType mod, const IntType* inv_root_of_unity_powers,
    const IntType* inv_scaled_root_of_unity_powers, IntType* elements) {
  // TODO(skim): Enable IFMA after investigation where the scaled inverse root
  // of unity is within 2**52 range - add with (bool use_ifma_if_possible)

  /*
  #ifdef LATTICE_HAS_AVX512IFMA
    // TODO(fboemer): Check 50-bit limit more carefully
    constexpr IntType ifma_mod_bound = (1UL << 50);
    if (use_ifma_if_possible && (mod < ifma_mod_bound)) {
      IVLOG(3, "Calling 52-bit AVX512-IFMA invNTT");

      NTT::InverseTransformToBitReverseAVX512<s_ifma_shift_bits>(
          n, mod, inv_root_of_unity_powers, inv_scaled_root_of_unity_powers,
          elements);
      return;
    }
  #endif
  */

#ifdef LATTICE_HAS_AVX512F
  IVLOG(3, "Calling 64-bit AVX512 invNTT");
  NTT::InverseTransformToBitReverseAVX512<s_default_shift_bits>(
      n, mod, inv_root_of_unity_powers, inv_scaled_root_of_unity_powers,
      elements);
  return;
#endif

  IVLOG(3, "Calling 64-bit default invNTT");
  NTT::InverseTransformToBitReverse64(n, mod, inv_root_of_unity_powers,
                                      inv_scaled_root_of_unity_powers,
                                      elements);
}

}  // namespace lattice
}  // namespace intel
