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

// based on
// https://github.com/microsoft/SEAL/blob/master/native/src/seal/util/ntt.cpp#L200
void NTT::ForwardTransformToBitReverse64(
    IntType degree, IntType mod, const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements) {
  LATTICE_CHECK(CheckArguments(degree, mod), "");

  uint64_t twice_mod = mod << 1;

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
    IntType degree, IntType mod, const IntType* root_of_unity_powers,
    IntType* elements) {
  LATTICE_CHECK(CheckArguments(degree, mod), "");

  size_t t = (degree >> 1);
  for (size_t m = 1; m < degree; m <<= 1) {
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
    IntType degree, IntType mod, const IntType* root_of_unity_powers,
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
        degree, mod, root_of_unity_powers, precon_root_of_unity_powers,
        elements);
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512F
  IVLOG(3, "Calling 64-bit AVX512 NTT");
  NTT::ForwardTransformToBitReverseAVX512<s_default_shift_bits>(
      degree, mod, root_of_unity_powers, precon_root_of_unity_powers, elements);
  return;
#endif

  IVLOG(3, "Calling 64-bit default NTT");
  NTT::ForwardTransformToBitReverse64(degree, mod, root_of_unity_powers,
                                      precon_root_of_unity_powers, elements);
}

}  // namespace lattice
}  // namespace intel
