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
#include "number-theory.hpp"

namespace intel {
namespace ntt {

// based on
// https://github.com/microsoft/SEAL/blob/master/native/src/seal/util/ntt.cpp#L200
void NTT::ForwardTransformToBitReverse64(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements) {
  NTT_CHECK(CheckArguments(degree, mod), "");

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
#pragma unroll 4
      for (size_t j = j1; j < j2; j++) {
        // The Harvey butterfly: assume X, Y in [0, 2p), and return X', Y' in
        // [0, 4p).
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
    NTT_CHECK(elements[i] < mod,
              "Incorrect modulus reduction " << elements[i] << " >= " << mod);
  }
}

void NTT::ForwardTransformToBitReverse(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements,
    bool use_ifma_if_possible) {
#ifdef NTT_HAS_AVX512IFMA
  // TODO(fboemer): Check 50-bit limit more carefully
  constexpr IntType ifma_mod_bound = (1UL << 50);
  if (use_ifma_if_possible && (mod < ifma_mod_bound)) {
    IVLOG(3, "Calling 52-bit AVX512-IFMA NTT");
    NTT::ForwardTransformToBitReverseAVX512<52>(
        degree, mod, root_of_unity_powers, precon_root_of_unity_powers,
        elements);
    return;
  }
#endif

#ifdef NTT_HAS_AVX512F
  IVLOG(3, "Calling 64-bit AVX512 NTT");
  NTT::ForwardTransformToBitReverseAVX512<64>(
      degree, mod, root_of_unity_powers, precon_root_of_unity_powers, elements);
  return;
#endif

  IVLOG(3, "Calling 64-bit default NTT");
  NTT::ForwardTransformToBitReverse64(degree, mod, root_of_unity_powers,
                                      precon_root_of_unity_powers, elements);
}

}  // namespace ntt
}  // namespace intel
