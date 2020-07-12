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

#include "logging/logging.hpp"
#include "number-theory.hpp"

namespace intel {
namespace ntt {

void NTT::ComputeRootOfUnityPowers() {
  m_rootOfUnityPowers.clear();
  m_rootOfUnityPowers.resize(m_degree);

  m_rootOfUnityPowers[0] = MultiplyFactor(1, m_p);
  int idx = 0;
  int prev_idx = idx;
  for (size_t i = 1; i < m_degree; i++) {
    idx = ReverseBitsUInt(i, m_degree_bits);
    m_rootOfUnityPowers[idx] = MultiplyFactor(
        MultiplyUIntMod(m_rootOfUnityPowers[prev_idx].Operand(), m_w, m_p),
        m_p);
    prev_idx = idx;
  }
}

void NTT::ComputeInverseRootOfUnityPowers() {
  // TODO(sejun)
}

// based on
// https://github.com/microsoft/SEAL/blob/master/native/src/seal/util/ntt.cpp#L200
void NTT::ForwardTransformToBitReverse(IntType* elements) {
  uint64_t mod = m_p;
  uint64_t twice_mod = mod << 1;

  size_t n = m_degree;
  size_t t = (n >> 1);

  uint64_t* input = elements;

  for (size_t m = 1; m < n; m <<= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++) {
      size_t j2 = j1 + t;
      const MultiplyFactor W = m_rootOfUnityPowers[m + i];

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
        Q = MultiplyUIntModLazy(*Y, W, mod);
        *X++ = tx + Q;
        *Y++ = tx + twice_mod - Q;
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }
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
