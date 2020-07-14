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

#include <complex>
#include <vector>

#include "number-theory.hpp"
#include "util.hpp"

namespace intel {
namespace ntt {

using IntType = std::uint64_t;

// Performs negacyclic NTT and inverse NTT, i.e. the number-theoretic transform
// over Z_p[X]/(X^N+1).
// See Faster arithmetic for number-theoretic transforms - David Harvey
// (https://arxiv.org/abs/1205.2926) for more details.
class NTT {
 public:
  // Initializes an NTT object with degree degree and prime modulus p.
  // @param degree Size of the NTT transform, a.k.a N. Must be a power of 2
  // @param p Prime modulus. Must satisfy p == 1 mod 2N
  // @brief Performs pre-computation necessary for forward and inverse
  // transforms
  NTT(IntType degree, IntType p)
      : NTT(degree, p, MinimalPrimitiveRoot(2 * degree, p)) {}

  // Initializes an NTT object with degree degree and prime modulus p.
  // @param degree Size of the NTT transform, a.k.a N. Must be a power of 2
  // @param p Prime modulus. Must satisfy p == 1 mod 2N
  // @param root_of_unity 2N'th root of unity in F_p
  // @brief Performs pre-computation necessary for forward and inverse
  // transforms
  NTT(IntType degree, IntType p, IntType root_of_unity)
      : m_p(p), m_degree(degree) {
    NTT_CHECK(IsPowerOfTwo(m_degree),
              "degree " << degree << " is not a power of 2");
    NTT_CHECK(m_degree <= (1 << 20),
              "degree should be less than 2^20, got " << degree);

    NTT_CHECK(p % (2 * degree) == 1, "p mod 2n != 1");

    m_degree_bits = Log2(m_degree);
    m_w = root_of_unity;
    m_winv = InverseUIntMod(m_w, m_p);
    ComputeRootOfUnityPowers();
    ComputeInverseRootOfUnityPowers();
  }

  IntType GetMinimalRootOfUnity() const { return m_w; }

  MultiplyFactor GetRootOfUnityPower(size_t i) {
    return m_rootOfUnityPowers[i];
  }

  // Compute in-place NTT.
  // Results are bit-reversed.
  void ForwardTransformToBitReverse(IntType* elements);

  // TODO(sejun) implement
  // Compute in-place inverse NTT.
  // Results are bit-reversed.
  void ReverseTransformFromBitReverse(IntType* elements);

 private:
  // Computed bit-scrambled vector of first m_degree powers
  // of a primitive root.
  void ComputeRootOfUnityPowers();

  // TODO(sejun): implement if needed
  void ComputeInverseRootOfUnityPowers();

  size_t m_p;            // prime modulus
  size_t m_degree;       // N: size of NTT transform, should be power of 2
  size_t m_degree_bits;  // log_2(m_degree)

  uint64_t m_w;     // A 2N'th root of unity
  uint64_t m_winv;  // Inverse of minimal root of unity

  // w^{2^{MaxRoot -j}}, where w is a primitive root of unity for p in
  // bit-reverse order
  // TODO(fboemer)

  std::vector<MultiplyFactor> m_rootOfUnityPowers;

  // TODO(sejun)
  std::vector<IntType> m_inverseRootOfUnityPowers;
};

}  // namespace ntt
}  // namespace intel
