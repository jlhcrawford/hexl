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

using IntType = std::uint64_t;  // TODO(fboemer): used signed or unsigned?
// NFLLib and SEAL use unsigned, so let's go with unsigned for now

class NTT {
 public:
  // Initializes an NTT object with prime p and degree size
  NTT(size_t p, size_t degree) : m_p(p), m_degree(degree) {
    IVLOG(4, "NTT p " << p << " degree " << degree);
    NTT_CHECK(IsPowerOfTwo(m_degree),
              "degree " << degree << " is not a power of 2");
    NTT_CHECK(p % (2 * degree) == 1, "p mod 2n != 1");

    m_degree_bits = Log2(m_degree);
    m_w = MinimalPrimitiveRoot(2 * degree, m_p);
    IVLOG(4, "m_w " << m_w);
    m_winv = InverseUIntMod(m_w, m_p);
    IVLOG(4, "m_winv " << m_winv);
    ComputeRootOfUnityPowers();
    ComputeInverseRootOfUnityPowers();
  }

  IntType GetMinimalRootOfUnity() const { return m_w; }

  IntType GetRootOfUnityPower(size_t i) { return m_rootOfUnityPowers[i]; }

  void ForwardTransformToBitReverse(std::vector<IntType>* elements);
  void ForwardTransformToBitReverse2(std::vector<IntType>* elements);
  void ReverseTransformFromBitReverse(std::vector<IntType>* elements);

 private:
  // Computed bit-scrambled vector of first m_degree powers
  // of a primitive root.
  void ComputeRootOfUnityPowers();
  void ComputeInverseRootOfUnityPowers();

  size_t m_p;            // prime modulus
  size_t m_degree;       // size of NTT transform, should be power of 2
  size_t m_degree_bits;  // log_2(m_degree)

  uint64_t m_w;     // Minimal 2N'th root of unity
  uint64_t m_winv;  // Inverse of minimal root of unity

  // w^{2^{MaxRoot -j}}, where w is a primitive root of unity for p in
  // bit-reverse order
  // TODO(fboemer)
  std::vector<IntType> m_rootOfUnityPowers;

  // TODO(fboemer)
  std::vector<IntType> m_inverseRootOfUnityPowers;
};

}  // namespace ntt
}  // namespace intel
