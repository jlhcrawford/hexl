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

#include <unordered_map>
#include <utility>
#include <vector>

#include "number-theory/number-theory.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

struct hash_pair {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2>& p) const {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash1 ^ hash2;
  }
};

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
      : m_p(p), m_degree(degree), m_w(root_of_unity) {
    LATTICE_CHECK(CheckArguments(degree, p), "");
    LATTICE_CHECK(
        IsPrimitiveRoot(m_w, 2 * degree, p),
        m_w << " is not a primitive 2*" << degree << "'th root of unity");

#ifdef LATTICE_HAS_AVX512IFMA
    if (m_p < s_max_ifma_modulus) {
      m_bit_shift = s_max_ifma_modulus_bits;
    }
#endif

    m_degree_bits = Log2(m_degree);

    m_winv = InverseUIntMod(m_w, m_p);
    ComputeRootOfUnityPowers();
  }

  IntType GetMinimalRootOfUnity() const { return m_w; }

  IntType GetDegree() const { return m_degree; }

  IntType GetModulus() const { return m_p; }

  // Returns map of pre-computed root of unity powers
  // Access via s_root_of_unity_powers[<prime modulus, root of
  // unity>] Map (degree, prime modulus, root of unity) to root of unity powers
  static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                            hash_pair>&
  GetStaticRootOfUnityPowers() {
    static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                              hash_pair>
        s_root_of_unity_powers;

    return s_root_of_unity_powers;
  }

  static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                            hash_pair>&
  GetStaticPreconRootOfUnityPowers() {
    static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                              hash_pair>
        s_precon_root_of_unity_powers;

    return s_precon_root_of_unity_powers;
  }

  std::vector<IntType> GetPreconRootOfUnityPowers() {
    std::pair<IntType, IntType> key{m_p, m_w};
    auto it = GetStaticPreconRootOfUnityPowers().find(key);
    LATTICE_CHECK(it != GetStaticPreconRootOfUnityPowers().end(),
                  "Could not find root of unity power");
    return it->second;
  }

  std::vector<IntType> GetRootOfUnityPowers() {
    std::pair<IntType, IntType> key{m_p, m_w};
    auto it = GetStaticRootOfUnityPowers().find(key);
    LATTICE_CHECK(it != GetStaticRootOfUnityPowers().end(),
                  "Could not find root of unity power");
    return it->second;
  }

  IntType GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  // Compute in-place NTT.
  // Results are bit-reversed.

  inline void ForwardTransformToBitReverse(IntType* elements) {
    const auto& root_of_unity_powers = GetRootOfUnityPowers();
    const auto& precon_root_of_unity_powers = GetPreconRootOfUnityPowers();

    ForwardTransformToBitReverse(m_degree, m_p, root_of_unity_powers.data(),
                                 precon_root_of_unity_powers.data(), elements);
  }

  static void ForwardTransformToBitReverse(
      const IntType degree, const IntType mod,
      const IntType* root_of_unity_powers,
      const IntType* precon_root_of_unity_powers, IntType* elements,
      bool use_ifma_if_possible = true);

  static void ForwardTransformToBitReverse64(
      const IntType degree, const IntType mod,
      const IntType* root_of_unity_powers,
      const IntType* precon_root_of_unity_powers, IntType* elements);

#ifdef LATTICE_HAS_AVX512F
  template <int BitShift>
  static void ForwardTransformToBitReverseAVX512(
      const IntType degree, const IntType mod,
      const IntType* root_of_unity_powers,
      const IntType* precon_root_of_unity_powers, IntType* elements);
#endif

  // TODO(sejun) implement
  // Compute in-place inverse NTT.
  // Results are bit-reversed.
  void ReverseTransformFromBitReverse(IntType* elements);

 private:
  static bool CheckArguments(IntType degree, IntType p) {
    // Avoid unused parameter warnings
    (void)degree;
    (void)p;
    LATTICE_CHECK(IsPowerOfTwo(degree),
                  "degree " << degree << " is not a power of 2");
    LATTICE_CHECK(degree <= (1 << s_max_degree_bits),
                  "degree should be less than 2^" << s_max_degree_bits
                                                  << " got " << degree);

    LATTICE_CHECK(p % (2 * degree) == 1, "p mod 2n != 1");
    return true;
  }

  // Computed bit-scrambled vector of first m_degree powers
  // of a primitive root.
  inline void ComputeRootOfUnityPowers() {
    std::pair<IntType, IntType> key = std::make_pair(m_p, m_w);

    auto it = NTT::GetStaticRootOfUnityPowers().find(key);
    if (it != NTT::GetStaticRootOfUnityPowers().end()) {
      return;
    }

    std::vector<IntType> root_of_unity_powers(m_degree);
    std::vector<IntType> precon_root_of_unity_powers(m_degree);

    MultiplyFactor first(1, m_bit_shift, m_p);
    root_of_unity_powers[0] = first.Operand();
    precon_root_of_unity_powers[0] = first.BarrettFactor();
    int idx = 0;
    int prev_idx = idx;
    for (size_t i = 1; i < m_degree; i++) {
      idx = ReverseBitsUInt(i, m_degree_bits);
      MultiplyFactor mf(
          MultiplyUIntMod(root_of_unity_powers[prev_idx], m_w, m_p),
          m_bit_shift, m_p);
      root_of_unity_powers[idx] = mf.Operand();
      precon_root_of_unity_powers[idx] = mf.BarrettFactor();
      prev_idx = idx;
    }

    NTT::GetStaticRootOfUnityPowers()[key] = std::move(root_of_unity_powers);
    NTT::GetStaticPreconRootOfUnityPowers()[key] =
        std::move(precon_root_of_unity_powers);
  }

  size_t m_p;  // prime modulus

  // Bit shift to use in computing Barrett reduction
  // It should be 52 if m_p < (1 << 52) and HAS_AVX512IFMA is defined
  // Otherwise, it defaults to 64
  size_t m_bit_shift{64};
  size_t m_degree;       // N: size of NTT transform, should be power of 2
  size_t m_degree_bits;  // log_2(m_degree)
  static const size_t s_max_degree_bits{20};  // Maximum power of 2 in degree

  // Maximum number of bits in modulus;
  static const size_t s_max_modulus_bits{62};

  // Maximum number of bits in modulus to use IFMA acceleration;
  static const size_t s_max_ifma_modulus_bits{52};

  // Maximum modulus size to use IFMA acceleration;
  static const size_t s_max_ifma_modulus{(1UL << s_max_ifma_modulus_bits) - 1};

  uint64_t m_w;     // A 2N'th root of unity
  uint64_t m_winv;  // Inverse of minimal root of unity
};

}  // namespace lattice
}  // namespace intel
