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

  IntType GetMinimalRootOfUnity() const { return m_w; }

  IntType GetDegree() const { return m_degree; }

  IntType GetModulus() const { return m_p; }

  // Returns the map of pre-computed root of unity powers
  // Access via s_root_of_unity_powers[<prime modulus, root of
  // unity>]
  static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                            hash_pair>&
  GetStaticRootOfUnityPowers() {
    static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                              hash_pair>
        s_root_of_unity_powers;

    return s_root_of_unity_powers;
  }

  // Returns the map of pre-computed pre-conditioned root of unity powers
  // Access via s_precon_root_of_unity_powers[<prime modulus, root of
  // unity>]
  static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                            hash_pair>&
  GetStaticPreconRootOfUnityPowers() {
    static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                              hash_pair>
        s_precon_root_of_unity_powers;

    return s_precon_root_of_unity_powers;
  }

  // Returns map of pre-computed inverse root of unity powers
  // Access via s_inv_root_of_unity_powers[<prime modulus, inverse root of
  // unity>] Map (degree, prime modulus, inverse root of unity) to inverse root
  // of unity powers
  static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                            hash_pair>&
  GetStaticInvRootOfUnityPowers() {
    static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                              hash_pair>
        s_inv_root_of_unity_powers;
    return s_inv_root_of_unity_powers;
  }

  // Returns map of pre-computed 64bit scaled inverse root of unity powers
  // Access via s_inv_scaled_root_of_unity_powers[<prime modulus, inverse root
  // of unity>] Map (degree, prime modulus, inverse root of unity) to scaled
  // inverse root of unity powers
  static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                            hash_pair>&
  GetStaticInvScaledRootOfUnityPowers() {
    static std::unordered_map<std::pair<IntType, IntType>, std::vector<IntType>,
                              hash_pair>
        s_inv_scaled_root_of_unity_powers;

    return s_inv_scaled_root_of_unity_powers;
  }

  // Returns the vector of pre-conditioned pre-computed root of unity powers for
  // the modulus and root of unity
  std::vector<IntType> GetPreconRootOfUnityPowers() {
    std::pair<IntType, IntType> key{m_p, m_w};
    auto it = GetStaticPreconRootOfUnityPowers().find(key);
    LATTICE_CHECK(it != GetStaticPreconRootOfUnityPowers().end(),
                  "Could not find root of unity power");
    return it->second;
  }

  // Returns the vector of pre-computed root of unity powers for the modulus and
  // root of unity
  std::vector<IntType> GetRootOfUnityPowers() {
    std::pair<IntType, IntType> key{m_p, m_w};
    auto it = GetStaticRootOfUnityPowers().find(key);
    LATTICE_CHECK(it != GetStaticRootOfUnityPowers().end(),
                  "Could not find root of unity power");
    return it->second;
  }

  // Returns the root of unity at index i
  IntType GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  std::vector<IntType> GetInvScaledRootOfUnityPowers() {
    std::pair<IntType, IntType> key{m_p, m_winv};
    auto it = GetStaticInvScaledRootOfUnityPowers().find(key);
    LATTICE_CHECK(it != GetStaticInvScaledRootOfUnityPowers().end(),
                  "Could not find inversed root of unity power");
    return it->second;
  }

  std::vector<IntType> GetInvRootOfUnityPowers() {
    std::pair<IntType, IntType> key{m_p, m_winv};
    auto it = GetStaticInvRootOfUnityPowers().find(key);
    LATTICE_CHECK(it != GetStaticInvRootOfUnityPowers().end(),
                  "Could not find inversed root of unity power");
    return it->second;
  }

  IntType GetInvRootOfUnityPower(size_t i) {
    return GetInvRootOfUnityPowers()[i];
  }

  // Compute in-place NTT.
  // Results are bit-reversed.
  void ForwardTransformToBitReverse(IntType* elements) {
    const auto& root_of_unity_powers = GetRootOfUnityPowers();
    const auto& precon_root_of_unity_powers = GetPreconRootOfUnityPowers();

    ForwardTransformToBitReverse(m_degree, m_p, root_of_unity_powers.data(),
                                 precon_root_of_unity_powers.data(), elements,
                                 m_bit_shift);
  }

  // Computes the in-place forward NTT
  // @param n Size of the transfrom, a.k.a. degree. Must be a power of two.
  // @param mod Prime modulus. Must satisfy Must satisfy p == 1 mod 2N
  // @param root_of_unity_powers Powers of 2N'th root of unity in F_p. In
  // bit-reversed order
  // @param precon_root_of_unity_powers Preconditioned root_of_unity_powers
  // @param elements Input data. Overwritten with NTT output
  // @param bit_shift The bit shift used in preconditioning. Should be
  // s_ifma_shift_bits for IFMA and s_default_shift_bits otherwise
  static void ForwardTransformToBitReverse(
      IntType n, IntType mod, const IntType* root_of_unity_powers,
      const IntType* precon_root_of_unity_powers, IntType* elements,
      IntType bit_shift);

  static void ForwardTransformToBitReverse64(
      IntType n, IntType mod, const IntType* root_of_unity_powers,
      const IntType* precon_root_of_unity_powers, IntType* elements);

  // Reference NTT which is written for clarity rather than performance
  // Use for debugging
  // @param n Size of the transfrom, a.k.a. degree. Must be a power of two.
  // @param mod Prime modulus. Must satisfy Must satisfy p == 1 mod 2N
  // @param root_of_unity_powers Powers of 2N'th root of unity in F_p. In
  // bit-reversed order
  // @param elements Input data. Overwritten with NTT output
  static void ReferenceForwardTransformToBitReverse(
      IntType n, IntType mod, const IntType* root_of_unity_powers,
      IntType* elements);

#ifdef LATTICE_HAS_AVX512F
  template <int BitShift>
  static void ForwardTransformToBitReverseAVX512(
      const IntType n, const IntType mod, const IntType* root_of_unity_powers,
      const IntType* precon_root_of_unity_powers, IntType* elements);
#endif

  inline void InverseTransformToBitReverse(IntType* elements) {
    const auto& inv_root_of_unity_powers = GetInvRootOfUnityPowers();
    const auto& inv_scaled_root_of_unity_powers =
        GetInvScaledRootOfUnityPowers();

    InverseTransformToBitReverse(m_degree, m_p, inv_root_of_unity_powers.data(),
                                 inv_scaled_root_of_unity_powers.data(),
                                 elements);
  }

  static void InverseTransformToBitReverse(
      const IntType n, const IntType mod,
      const IntType* inv_root_of_unity_powers,
      const IntType* inv_scaled_root_of_unity_powers, IntType* elements);
  // TODO(skim) Add after invNTT AVX512 IFMA - (bool use_ifma_if_possible =
  // true)

  // Inverse negacyclic NTT using Harvey's butterfly. (See Patrick Longa and
  // Michael Naehrig - https://eprint.iacr.org/2016/504.pdf) Merge inverse root
  // of unity with inverse degree and modulus
  static void InverseTransformToBitReverse64(
      const IntType n, const IntType mod,
      const IntType* inv_root_of_unity_powers,
      const IntType* inv_scaled_root_of_unity_powers, IntType* elements);

// TODO(sejun) investigate how to use IFMA for inverse
#ifdef LATTICE_HAS_AVX512F
  template <int BitShift>
  static void InverseTransformToBitReverseAVX512(
      const IntType n, const IntType mod,
      const IntType* inv_root_of_unity_powers,
      const IntType* inv_scaled_root_of_unity_powers, IntType* elements);
#endif

  static const size_t s_max_degree_bits{20};  // Maximum power of 2 in degree

  // Maximum number of bits in modulus;
  static const size_t s_max_modulus_bits{62};

  // Default bit shift used in Barrett precomputation
  static const size_t s_default_shift_bits{64};

  // Maximum number of bits in modulus to use IFMA acceleration
  static const size_t s_max_ifma_modulus_bits{50};

  // Bit shift used in Barrett precomputation when IFMA acceleration is enabled
  static const size_t s_ifma_shift_bits{52};

  // Maximum modulus size to use IFMA acceleration
  static const size_t s_max_ifma_modulus{1UL << s_max_ifma_modulus_bits};

 private:
  // Computes the bit-scrambled vector of first m_degree powers
  // of a primitive root.
  void ComputeRootOfUnityPowers() {
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
    std::vector<IntType> inv_scaled_root_of_unity_powers(m_degree);

    MultiplyFactor first(1, m_bit_shift, m_p);
    root_of_unity_powers[0] = first.Operand();
    precon_root_of_unity_powers[0] = first.BarrettFactor();

    MultiplyFactor first_inv(InverseUIntMod(first.Operand(), m_p), m_bit_shift,
                             m_p);
    inv_root_of_unity_powers[0] = first_inv.Operand();
    inv_scaled_root_of_unity_powers[0] = first_inv.BarrettFactor();
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
      inv_scaled_root_of_unity_powers[idx] = mf_inv.BarrettFactor();

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

    // Reordering inv_scaled_root_of_powers
    temp[0] = inv_scaled_root_of_unity_powers[0];
    idx = 1;
    for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
      for (size_t i = 0; i < m; i++) {
        temp[idx] = inv_scaled_root_of_unity_powers[m + i];
        idx++;
      }
    }
    inv_scaled_root_of_unity_powers = std::move(temp);

    NTT::GetStaticRootOfUnityPowers()[key] = std::move(root_of_unity_powers);
    NTT::GetStaticPreconRootOfUnityPowers()[key] =
        std::move(precon_root_of_unity_powers);

    NTT::GetStaticInvRootOfUnityPowers()[key_inv] =
        std::move(inv_root_of_unity_powers);
    NTT::GetStaticInvScaledRootOfUnityPowers()[key_inv] =
        std::move(inv_scaled_root_of_unity_powers);
  }

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

  size_t m_degree;  // N: size of NTT transform, should be power of 2
  size_t m_p;       // prime modulus

  size_t m_degree_bits;  // log_2(m_degree)
  // Bit shift to use in computing Barrett reduction
  size_t m_bit_shift{s_default_shift_bits};

  uint64_t m_w;     // A 2N'th root of unity
  uint64_t m_winv;  // Inverse of minimal root of unity
};

}  // namespace lattice
}  // namespace intel
