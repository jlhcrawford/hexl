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

#include <stdint.h>

#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ntt/ntt.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"
#include "util/util-internal.hpp"
#include "util/util.hpp"

namespace intel {
namespace lattice {

class NTT::NTTImpl {
 public:
  NTTImpl(uint64_t degree, uint64_t p, uint64_t root_of_unity);
  NTTImpl(uint64_t degree, uint64_t p);

  ~NTTImpl();

  uint64_t GetMinimalRootOfUnity() const { return m_w; }

  uint64_t GetDegree() const { return m_degree; }

  uint64_t GetModulus() const { return m_p; }

  uint64_t GetBitShift() const { return m_bit_shift; }

  uint64_t* GetPrecon64RootOfUnityPowersPtr() {
    auto it = GetStaticPrecon64RootOfUnityPowers().find(m_key);
    LATTICE_CHECK(it != GetStaticPrecon64RootOfUnityPowers().end(),
                  "Could not find 64-bit pre-conditioned root of unity power");
    return it->second.data();
  }

  uint64_t* GetPrecon52RootOfUnityPowersPtr() {
    auto it = GetStaticPrecon52RootOfUnityPowers().find(m_key);
    LATTICE_CHECK(it != GetStaticPrecon52RootOfUnityPowers().end(),
                  "Could not find 52-bit pre-conditioned root of unity power");
    return it->second.data();
  }

  uint64_t* GetRootOfUnityPowersPtr() {
    auto it = GetStaticRootOfUnityPowers().find(m_key);
    LATTICE_CHECK(it != GetStaticRootOfUnityPowers().end(),
                  "Could not find root of unity power");
    return it->second.data();
  }

  std::vector<uint64_t> GetPrecon64RootOfUnityPowers() {
    auto it = GetStaticPrecon64RootOfUnityPowers().find(m_key);
    LATTICE_CHECK(it != GetStaticPrecon64RootOfUnityPowers().end(),
                  "Could not find 64-bit pre-conditioned root of unity power");
    return it->second;
  }

  std::vector<uint64_t> GetPrecon52RootOfUnityPowers() {
    auto it = GetStaticPrecon52RootOfUnityPowers().find(m_key);
    LATTICE_CHECK(it != GetStaticPrecon52RootOfUnityPowers().end(),
                  "Could not find pre-conditioned root of unity power");
    return it->second;
  }

  // Returns the vector of pre-computed root of unity powers for the modulus
  // and root of unity.
  std::vector<uint64_t> GetRootOfUnityPowers() {
    auto it = GetStaticRootOfUnityPowers().find(m_key);
    LATTICE_CHECK(it != GetStaticRootOfUnityPowers().end(),
                  "Could not find root of unity power");
    return it->second;
  }

  // Returns the root of unity at index i.
  uint64_t GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  // Returns the vector of 64-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  std::vector<uint64_t> GetPrecon64InvRootOfUnityPowers() {
    auto it = GetStaticPrecon64InvRootOfUnityPowers().find(m_key_inv);
    LATTICE_CHECK(
        it != GetStaticPrecon64InvRootOfUnityPowers().end(),
        "Could not find 64-bit pre-conditioned inverse root of unity power");
    return it->second;
  }

  // Returns the vector of 52-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  std::vector<uint64_t> GetPrecon52InvRootOfUnityPowers() {
    auto it = GetStaticPrecon52InvRootOfUnityPowers().find(m_key_inv);
    LATTICE_CHECK(
        it != GetStaticPrecon52InvRootOfUnityPowers().end(),
        "Could not find 52-bit pre-conditioned inverse root of unity power");
    return it->second;
  }

  uint64_t* GetPrecon64InvRootOfUnityPowersPtr() {
    auto it = GetStaticPrecon64InvRootOfUnityPowers().find(m_key_inv);
    LATTICE_CHECK(
        it != GetStaticPrecon64InvRootOfUnityPowers().end(),
        "Could not find 64-bit pre-conditioned inverse root of unity power");
    return it->second.data();
  }

  uint64_t* GetPrecon52InvRootOfUnityPowersPtr() {
    auto it = GetStaticPrecon52InvRootOfUnityPowers().find(m_key_inv);
    LATTICE_CHECK(
        it != GetStaticPrecon52InvRootOfUnityPowers().end(),
        "Could not find 52-bit pre-conditioned inverse root of unity power");
    return it->second.data();
  }

  std::vector<uint64_t> GetInvRootOfUnityPowers() {
    auto it = GetStaticInvRootOfUnityPowers().find(m_key_inv);
    LATTICE_CHECK(it != GetStaticInvRootOfUnityPowers().end(),
                  "Could not find inversed root of unity power");
    return it->second;
  }

  uint64_t* GetInvRootOfUnityPowersPtr() {
    auto it = GetStaticInvRootOfUnityPowers().find(m_key_inv);
    LATTICE_CHECK(it != GetStaticInvRootOfUnityPowers().end(),
                  "Could not find inversed root of unity power");
    return it->second.data();
  }

  uint64_t GetInvRootOfUnityPower(size_t i) {
    return GetInvRootOfUnityPowers()[i];
  }

  // Returns the map of pre-computed root of unity powers
  // Access via s_root_of_unity_powers[<degree, prime modulus, root of
  // unity>]
  static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                            std::vector<uint64_t>, hash_tuple>&
  GetStaticRootOfUnityPowers() {
    static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                              std::vector<uint64_t>, hash_tuple>
        s_root_of_unity_powers;

    return s_root_of_unity_powers;
  }

  // Returns the map of pre-computed pre-conditioned root of unity powers.
  // Access via s_precon_root_of_unity_powers[<degree, prime modulus, root of
  // unity>]
  static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                            std::vector<uint64_t>, hash_tuple>&
  GetStaticPrecon64RootOfUnityPowers() {
    static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                              std::vector<uint64_t>, hash_tuple>
        s_precon64_root_of_unity_powers;

    return s_precon64_root_of_unity_powers;
  }

  static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                            std::vector<uint64_t>, hash_tuple>&
  GetStaticPrecon52RootOfUnityPowers() {
    static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                              std::vector<uint64_t>, hash_tuple>
        s_precon52_root_of_unity_powers;

    return s_precon52_root_of_unity_powers;
  }

  // Returns map of pre-computed inverse root of unity powers.
  // Access via s_inv_root_of_unity_powers[<degree, prime modulus, root of
  // unity>]
  static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                            std::vector<uint64_t>, hash_tuple>&
  GetStaticInvRootOfUnityPowers() {
    static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                              std::vector<uint64_t>, hash_tuple>
        s_inv_root_of_unity_powers;
    return s_inv_root_of_unity_powers;
  }

  // Returns map of pre-computed 64-bit pre-conditioned inverse root of unity
  // powers. Access via s_precon_inv_root_of_unity_powers[<degree, prime
  // modulus, root of unity>]
  static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                            std::vector<uint64_t>, hash_tuple>&
  GetStaticPrecon64InvRootOfUnityPowers() {
    static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                              std::vector<uint64_t>, hash_tuple>
        s_precon64_inv_root_of_unity_powers;

    return s_precon64_inv_root_of_unity_powers;
  }

  // Returns map of pre-computed 52-bit pre-conditioned inverse root of unity
  // powers. Access via s_precon_inv_root_of_unity_powers[<degree, prime
  // modulus, root of unity>]
  static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                            std::vector<uint64_t>, hash_tuple>&
  GetStaticPrecon52InvRootOfUnityPowers() {
    static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t>,
                              std::vector<uint64_t>, hash_tuple>
        s_precon52_inv_root_of_unity_powers;

    return s_precon52_inv_root_of_unity_powers;
  }

  void ComputeForward(uint64_t* elements);

  void ComputeInverse(uint64_t* elements);

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
  std::mutex mtx;
  void ComputeRootOfUnityPowers();
  uint64_t m_degree;  // N: size of NTT transform, should be power of 2
  uint64_t m_p;       // prime modulus

  uint64_t m_degree_bits;  // log_2(m_degree)
  // Bit shift to use in computing Barrett reduction
  uint64_t m_bit_shift{s_default_shift_bits};

  uint64_t m_winv;  // Inverse of minimal root of unity
  uint64_t m_w;     // A 2N'th root of unity

  // Key for pre-computed root of unity maps <m_degree, m_p, m_w>
  std::tuple<uint64_t, uint64_t, uint64_t> m_key;

  // Key for pre-computed inverse root of unity maps <m_degree, m_p, m_winv>
  std::tuple<uint64_t, uint64_t, uint64_t> m_key_inv;
};

void ForwardTransformToBitReverse64(uint64_t n, uint64_t mod,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t* elements);

// Reference NTT which is written for clarity rather than performance
// Use for debugging
// @param n Size of the transfrom, a.k.a. degree. Must be a power of two.
// @param mod Prime modulus. Must satisfy Must satisfy p == 1 mod 2N
// @param root_of_unity_powers Powers of 2N'th root of unity in F_p. In
// bit-reversed order
// @param elements Input data. Overwritten with NTT output
void ReferenceForwardTransformToBitReverse(uint64_t n, uint64_t mod,
                                           const uint64_t* root_of_unity_powers,
                                           uint64_t* elements);

void InverseTransformFromBitReverse64(
    uint64_t n, uint64_t mod, const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements);

bool CheckArguments(uint64_t degree, uint64_t p);

}  // namespace lattice
}  // namespace intel
