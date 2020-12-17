// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

#pragma once

#include <stdint.h>

#include <utility>

#include "intel-lattice/ntt/ntt.hpp"
#include "intel-lattice/util/util.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"
#include "util/check.hpp"
#include "util/util-internal.hpp"

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

  AlignedVector<uint64_t>& GetPrecon64RootOfUnityPowers() {
    return m_precon64_root_of_unity_powers;
  }

  uint64_t* GetPrecon64RootOfUnityPowersPtr() {
    return GetPrecon64RootOfUnityPowers().data();
  }

  AlignedVector<uint64_t>& GetPrecon52RootOfUnityPowers() {
    return m_precon52_root_of_unity_powers;
  }

  uint64_t* GetPrecon52RootOfUnityPowersPtr() {
    return GetPrecon52RootOfUnityPowers().data();
  }

  uint64_t* GetRootOfUnityPowersPtr() { return GetRootOfUnityPowers().data(); }

  // Returns the vector of pre-computed root of unity powers for the modulus
  // and root of unity.
  AlignedVector<uint64_t>& GetRootOfUnityPowers() {
    return m_root_of_unity_powers;
  }

  // Returns the root of unity at index i.
  uint64_t GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  // Returns the vector of 64-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  AlignedVector<uint64_t>& GetPrecon64InvRootOfUnityPowers() {
    return m_precon64_inv_root_of_unity_powers;
  }

  uint64_t* GetPrecon64InvRootOfUnityPowersPtr() {
    return GetPrecon64InvRootOfUnityPowers().data();
  }

  // Returns the vector of 52-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  AlignedVector<uint64_t>& GetPrecon52InvRootOfUnityPowers() {
    return m_precon52_inv_root_of_unity_powers;
  }

  uint64_t* GetPrecon52InvRootOfUnityPowersPtr() {
    return GetPrecon52InvRootOfUnityPowers().data();
  }

  AlignedVector<uint64_t>& GetInvRootOfUnityPowers() {
    return m_inv_root_of_unity_powers;
  }

  uint64_t* GetInvRootOfUnityPowersPtr() {
    return GetInvRootOfUnityPowers().data();
  }

  uint64_t GetInvRootOfUnityPower(size_t i) {
    return GetInvRootOfUnityPowers()[i];
  }

  void ComputeForward(uint64_t* elements, bool full_reduce = true);
  void ComputeForward(const uint64_t* elements, uint64_t* result,
                      bool full_reduce = true);

  void ComputeInverse(uint64_t* elements, bool full_reduce = true);
  void ComputeInverse(const uint64_t* elements, uint64_t* result,
                      bool full_reduce = true);

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
  void ComputeRootOfUnityPowers();
  uint64_t m_degree;  // N: size of NTT transform, should be power of 2
  uint64_t m_p;       // prime modulus

  uint64_t m_degree_bits;  // log_2(m_degree)
  // Bit shift to use in computing Barrett reduction
  uint64_t m_bit_shift{s_default_shift_bits};

  uint64_t m_winv;  // Inverse of minimal root of unity
  uint64_t m_w;     // A 2N'th root of unity

  AlignedVector<uint64_t> m_precon52_root_of_unity_powers;
  AlignedVector<uint64_t> m_precon64_root_of_unity_powers;
  AlignedVector<uint64_t> m_root_of_unity_powers;

  AlignedVector<uint64_t> m_precon52_inv_root_of_unity_powers;
  AlignedVector<uint64_t> m_precon64_inv_root_of_unity_powers;
  AlignedVector<uint64_t> m_inv_root_of_unity_powers;
};

void ForwardTransformToBitReverse64(uint64_t n, uint64_t mod,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t* elements, bool full_reduce);

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
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements,
    bool full_reduce);

bool CheckArguments(uint64_t degree, uint64_t p);

}  // namespace lattice
}  // namespace intel
