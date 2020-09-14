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

#include "ntt/ntt.hpp"

#include <iostream>
#include <unordered_map>
#include <utility>

#include "logging/logging.hpp"
#include "ntt/ntt-internal.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"

#ifdef LATTICE_HAS_AVX512F
#include "ntt/ntt-avx512.hpp"
#endif

namespace intel {
namespace lattice {

NTT::NTTImpl::NTTImpl(uint64_t degree, uint64_t p, uint64_t root_of_unity)
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

NTT::NTTImpl::NTTImpl(uint64_t degree, uint64_t p)
    : NTTImpl(degree, p, MinimalPrimitiveRoot(2 * degree, p)) {}

NTT::NTTImpl::~NTTImpl() = default;

std::vector<uint64_t> NTT::NTTImpl::GetPreconRootOfUnityPowers() {
  std::tuple<uint64_t, uint64_t, uint64_t> key{m_degree, m_p, m_w};
  auto it = GetStaticPreconRootOfUnityPowers().find(key);
  LATTICE_CHECK(it != GetStaticPreconRootOfUnityPowers().end(),
                "Could not find pre-conditioned root of unity power");
  return it->second;
}

std::vector<uint64_t> NTT::NTTImpl::GetRootOfUnityPowers() {
  std::tuple<uint64_t, uint64_t, uint64_t> key{m_degree, m_p, m_w};
  auto it = GetStaticRootOfUnityPowers().find(key);
  LATTICE_CHECK(it != GetStaticRootOfUnityPowers().end(),
                "Could not find root of unity power");
  return it->second;
}

uint64_t NTT::NTTImpl::GetRootOfUnityPower(size_t i) {
  return GetRootOfUnityPowers()[i];
}

std::vector<uint64_t> NTT::NTTImpl::GetPreconInvRootOfUnityPowers() {
  std::tuple<uint64_t, uint64_t, uint64_t> key{m_degree, m_p, m_winv};
  auto it = GetStaticPreconInvRootOfUnityPowers().find(key);
  LATTICE_CHECK(it != GetStaticPreconInvRootOfUnityPowers().end(),
                "Could not find pre-conditioned inverse root of unity power");
  return it->second;
}

std::vector<uint64_t> NTT::NTTImpl::GetInvRootOfUnityPowers() {
  std::tuple<uint64_t, uint64_t, uint64_t> key{m_degree, m_p, m_winv};
  auto it = GetStaticInvRootOfUnityPowers().find(key);
  LATTICE_CHECK(it != GetStaticInvRootOfUnityPowers().end(),
                "Could not find inversed root of unity power");
  return it->second;
}

uint64_t NTT::NTTImpl::GetInvRootOfUnityPower(size_t i) {
  return GetInvRootOfUnityPowers()[i];
}

void NTT::NTTImpl::ComputeRootOfUnityPowers() {
  {
    std::tuple<uint64_t, uint64_t, uint64_t> key =
        std::make_tuple(m_degree, m_p, m_w);
    std::tuple<uint64_t, uint64_t, uint64_t> key_inv =
        std::make_tuple(m_degree, m_p, m_winv);

    auto it = NTT::NTTImpl::GetStaticRootOfUnityPowers().find(key);
    if (it != NTT::NTTImpl::GetStaticRootOfUnityPowers().end()) {
      return;
    }

    auto it_inv = NTT::NTTImpl::GetStaticInvRootOfUnityPowers().find(key_inv);
    if (it_inv != NTT::NTTImpl::GetStaticInvRootOfUnityPowers().end()) {
      return;
    }

    std::vector<uint64_t> root_of_unity_powers(m_degree);
    std::vector<uint64_t> precon_root_of_unity_powers(m_degree);
    std::vector<uint64_t> inv_root_of_unity_powers(m_degree);
    std::vector<uint64_t> precon_inv_root_of_unity_powers(m_degree);

    MultiplyFactor first(1, m_bit_shift, m_p);
    root_of_unity_powers[0] = first.Operand();
    precon_root_of_unity_powers[0] = first.BarrettFactor();

    MultiplyFactor first_inv(InverseUIntMod(first.Operand(), m_p), m_bit_shift,
                             m_p);
    inv_root_of_unity_powers[0] = first_inv.Operand();
    precon_inv_root_of_unity_powers[0] = first_inv.BarrettFactor();
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
      precon_inv_root_of_unity_powers[idx] = mf_inv.BarrettFactor();

      prev_idx = idx;
    }

    // Reordering inv_root_of_powers
    std::vector<uint64_t> temp(m_degree);
    temp[0] = inv_root_of_unity_powers[0];
    idx = 1;
    for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
      for (size_t i = 0; i < m; i++) {
        temp[idx] = inv_root_of_unity_powers[m + i];
        idx++;
      }
    }
    inv_root_of_unity_powers = temp;

    // Reordering precon_inv_root_of_unity_powers
    temp[0] = precon_inv_root_of_unity_powers[0];
    idx = 1;
    for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
      for (size_t i = 0; i < m; i++) {
        temp[idx] = precon_inv_root_of_unity_powers[m + i];
        idx++;
      }
    }
    precon_inv_root_of_unity_powers = std::move(temp);

    NTT::NTTImpl::GetStaticRootOfUnityPowers()[key] =
        std::move(root_of_unity_powers);
    NTT::NTTImpl::GetStaticPreconRootOfUnityPowers()[key] =
        std::move(precon_root_of_unity_powers);

    NTT::NTTImpl::GetStaticInvRootOfUnityPowers()[key_inv] =
        std::move(inv_root_of_unity_powers);
    NTT::NTTImpl::GetStaticPreconInvRootOfUnityPowers()[key_inv] =
        std::move(precon_inv_root_of_unity_powers);
  }
}

void NTT::NTTImpl::ComputeForward(uint64_t* elements) {
  const auto& root_of_unity_powers = GetRootOfUnityPowers();
  const auto& precon_root_of_unity_powers = GetPreconRootOfUnityPowers();

  LATTICE_CHECK(
      m_bit_shift == s_ifma_shift_bits || m_bit_shift == s_default_shift_bits,
      "Bit shift " << m_bit_shift << " should be either " << s_ifma_shift_bits
                   << " or " << s_default_shift_bits);

#ifdef LATTICE_HAS_AVX512IFMA
  if (m_bit_shift == s_ifma_shift_bits && (m_p < s_max_ifma_modulus)) {
    IVLOG(3, "Calling 52-bit AVX512-IFMA NTT");
    ForwardTransformToBitReverseAVX512<s_ifma_shift_bits>(
        m_degree, m_p, root_of_unity_powers.data(),
        precon_root_of_unity_powers.data(), elements);
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512F
  IVLOG(3, "Calling 64-bit AVX512 NTT");
  ForwardTransformToBitReverseAVX512<s_default_shift_bits>(
      m_degree, m_p, root_of_unity_powers.data(),
      precon_root_of_unity_powers.data(), elements);
  return;
#endif

  IVLOG(3, "Calling 64-bit default NTT");
  ForwardTransformToBitReverse64(m_degree, m_p, root_of_unity_powers.data(),
                                 precon_root_of_unity_powers.data(), elements);
}

void NTT::NTTImpl::ComputeInverse(uint64_t* elements) {
  const auto& inv_root_of_unity_powers = GetInvRootOfUnityPowers();
  const auto& precon_inv_root_of_unity_powers = GetPreconInvRootOfUnityPowers();

  LATTICE_CHECK(
      m_bit_shift == s_ifma_shift_bits || m_bit_shift == s_default_shift_bits,
      "Bit shift " << m_bit_shift << " should be either " << s_ifma_shift_bits
                   << " or " << s_default_shift_bits);

#ifdef LATTICE_HAS_AVX512IFMA
  if (m_bit_shift == s_ifma_shift_bits && (m_p < s_max_ifma_modulus)) {
    IVLOG(3, "Calling 52-bit AVX512-IFMA InvNTT");
    InverseTransformFromBitReverseAVX512<s_ifma_shift_bits>(
        m_degree, m_p, inv_root_of_unity_powers.data(),
        precon_inv_root_of_unity_powers.data(), elements);
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512F
  IVLOG(3, "Calling 64-bit AVX512 InvNTT");
  InverseTransformFromBitReverseAVX512<s_default_shift_bits>(
      m_degree, m_p, inv_root_of_unity_powers.data(),
      precon_inv_root_of_unity_powers.data(), elements);
  return;
#endif

  IVLOG(3, "Calling 64-bit default InvNTT");
  InverseTransformFromBitReverse64(
      m_degree, m_p, inv_root_of_unity_powers.data(),
      precon_inv_root_of_unity_powers.data(), elements);
}

// NTT API
NTT::NTT() = default;

NTT::NTT(uint64_t degree, uint64_t p)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, p)) {}

NTT::NTT(uint64_t degree, uint64_t p, uint64_t root_of_unity)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, p, root_of_unity)) {}

NTT::~NTT() = default;

void NTT::ComputeForward(uint64_t* elements) {
  m_impl->ComputeForward(elements);
}

void NTT::ComputeInverse(uint64_t* elements) {
  m_impl->ComputeInverse(elements);
}

// Free functions

void ForwardTransformToBitReverse64(uint64_t n, uint64_t mod,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t* elements) {
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
        // The Harvey butterfly: assume X, Y in [0, 4p), and return X', Y'
        // in [0, 4p). See Algorithm 4 of
        // https://arxiv.org/pdf/1205.2926.pdf
        // X', Y' = X + WY, X - WY (mod p).
        LATTICE_CHECK(*X <= (mod * 4), "input X " << (tx + Q) << " too large");
        LATTICE_CHECK(*Y <= (mod * 4), "input Y " << (tx + Q) << " too large");

        tx = *X - (twice_mod & static_cast<uint64_t>(
                                   -static_cast<int64_t>(*X >= twice_mod)));
        Q = MultiplyUIntModLazy<64>(*Y, W_op, W_precon, mod);

        *X++ = tx + Q;
        *Y++ = tx + twice_mod - Q;

        LATTICE_CHECK(tx + Q <= (mod * 4),
                      "ouput X " << (tx + Q) << " too large");
        LATTICE_CHECK(tx + twice_mod - Q <= (mod * 4),
                      "output Y " << (tx + twice_mod - Q) << " too large");
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
    LATTICE_CHECK(elements[i] < mod, "Incorrect modulus reduction in NTT "
                                         << elements[i] << " >= " << mod);
  }
}

void ReferenceForwardTransformToBitReverse(uint64_t n, uint64_t mod,
                                           const uint64_t* root_of_unity_powers,
                                           uint64_t* elements) {
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

void InverseTransformFromBitReverse64(
    uint64_t n, uint64_t mod, const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements) {
  LATTICE_CHECK(CheckArguments(n, mod), "");

  uint64_t twice_mod = mod << 1;
  size_t t = 1;
  size_t root_index = 1;

  for (size_t m = (n >> 1); m > 1; m >>= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++, root_index++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = inv_root_of_unity_powers[root_index];
      const uint64_t W_op_precon = precon_inv_root_of_unity_powers[root_index];

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
        // The Harvey butterfly: assume X, Y in [0, 4p), and return X', Y'
        // in [0, 4p). X', Y' = X + Y (mod p), W(X - Y) (mod p).
        tx = *X + *Y;
        ty = *X + twice_mod - *Y;

        *X++ = tx - (twice_mod & static_cast<uint64_t>(
                                     (-static_cast<int64_t>(tx >= twice_mod))));
        *Y++ = MultiplyUIntModLazy<64>(ty, W_op, W_op_precon, mod);
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
    LATTICE_CHECK(elements[i] < mod, "Incorrect modulus reduction in InvNTT"
                                         << elements[i] << " >= " << mod);
  }
}

bool CheckArguments(uint64_t degree, uint64_t p) {
  // Avoid unused parameter warnings
  (void)degree;
  (void)p;
  LATTICE_CHECK(IsPowerOfTwo(degree),
                "degree " << degree << " is not a power of 2");
  LATTICE_CHECK(degree <= (1 << NTT::NTTImpl::s_max_degree_bits),
                "degree should be less than 2^"
                    << NTT::NTTImpl::s_max_degree_bits << " got " << degree);

  LATTICE_CHECK(p % (2 * degree) == 1, "p mod 2n != 1");
  return true;
}

}  // namespace lattice
}  // namespace intel
