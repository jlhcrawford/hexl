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

#include "ntt/ntt-internal.hpp"

#include <iostream>
#include <memory>
#include <utility>

#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
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

void NTT::NTTImpl::ComputeRootOfUnityPowers() {
  AlignedVector<uint64_t> root_of_unity_powers(m_degree);
  AlignedVector<uint64_t> inv_root_of_unity_powers(m_degree);

  // 64-bit  precon
  root_of_unity_powers[0] = 1;
  inv_root_of_unity_powers[0] = InverseUIntMod(1, m_p);
  int idx = 0;
  int prev_idx = idx;

  for (size_t i = 1; i < m_degree; i++) {
    idx = ReverseBitsUInt(i, m_degree_bits);
    root_of_unity_powers[idx] =
        MultiplyUIntMod(root_of_unity_powers[prev_idx], m_w, m_p);
    inv_root_of_unity_powers[idx] =
        InverseUIntMod(root_of_unity_powers[idx], m_p);

    prev_idx = idx;
  }

  // Reordering inv_root_of_powers
  AlignedVector<uint64_t> temp(m_degree);
  temp[0] = inv_root_of_unity_powers[0];
  idx = 1;

  for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
    for (size_t i = 0; i < m; i++) {
      temp[idx] = inv_root_of_unity_powers[m + i];
      idx++;
    }
  }
  inv_root_of_unity_powers = temp;

  // 64-bit preconditioned root of unity powers
  AlignedVector<uint64_t> precon64_root_of_unity_powers;
  precon64_root_of_unity_powers.reserve(m_degree);
  for (uint64_t root_of_unity : root_of_unity_powers) {
    MultiplyFactor mf(root_of_unity, 64, m_p);
    precon64_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon64RootOfUnityPowers() =
      std::move(precon64_root_of_unity_powers);

  // 52-bit preconditioned root of unity powers
  AlignedVector<uint64_t> precon52_root_of_unity_powers;
  precon52_root_of_unity_powers.reserve(m_degree);
  for (uint64_t root_of_unity : root_of_unity_powers) {
    MultiplyFactor mf(root_of_unity, 52, m_p);
    precon64_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon52RootOfUnityPowers() =
      std::move(precon64_root_of_unity_powers);

  NTT::NTTImpl::GetRootOfUnityPowers() = std::move(root_of_unity_powers);

  // 64-bit preconditioned inverse root of unity powers
  AlignedVector<uint64_t> precon64_inv_root_of_unity_powers;
  precon64_inv_root_of_unity_powers.reserve(m_degree);
  for (uint64_t inv_root_of_unity : inv_root_of_unity_powers) {
    MultiplyFactor mf(inv_root_of_unity, 64, m_p);
    precon64_inv_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon64InvRootOfUnityPowers() =
      std::move(precon64_inv_root_of_unity_powers);

  // 52-bit preconditioned inverse root of unity powers
  AlignedVector<uint64_t> precon52_inv_root_of_unity_powers;
  precon52_inv_root_of_unity_powers.reserve(m_degree);
  for (uint64_t inv_root_of_unity : inv_root_of_unity_powers) {
    MultiplyFactor mf(inv_root_of_unity, 52, m_p);
    precon52_inv_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon52InvRootOfUnityPowers() =
      std::move(precon52_inv_root_of_unity_powers);

  NTT::NTTImpl::GetInvRootOfUnityPowers() = std::move(inv_root_of_unity_powers);
}

void NTT::NTTImpl::ComputeForward(uint64_t* elements, bool full_reduce) {
  LATTICE_CHECK(
      m_bit_shift == s_ifma_shift_bits || m_bit_shift == s_default_shift_bits,
      "Bit shift " << m_bit_shift << " should be either " << s_ifma_shift_bits
                   << " or " << s_default_shift_bits);

#ifdef LATTICE_HAS_AVX512IFMA
  if (has_avx512_ifma && m_bit_shift == s_ifma_shift_bits &&
      (m_p < s_max_ifma_modulus && (m_degree >= 16))) {
    const uint64_t* root_of_unity_powers = GetRootOfUnityPowersPtr();
    const uint64_t* precon_root_of_unity_powers =
        GetPrecon52RootOfUnityPowersPtr();

    IVLOG(3, "Calling 52-bit AVX512-IFMA NTT");
    if (full_reduce) {
      ForwardTransformToBitReverseAVX512<s_ifma_shift_bits>(
          m_degree, m_p, root_of_unity_powers, precon_root_of_unity_powers,
          elements, true);
    } else {
      ForwardTransformToBitReverseAVX512<s_ifma_shift_bits>(
          m_degree, m_p, root_of_unity_powers, precon_root_of_unity_powers,
          elements, false);
    }
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512DQ
  if (has_avx512_dq && m_degree >= 16) {
    IVLOG(3, "Calling 64-bit AVX512 NTT");
    const uint64_t* root_of_unity_powers = GetRootOfUnityPowersPtr();
    const uint64_t* precon_root_of_unity_powers =
        GetPrecon64RootOfUnityPowersPtr();

    if (full_reduce) {
      ForwardTransformToBitReverseAVX512<s_default_shift_bits>(
          m_degree, m_p, root_of_unity_powers, precon_root_of_unity_powers,
          elements, true);
    } else {
      ForwardTransformToBitReverseAVX512<s_default_shift_bits>(
          m_degree, m_p, root_of_unity_powers, precon_root_of_unity_powers,
          elements, false);
    }
    return;
  }
#endif

  IVLOG(3, "Calling 64-bit default NTT");
  const uint64_t* root_of_unity_powers = GetRootOfUnityPowersPtr();
  const uint64_t* precon_root_of_unity_powers =
      GetPrecon64RootOfUnityPowersPtr();

  ForwardTransformToBitReverse64(m_degree, m_p, root_of_unity_powers,
                                 precon_root_of_unity_powers, elements,
                                 full_reduce);
}

void NTT::NTTImpl::ComputeForward(const uint64_t* elements, uint64_t* result,
                                  bool full_reduce) {
  if (elements != result) {
    std::memcpy(result, elements, m_degree * sizeof(uint64_t));
  }
  ComputeForward(result, full_reduce);
}

void NTT::NTTImpl::ComputeInverse(uint64_t* elements, bool full_reduce) {
  LATTICE_CHECK(
      m_bit_shift == s_ifma_shift_bits || m_bit_shift == s_default_shift_bits,
      "Bit shift " << m_bit_shift << " should be either " << s_ifma_shift_bits
                   << " or " << s_default_shift_bits);

#ifdef LATTICE_HAS_AVX512IFMA
  if (m_bit_shift == s_ifma_shift_bits && (m_p < s_max_ifma_modulus) &&
      (m_degree >= 16)) {
    IVLOG(3, "Calling 52-bit AVX512-IFMA InvNTT");
    const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowersPtr();
    const uint64_t* precon_inv_root_of_unity_powers =
        GetPrecon52InvRootOfUnityPowersPtr();
    InverseTransformFromBitReverseAVX512<s_ifma_shift_bits>(
        m_degree, m_p, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, elements, full_reduce);
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512DQ
  if (has_avx512_dq && m_degree >= 16) {
    IVLOG(3, "Calling 64-bit AVX512 InvNTT");
    const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowersPtr();
    const uint64_t* precon_inv_root_of_unity_powers =
        GetPrecon64InvRootOfUnityPowersPtr();

    InverseTransformFromBitReverseAVX512<s_default_shift_bits>(
        m_degree, m_p, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, elements, full_reduce);
    return;
  }
#endif

  IVLOG(3, "Calling 64-bit default InvNTT");
  const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowersPtr();
  const uint64_t* precon_inv_root_of_unity_powers =
      GetPrecon64InvRootOfUnityPowersPtr();
  InverseTransformFromBitReverse64(m_degree, m_p, inv_root_of_unity_powers,
                                   precon_inv_root_of_unity_powers, elements,
                                   full_reduce);
}

void NTT::NTTImpl::ComputeInverse(const uint64_t* elements, uint64_t* result,
                                  bool full_reduce) {
  if (elements != result) {
    std::memcpy(result, elements, m_degree * sizeof(uint64_t));
  }
  ComputeInverse(result, full_reduce);
}

// NTT API
NTT::NTT() = default;

NTT::NTT(uint64_t degree, uint64_t p)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, p)) {}

NTT::NTT(uint64_t degree, uint64_t p, uint64_t root_of_unity)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, p, root_of_unity)) {}

NTT::~NTT() = default;

void NTT::ComputeForward(uint64_t* elements, bool full_reduce) {
  m_impl->ComputeForward(elements, full_reduce);
}

void NTT::ComputeForward(const uint64_t* elements, uint64_t* result,
                         bool full_reduce) {
  m_impl->ComputeForward(elements, result, full_reduce);
}

void NTT::ComputeInverse(uint64_t* elements, bool full_reduce) {
  m_impl->ComputeInverse(elements, full_reduce);
}

void NTT::ComputeInverse(const uint64_t* elements, uint64_t* result,
                         bool full_reduce) {
  m_impl->ComputeInverse(elements, result, full_reduce);
}

// Free functions

void ForwardTransformToBitReverse64(uint64_t n, uint64_t mod,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t* elements, bool full_reduce) {
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
        LATTICE_CHECK(*X <= (mod * 4), "input X " << (*X) << " too large");
        LATTICE_CHECK(*Y <= (mod * 4), "input Y " << (*Y) << " too large");

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
  if (full_reduce) {
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
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements,
    bool full_reduce) {
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
        // The Harvey butterfly: assume X, Y in [0, 2p), and return X', Y'
        // in [0, 2p). X', Y' = X + Y (mod p), W(X - Y) (mod p).
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

  if (full_reduce) {
    // Reduce from [0, 2p) to [0,p)
    for (size_t i = 0; i < n; ++i) {
      if (elements[i] >= mod) {
        elements[i] -= mod;
      }
      LATTICE_CHECK(elements[i] < mod, "Incorrect modulus reduction in InvNTT"
                                           << elements[i] << " >= " << mod);
    }
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
