// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020-2021 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

#include "eltwise/eltwise-add-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-add-mod-internal.hpp"
#include "intel-lattice/eltwise/eltwise-add-mod.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

void EltwiseAddModAVX512(uint64_t* operand1, const uint64_t* operand2,
                         uint64_t n, const uint64_t modulus) {
  LATTICE_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  LATTICE_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  LATTICE_CHECK(n != 0, "Require n != 0");
  LATTICE_CHECK(modulus > 1, "Require modulus > 1");
  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "pre-add value in operand1 exceeds bound " << modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus,
                       "pre-add value in operand2 exceeds bound " << modulus);

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseAddModNative(operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_modulus = _mm512_set1_epi64(modulus);
  __m512i* vp_operand1 = reinterpret_cast<__m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    __m512i v_result =
        _mm512_il_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);

    _mm512_storeu_si512(vp_operand1, v_result);

    ++vp_operand1;
    ++vp_operand2;
  }

  LATTICE_CHECK_BOUNDS(operand1, n, modulus,
                       "post-mult value in operand1 exceeds bound " << modulus);
}

}  // namespace lattice
}  // namespace intel
