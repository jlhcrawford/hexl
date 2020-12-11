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

#include "eltwise/eltwise-cmp-sub-mod.hpp"

#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-cmp-sub-mod-avx512.hpp"
#include "util/avx512-util.hpp"
#endif

namespace intel {
namespace lattice {

void EltwiseCmpSubMod(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                      uint64_t diff, uint64_t modulus, uint64_t n) {
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");

#ifdef LATTICE_HAS_AVX512DQ
  if (has_avx512_dq) {
    EltwiseCmpSubModAVX512(operand1, cmp, bound, diff, modulus, n);
    return;
  }
#endif
  EltwiseCmpSubModNative(operand1, cmp, bound, diff, modulus, n);
}

void EltwiseCmpSubModNative(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                            uint64_t diff, uint64_t modulus, uint64_t n) {
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");

  IVLOG(3, "Calling EltwiseCmpSubModNative");

  LATTICE_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);
  for (size_t i = 0; i < n; ++i) {
    uint64_t op = operand1[i];

    bool op_cmp = Compare(cmp, op, bound);
    op %= modulus;

    if (op_cmp) {
      op = SubUIntMod(op, diff, modulus);
    }
    operand1[i] = op;
  }
}

#ifdef LATTICE_HAS_AVX512DQ
void EltwiseCmpSubModAVX512(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                            uint64_t diff, uint64_t modulus, uint64_t n) {
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");

  IVLOG(3, "Calling CmpSubModAVX512");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseCmpSubModNative(operand1, cmp, bound, diff, modulus, n_mod_8);
    operand1 += n_mod_8;
    n -= n_mod_8;
  }
  LATTICE_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);

  __m512i* v_op_ptr = reinterpret_cast<__m512i*>(operand1);
  __m512i v_bound = _mm512_set1_epi64(bound);
  __m512i v_diff = _mm512_set1_epi64(diff);
  __m512i v_modulus = _mm512_set1_epi64(modulus);

  uint64_t mu = static_cast<uint64_t>((uint128_t(1) << 64) / modulus);
  __m512i v_mu = _mm512_set1_epi64(mu);

  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op = _mm512_loadu_si512(v_op_ptr);
    __mmask8 op_le_cmp = _mm512_il_cmp_epu64_mask(v_op, v_bound, Not(cmp));

    v_op = _mm512_il_barrett_reduce64(v_op, v_modulus, v_mu);

    __m512i v_to_add = _mm512_il_cmp_epi64(v_op, v_diff, CMPINT::LT, modulus);
    v_to_add = _mm512_sub_epi64(v_to_add, v_diff);
    v_to_add = _mm512_mask_set1_epi64(v_to_add, op_le_cmp, 0);

    v_op = _mm512_add_epi64(v_op, v_to_add);
    _mm512_storeu_si512(v_op_ptr, v_op);
    ++v_op_ptr;
  }
}
#endif

}  // namespace lattice
}  // namespace intel
