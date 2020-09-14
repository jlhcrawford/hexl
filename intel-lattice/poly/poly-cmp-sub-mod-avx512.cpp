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

#include "poly/poly-cmp-sub-mod-avx512.hpp"

#include "logging/logging.hpp"
#include "poly/poly-cmp-sub-mod-internal.hpp"
#include "poly/poly-cmp-sub-mod.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

void CmpGtSubModAVX512(uint64_t* operand1, uint64_t cmp, uint64_t diff,
                       uint64_t modulus, uint64_t n) {
  LATTICE_CHECK(n % 8 == 0,
                "CmpGtSubModAVX512 supports n % 8 == 0; got n = " << n);
  LATTICE_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);

  __m512i* v_op_ptr = reinterpret_cast<__m512i*>(operand1);
  __m512i v_diff = _mm512_set1_epi64(diff);
  __m512i v_cmp = _mm512_set1_epi64(cmp);
  __m512i v_modulus = _mm512_set1_epi64(modulus);

  uint64_t mu = static_cast<uint64_t>((uint128_t(1) << 64) / modulus);
  __m512i v_mu = _mm512_set1_epi64(mu);

  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op = _mm512_loadu_si512(v_op_ptr);
    __mmask8 op_le_cmp = _mm512_cmp_epu64_mask(v_op, v_cmp, CMPINT_ENUM::LE);

    v_op = _mm512_il_barrett_reduce64(v_op, v_modulus, v_mu);

    __m512i v_to_add =
        _mm512_il_cmp_epi64(v_op, v_diff, CMPINT_ENUM::LT, modulus);
    v_to_add = _mm512_sub_epi64(v_to_add, v_diff);
    v_to_add = _mm512_mask_set1_epi64(v_to_add, op_le_cmp, 0);

    v_op = _mm512_add_epi64(v_op, v_to_add);
    _mm512_storeu_si512(v_op_ptr, v_op);
    ++v_op_ptr;
  }
}

}  // namespace lattice
}  // namespace intel
