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

#include "poly/poly-cmp-add.hpp"

#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "poly/poly-cmp-add-internal.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

#ifdef LATTICE_HAS_AVX512F
#include "poly/poly-cmp-add-avx512.hpp"
#include "util/avx512-util.hpp"
#endif

namespace intel {
namespace lattice {

void CmpGtAdd(uint64_t* operand1, uint64_t cmp, uint64_t diff, uint64_t n) {
#ifdef LATTICE_HAS_AVX512F
  if (has_avx512_dq) {
    CmpGtAddAVX512(operand1, cmp, diff, n);
    return;
  }
#endif
  CmpGtAddNative(operand1, cmp, diff, n);
}

void CmpGtAddNative(uint64_t* operand1, uint64_t cmp, uint64_t diff,
                    uint64_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (operand1[i] > cmp) {
      operand1[i] += diff;
    }
  }
}

#ifdef LATTICE_HAS_AVX512F
void CmpGtAddAVX512(uint64_t* operand1, uint64_t cmp, uint64_t diff,
                    uint64_t n) {
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    CmpGtAddNative(operand1, cmp, diff, n_mod_8);
    operand1 += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_cmp = _mm512_set1_epi64(cmp);
  __m512i* v_op_ptr = reinterpret_cast<__m512i*>(operand1);
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op = _mm512_loadu_si512(v_op_ptr);
    __m512i v_add_diff = intel::lattice::_mm512_il_cmp_epi64(
        v_op, v_cmp, CMPINT_ENUM::NLE, diff);
    v_op = _mm512_add_epi64(v_op, v_add_diff);
    _mm512_storeu_si512(v_op_ptr, v_op);
    ++v_op_ptr;
  }
}
#endif

}  // namespace lattice
}  // namespace intel
