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

#include "eltwise/eltwise-cmp-add.hpp"

#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-cmp-add-avx512.hpp"
#include "util/avx512-util.hpp"
#endif

namespace intel {
namespace lattice {

void EltwiseCmpAdd(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                   uint64_t diff, uint64_t n) {
#ifdef LATTICE_HAS_AVX512DQ
  if (has_avx512_dq) {
    EltwiseCmpAddAVX512(operand1, cmp, bound, diff, n);
    return;
  }
#endif
  EltwiseCmpAddNative(operand1, cmp, bound, diff, n);
}

void EltwiseCmpAddNative(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                         uint64_t diff, uint64_t n) {
  switch (cmp) {
    case CMPINT::EQ: {
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] == bound) {
          operand1[i] += diff;
        }
      }
      return;
    }
    case CMPINT::LT:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] < bound) {
          operand1[i] += diff;
        }
      }
      return;
    case CMPINT::LE:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] <= bound) {
          operand1[i] += diff;
        }
      }
      return;
    case CMPINT::FALSE:
      return;
    case CMPINT::NE:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] != bound) {
          operand1[i] += diff;
        }
      }
      return;
    case CMPINT::NLT:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] >= bound) {
          operand1[i] += diff;
        }
      }
      return;

    case CMPINT::NLE:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] > bound) {
          operand1[i] += diff;
        }
      }
      return;
    case CMPINT::TRUE:
      for (size_t i = 0; i < n; ++i) {
        operand1[i] += diff;
      }
  }
}

#ifdef LATTICE_HAS_AVX512DQ
void EltwiseCmpAddAVX512(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                         uint64_t diff, uint64_t n) {
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseCmpAddNative(operand1, cmp, bound, diff, n_mod_8);
    operand1 += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_bound = _mm512_set1_epi64(bound);
  __m512i* v_op_ptr = reinterpret_cast<__m512i*>(operand1);
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op = _mm512_loadu_si512(v_op_ptr);
    __m512i v_add_diff = _mm512_il_cmp_epi64(v_op, v_bound, cmp, diff);
    v_op = _mm512_add_epi64(v_op, v_add_diff);
    _mm512_storeu_si512(v_op_ptr, v_op);
    ++v_op_ptr;
  }
}
#endif

}  // namespace lattice
}  // namespace intel
