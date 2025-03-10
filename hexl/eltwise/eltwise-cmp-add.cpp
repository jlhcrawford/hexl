// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-cmp-add.hpp"

#include "eltwise/eltwise-cmp-add-avx512.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/avx512-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpAdd(uint64_t* result, const uint64_t* operand1, uint64_t n,
                   CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    EltwiseCmpAddAVX512(result, operand1, n, cmp, bound, diff);
    return;
  }
#endif
  EltwiseCmpAddNative(result, operand1, n, cmp, bound, diff);
}

void EltwiseCmpAddNative(uint64_t* result, const uint64_t* operand1, uint64_t n,
                         CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");

  switch (cmp) {
    case CMPINT::EQ: {
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] == bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;
    }
    case CMPINT::LT:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] < bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;
    case CMPINT::LE:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] <= bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;
    case CMPINT::FALSE:
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand1[i];
      }
      break;
    case CMPINT::NE:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] != bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;
    case CMPINT::NLT:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] >= bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;
    case CMPINT::NLE:
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] > bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;
    case CMPINT::TRUE:
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand1[i] + diff;
      }
      break;
  }
}

#ifdef HEXL_HAS_AVX512DQ
void EltwiseCmpAddAVX512(uint64_t* result, const uint64_t* operand1, uint64_t n,
                         CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseCmpAddNative(result, operand1, n_mod_8, cmp, bound, diff);
    operand1 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_bound = _mm512_set1_epi64(static_cast<int64_t>(bound));
  const __m512i* v_op_ptr = reinterpret_cast<const __m512i*>(operand1);
  __m512i* v_result_ptr = reinterpret_cast<__m512i*>(result);
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op = _mm512_loadu_si512(v_op_ptr);
    __m512i v_add_diff = _mm512_hexl_cmp_epi64(v_op, v_bound, cmp, diff);
    v_op = _mm512_add_epi64(v_op, v_add_diff);
    _mm512_storeu_si512(v_result_ptr, v_op);

    ++v_result_ptr;
    ++v_op_ptr;
  }
}
#endif

}  // namespace hexl
}  // namespace intel
