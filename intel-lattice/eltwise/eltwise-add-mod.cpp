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

#include "eltwise/eltwise-add-mod.hpp"

#include "eltwise/eltwise-add-mod-internal.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-add-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

// Algorithm 1 of https://hal.archives-ouvertes.fr/hal-01215845/document
void EltwiseAddModNative(uint64_t* operand1, const uint64_t* operand2,
                         const uint64_t n, const uint64_t modulus) {
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");
  LATTICE_CHECK_BOUNDS(operand1, n, modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus);

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = 0; i < n; ++i) {
    uint64_t sum = *operand1 + *operand2;
    if (sum > modulus) {
      *operand1 = sum - modulus;
    } else {
      *operand1 = sum;
    }

    ++operand1;
    ++operand2;
  }
}

void EltwiseAddMod(uint64_t* operand1, const uint64_t* operand2,
                   const uint64_t n, const uint64_t modulus) {
#ifdef LATTICE_HAS_AVX512DQ
  EltwiseAddModAVX512(operand1, operand2, n, modulus);
  return;
#endif

  IVLOG(3, "Calling EltwiseAddModNative");
  EltwiseAddModNative(operand1, operand2, n, modulus);
}

}  // namespace lattice
}  // namespace intel
