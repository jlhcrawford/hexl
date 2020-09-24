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

#include "eltwise/eltwise-mult-mod.hpp"

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-mult-mod-avx512.hpp"
#endif

namespace intel {
namespace lattice {

void EltwiseMultModNative(uint64_t* operand1, const uint64_t* operand2,
                          const uint64_t n, const uint64_t barr_hi,
                          const uint64_t barr_lo, const uint64_t modulus) {
  LATTICE_CHECK_BOUNDS(operand1, n, modulus);
  LATTICE_CHECK_BOUNDS(operand2, n, modulus);

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
  for (size_t i = 0; i < n; ++i) {
    uint64_t prod_hi, prod_lo, rnd1_hi, rnd2_hi, rnd2_lo, rnd3_hi, rnd3_lo,
        floor_lo, floor_hi, result;

    // Multiply inputs
    MultiplyUInt64(*operand1, *operand2, &prod_hi, &prod_lo);

    // Reduces product using base 2^BitShift Barrett reduction
    // Each | indicates BitShift-bit chunks
    //
    //                        | barr_hi | barr_lo |
    //      X                 | prod_hi | prod_lo |
    // --------------------------------------------
    //                        | prod_lo x barr_lo | // Round 1
    // +            | prod_lo x barr_hi |           // Round 2
    // +            | prod_hi x barr_lo |           // Round 3
    // +  | barr_hi x prod_hi |                     // Round 4
    // --------------------------------------------
    //              |floor_hi | floor_lo|
    //               \-------/
    //                   \- The only BitShift-bit chunk we care about: vfloor_hi

    // Round 1
    rnd1_hi = MultiplyUInt64Hi<64>(prod_lo, barr_lo);
    // Round 2
    MultiplyUInt64(prod_lo, barr_hi, &rnd2_hi, &rnd2_lo);
    floor_hi = rnd2_hi + AddUInt64(rnd2_lo, rnd1_hi, &floor_lo);

    // Round 3
    MultiplyUInt64(prod_hi, barr_lo, &rnd3_hi, &rnd3_lo);
    floor_hi += rnd3_hi + AddUInt64(floor_lo, rnd3_lo, &floor_lo);

    // Round 4
    floor_hi += prod_hi * barr_hi;

    // Barrett subtraction
    result = prod_lo - floor_hi * modulus;

    // Conditional subtraction
    *operand1 = result - (modulus & static_cast<uint64_t>(-static_cast<int64_t>(
                                        result >= modulus)));

    ++operand1;
    ++operand2;
  }
}

void EltwiseMultMod(uint64_t* operand1, const uint64_t* operand2,
                    const uint64_t n, const uint64_t modulus) {
#ifdef LATTICE_HAS_AVX512DQ
  if (has_avx512_dq && modulus < (1UL << 50)) {
    IVLOG(3, "Calling EltwiseMultModAVX512Float");
    EltwiseMultModAVX512Float(operand1, operand2, n, modulus);
    return;
  }
#endif

#ifdef LATTICE_HAS_AVX512IFMA
  if (has_avx512_ifma && modulus < (1UL << 52)) {
    IVLOG(3, "Calling EltwiseMultModAVX512Int<52>");
    EltwiseMultModAVX512Int<52>(operand1, operand2, n, modulus);
    return;
  }
#endif

  IVLOG(3, "Calling EltwiseMultModNative");
  EltwiseMultModNative(operand1, operand2, n, modulus);
}

}  // namespace lattice
}  // namespace intel
