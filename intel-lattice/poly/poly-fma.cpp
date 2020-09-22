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

#include "poly/poly-fma.hpp"

#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "poly/poly-fma-internal.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "poly/poly-fma-avx512.hpp"
#endif

namespace intel {
namespace lattice {

void FMAModScalar(const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3,
                  uint64_t* out, uint64_t n, uint64_t modulus) {
#ifdef LATTICE_HAS_AVX512IFMA
  if (has_avx512_ifma && modulus < (1UL << 52)) {
    IVLOG(3, "Calling 52-bit FMAModScalarAVX512");
    MultiplyFactor mf(arg2, 52, modulus);
    FMAModScalarAVX512<52>(arg1, arg2, arg3, out, mf.BarrettFactor(), n,
                           modulus);
    return;
  }
#endif
#ifdef LATTICE_HAS_AVX512DQ
  if (has_avx512_dq) {
    IVLOG(3, "Calling 64-bit FMAModScalarAVX512");
    MultiplyFactor mf(arg2, 64, modulus);
    FMAModScalarAVX512<64>(arg1, arg2, arg3, out, mf.BarrettFactor(), n,
                           modulus);
  }
  return;
#endif
  IVLOG(3, "Calling FMAModScalarNative");
  MultiplyFactor mf(arg2, 64, modulus);
  FMAModScalarNative(arg1, arg2, arg3, out, mf.BarrettFactor(), n, modulus);
}

void FMAModScalarNative(const uint64_t* arg1, uint64_t arg2,
                        const uint64_t* arg3, uint64_t* out, uint64_t arg2_barr,
                        uint64_t n, uint64_t modulus) {
  if (arg3) {
    for (size_t i = 0; i < n; ++i) {
      *out = MultiplyMod(*arg1, arg2, arg2_barr, modulus);
      *out++ = AddUIntMod(*arg1++, *arg3++, modulus);
    }
    return;
  }

  // arg3 == nullptr
  for (size_t i = 0; i < n; ++i) {
    *out++ = MultiplyMod(*arg1++, arg2, arg2_barr, modulus);
  }
}

}  // namespace lattice
}  // namespace intel
