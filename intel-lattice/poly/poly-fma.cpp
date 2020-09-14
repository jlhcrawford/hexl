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

#include "poly/poly-fma-internal.hpp"

#ifdef LATTICE_HAS_AVX512F
#include "poly/poly-fma-avx512.hpp"
#endif

#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

void FMAModScalar(const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3,
                  uint64_t* out, uint64_t n, uint64_t modulus) {
#ifdef LATTICE_HAS_AVX512IFMA
  // TODO(fboemer): check behavior around 50-52 bits
  if (modulus < (1UL << 52) && (n % 8 == 0)) {
    IVLOG(3, "Calling 52-bit FMAModScalarAVX512");
    MultiplyFactor mf(arg2, 52, modulus);
    FMAModScalarAVX512<52>(arg1, arg2, arg3, out, mf.BarrettFactor(), n,
                           modulus);
    return;
  }
#endif
#ifdef LATTICE_HAS_AVX512F
  if (n % 8 == 0) {
    IVLOG(3, "Calling 64-bit FMAModScalarAVX512");
    MultiplyFactor mf(arg2, 64, modulus);
    FMAModScalarAVX512<64>(arg1, arg2, arg3, out, mf.BarrettFactor(), n,
                           modulus);
    return;
  }
#endif

  IVLOG(3, "Calling 64-bit default FMAModScalar64");
  MultiplyFactor mf(arg2, 64, modulus);
  FMAModScalar64(arg1, arg2, arg3, out, mf.BarrettFactor(), n, modulus);
}

void FMAModScalar64(const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3,
                    uint64_t* out, uint64_t arg2_barr, uint64_t n,
                    uint64_t modulus) {
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
