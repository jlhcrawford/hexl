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

#include "eltwise/eltwise-fma.hpp"

#include "eltwise/eltwise-fma-internal.hpp"
#include "logging/logging.hpp"
#include "number-theory/number-theory.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "eltwise/eltwise-fma-avx512.hpp"
#endif

namespace intel {
namespace lattice {

void EltwiseFMAMod(const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3,
                   uint64_t* out, uint64_t n, uint64_t modulus) {
  LATTICE_CHECK(modulus != 0, "Require modulus != 0");

#ifdef LATTICE_HAS_AVX512IFMA
  if (has_avx512_ifma && modulus < (1UL << 52)) {
    IVLOG(3, "Calling 52-bit EltwiseFMAModAVX512");
    MultiplyFactor mf(arg2, 52, modulus);
    EltwiseFMAModAVX512<52>(arg1, arg2, arg3, out, mf.BarrettFactor(), n,
                            modulus);
    return;
  }
#endif
#ifdef LATTICE_HAS_AVX512DQ
  if (has_avx512_dq) {
    IVLOG(3, "Calling 64-bit EltwiseFMAModAVX512");
    MultiplyFactor mf(arg2, 64, modulus);
    EltwiseFMAModAVX512<64>(arg1, arg2, arg3, out, mf.BarrettFactor(), n,
                            modulus);
  }
  return;
#endif
  IVLOG(3, "Calling EltwiseFMAModNative");
  EltwiseFMAModNative(arg1, arg2, arg3, out, n, modulus);
}

void EltwiseFMAModNative(const uint64_t* arg1, uint64_t arg2,
                         const uint64_t* arg3, uint64_t* out, uint64_t n,
                         uint64_t modulus) {
  MultiplyFactor mf(arg2, 64, modulus);
  if (arg3) {
    for (size_t i = 0; i < n; ++i) {
      *out = MultiplyMod(*arg1, arg2, mf.BarrettFactor(), modulus);
      *out++ = AddUIntMod(*arg1++, *arg3++, modulus);
    }
    return;
  }

  // arg3 == nullptr
  for (size_t i = 0; i < n; ++i) {
    *out++ = MultiplyMod(*arg1++, arg2, mf.BarrettFactor(), modulus);
  }
}

}  // namespace lattice
}  // namespace intel
