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

#include "eltwise/eltwise-fma-avx512.hpp"

namespace intel {
namespace lattice {

#ifdef LATTICE_HAS_AVX512IFMA
template void EltwiseFMAModAVX512<52>(const uint64_t* arg1, const uint64_t arg2,
                                      const uint64_t* arg3, uint64_t* out,
                                      const uint64_t b_barr, const uint64_t n,
                                      const uint64_t modulus);
#endif

template void EltwiseFMAModAVX512<64>(const uint64_t* arg1, const uint64_t arg2,
                                      const uint64_t* arg3, uint64_t* out,
                                      const uint64_t b_barr, const uint64_t n,
                                      const uint64_t modulus);

}  // namespace lattice
}  // namespace intel
