// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020-2021 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

#include "ntt/inv-ntt-avx512.hpp"

#include "intel-lattice/ntt/ntt.hpp"
#include "ntt/ntt-internal.hpp"

namespace intel {
namespace lattice {

#ifdef LATTICE_HAS_AVX512IFMA
template void
InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
    const uint64_t degree, const uint64_t mod,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements,
    uint64_t input_mod_factor, uint64_t output_mod_factor);
#endif

template void
InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_default_shift_bits>(
    const uint64_t degree, const uint64_t mod,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements,
    uint64_t input_mod_factor, uint64_t output_mod_factor);

}  // namespace lattice
}  // namespace intel
