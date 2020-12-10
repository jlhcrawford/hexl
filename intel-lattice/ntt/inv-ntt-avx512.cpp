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

#include "ntt/inv-ntt-avx512.hpp"

#include "ntt/ntt-internal.hpp"
#include "ntt/ntt.hpp"

namespace intel {
namespace lattice {

#ifdef LATTICE_HAS_AVX512IFMA
template void
InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_ifma_shift_bits>(
    const uint64_t degree, const uint64_t mod,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements,
    bool full_reduce);
#endif

template void
InverseTransformFromBitReverseAVX512<NTT::NTTImpl::s_default_shift_bits>(
    const uint64_t degree, const uint64_t mod,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t* elements,
    bool full_reduce);

}  // namespace lattice
}  // namespace intel
