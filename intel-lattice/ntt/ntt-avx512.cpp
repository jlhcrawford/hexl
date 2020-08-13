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

#include "ntt/ntt-avx512.hpp"

#include <immintrin.h>

#include <iostream>

#include "logging/logging.hpp"
#include "ntt/ntt.hpp"
#include "util/avx512_util.hpp"

namespace intel {
namespace lattice {

#ifdef LATTICE_HAS_AVX512IFMA
template void NTT::ForwardTransformToBitReverseAVX512<NTT::s_ifma_shift_bits>(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements);
#endif

template void
NTT::ForwardTransformToBitReverseAVX512<NTT::s_default_shift_bits>(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements);

#ifdef LATTICE_HAS_AVX512IFMA
template void NTT::InverseTransformToBitReverseAVX512<NTT::s_ifma_shift_bits>(
    const IntType degree, const IntType mod,
    const IntType* inv_root_of_unity_powers,
    const IntType* inv_scaled_root_of_unity_powers, IntType* elements);
#endif

template void
NTT::InverseTransformToBitReverseAVX512<NTT::s_default_shift_bits>(
    const IntType degree, const IntType mod,
    const IntType* inv_root_of_unity_powers,
    const IntType* inv_scaled_root_of_unity_powers, IntType* elements);

}  // namespace lattice
}  // namespace intel
