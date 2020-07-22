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
#include "ntt/avx512_util.hpp"
#include "ntt/ntt.hpp"
#include "ntt/number-theory.hpp"

namespace intel {
namespace ntt {

#ifdef NTT_HAS_AVX512IFMA
template void NTT::ForwardTransformToBitReverseAVX512<52>(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements);
#endif

template void NTT::ForwardTransformToBitReverseAVX512<64>(
    const IntType degree, const IntType mod,
    const IntType* root_of_unity_powers,
    const IntType* precon_root_of_unity_powers, IntType* elements);

}  // namespace ntt
}  // namespace intel
