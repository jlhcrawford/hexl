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

#include "poly/poly-fma-avx512.hpp"

namespace intel {
namespace lattice {

#ifdef LATTICE_HAS_AVX512IFMA
template void FMAModScalarAVX512<52>(const uint64_t* arg1, const uint64_t arg2,
                                     const uint64_t* arg3, uint64_t* out,
                                     const uint64_t b_barr, const uint64_t n,
                                     const uint64_t modulus);
#endif

template void FMAModScalarAVX512<64>(const uint64_t* arg1, const uint64_t arg2,
                                     const uint64_t* arg3, uint64_t* out,
                                     const uint64_t b_barr, const uint64_t n,
                                     const uint64_t modulus);

}  // namespace lattice
}  // namespace intel
