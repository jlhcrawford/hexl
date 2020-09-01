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

#pragma once

#include <functional>

#include "number-theory/number-theory.hpp"

namespace intel {
namespace lattice {

// @brief Computes fused multiply-add (arg1 * arg2 + arg3) mod modulus
// element-wise, broadcasting scalars to vectors.
// @param arg1 Vector to multiply
// @param arg2 Scalar to multiply
// @param arg3 Vector to add
// @param out Stores the output
// @param n Number of elements in each vector
// @param modulus Modulus with which to perform modular reduction
void FMAModScalar(const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3,
                  uint64_t* out, uint64_t n, uint64_t modulus);

// @brief Computes fused multiply-add (arg1 * arg2 + arg3) mod modulus
// element-wise, broadcasting scalars to vectors
// @param arg1 Vector to multiply
// @param arg2 Scalar to multiply
// @param arg3 Vector to add
// @param out Stores the outpud
// @param b_barr floor(2**64 / modulus)
// @param n Number of elements in each vector
// @param modulus Modulus with which to perform modular reduction
void FMAModScalar64(const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3,
                    uint64_t* out, uint64_t b_barr, uint64_t n,
                    uint64_t modulus);

inline void FMAModScalar64(const uint64_t* arg1, uint64_t arg2,
                           const uint64_t* arg3, uint64_t* out, uint64_t n,
                           uint64_t modulus) {
  MultiplyFactor mf(arg2, 64, modulus);
  FMAModScalar64(arg1, arg2, arg3, out, mf.BarrettFactor(), n, modulus);
}

#ifdef LATTICE_HAS_AVX512F

template <int BitShift>
void FMAModScalarAVX512(const uint64_t* arg1, uint64_t arg2,
                        const uint64_t* arg3, uint64_t* out, uint64_t b_barr,
                        uint64_t n, uint64_t modulus);

template <int BitShift>
inline void FMAModScalarAVX512(const uint64_t* arg1, uint64_t arg2,
                               const uint64_t* arg3, uint64_t* out, uint64_t n,
                               uint64_t modulus) {
  MultiplyFactor mf(arg2, BitShift, modulus);

  FMAModScalarAVX512<BitShift>(arg1, arg2, arg3, out, mf.BarrettFactor(), n,
                               modulus);
}
#endif

}  // namespace lattice
}  // namespace intel
