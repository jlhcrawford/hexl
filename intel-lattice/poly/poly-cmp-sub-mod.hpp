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

#include "number-theory/number-theory.hpp"

namespace intel {
namespace lattice {

// @brief Computes element-wise: if (operand1 > cmp) operand1 = (operand1 -
// diff) mod modulus
// @param operand1 Vector of elements to compare; stores result
// @param cmp Scalar to compare against
// @param diff Scalar to subtract by
// @param modulus Modulus to redce by
// @param n Number of elements in operand1
void CmpGtSubMod(uint64_t* operand1, uint64_t cmp, uint64_t diff,
                 uint64_t modulus, uint64_t n);

// @brief Computes element-wise: if (operand1 > cmp) operand1 = (operand1 -
// diff) mod modulus
// @param operand1 Vector of elements to compare; stores result
// @param cmp Scalar to compare against
// @param diff Scalar to subtract by
// @param modulus Modulus to redce by
// @param n Number of elements in operand
void CmpGtSubModNative(uint64_t* operand1, uint64_t cmp, uint64_t diff,
                       uint64_t modulus, uint64_t n);

#ifdef LATTICE_HAS_AVX512F
// @brief Computes element-wise: if (operand1 > cmp) operand1 = (operand1 -
// diff) mod modulus
// @param operand1 Vector of elements to compare; stores result
// @param cmp Scalar to compare against
// @param diff Scalar to subtract by
// @param modulus Modulus to redce by
// @param n Number of elements in operand1
void CmpGtSubModAVX512(uint64_t* operand1, uint64_t cmp, uint64_t diff,
                       uint64_t modulus, uint64_t n);

#endif

}  // namespace lattice
}  // namespace intel
