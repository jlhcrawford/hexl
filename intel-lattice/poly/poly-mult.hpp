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

#include <limits>
#include <vector>

#include "logging/logging.hpp"
#include "util/check.hpp"

namespace intel {
namespace lattice {

// @brief Multiplies two vectors elementwise with modular reduction
// @param operand1 Vector of elements to multiply; stores result
// @param operand2 Vector of elements to multiply
// @param n Number of elements in each vector
// @param barrett_hi High 64 bits of Barrett precomputation floor(2^128 /
// modulus)
// @param barrett_lo Low 64 bits of Barrett precomputation floor(2^128 /
// modulus)
// @param modulus Modulus with which to perform modular reduction
void MultiplyModInPlace(uint64_t* operand1, const uint64_t* operand2,
                        const uint64_t n, const uint64_t barrett_hi,
                        const uint64_t barrett_low, const uint64_t modulus);

}  // namespace lattice
}  // namespace intel
