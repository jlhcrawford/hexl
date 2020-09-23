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

#include <stdint.h>

#include "intel-lattice/util/util.hpp"

namespace intel {
namespace lattice {

// @brief Computes element-wise: if (cmp(operand1, bound)) operand1 += diff
// @param operand1 Vector of elements to compare; stores result
// @param cmp Comparison operation
// @param bound Scalar to compare against
// @param diff Scalar to increment by
// @param n Number of elements in operand1
void EltwiseCmpAdd(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                   uint64_t diff, uint64_t n);

}  // namespace lattice
}  // namespace intel
