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

#pragma once

#include <stdint.h>

namespace intel {
namespace lattice {

// @brief Adds two vectors elementwise with modular reduction
// @param operand1 Vector of elements to add; stores result
// @param operand2 Vector of elements to add
// @param n Number of elements in each vector
// @param modulus Modulus with which to perform modular reduction
void EltwiseAddMod(uint64_t* operand1, const uint64_t* operand2,
                   const uint64_t n, const uint64_t modulus);

}  // namespace lattice
}  // namespace intel
