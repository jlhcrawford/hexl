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

#pragma once

#include <stdint.h>

namespace intel {
namespace lattice {

/// @brief Multiplies in-place two vectors elementwise with modular reduction
/// @param[in,out] operand1 Vector of elements to multiply; stores result
/// @param[in] operand2 Vector of elements to multiply
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction
/// @details Computes \p operand1[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1
void EltwiseMultMod(uint64_t* operand1, const uint64_t* operand2,
                    const uint64_t n, const uint64_t modulus);

/// @brief Multiplies two vectors elementwise with modular reduction
/// @param[in] result Result of element-wise multiplication
/// @param[in] operand1 Vector of elements to multiply; stores result
/// @param[in] operand2 Vector of elements to multiply
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction
/// @details Computes \p result[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1
void EltwiseMultMod(uint64_t* result, const uint64_t* operand1,
                    const uint64_t* operand2, const uint64_t n,
                    const uint64_t modulus);

}  // namespace lattice
}  // namespace intel
