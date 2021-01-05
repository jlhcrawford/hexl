// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020-2021 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// ******************************************************************************

#pragma once

#include <stdint.h>

#include "intel-lattice/util/util.hpp"

namespace intel {
namespace lattice {

/// @brief Computes element-wise conditional moduluar subtraction.
/// @param[in,out] operand1 Vector of elements to compare; stores result
/// @param[in] cmp Comparison function
/// @param[in] bound Scalar to compare against
/// @param[in] diff Scalar to subtract by
/// @param[in] modulus Modulus to reduce by
/// @param[in] n Number of elements in \p operand1
/// @details Computes \p operand1[i] = (\p cmp(\p operand1, \p bound)) ? (\p
/// operand1 - \p diff) mod \p modulus : \p operand1 for all i=0, ..., n-1
void EltwiseCmpSubMod(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                      uint64_t diff, uint64_t modulus, uint64_t n);

}  // namespace lattice
}  // namespace intel
