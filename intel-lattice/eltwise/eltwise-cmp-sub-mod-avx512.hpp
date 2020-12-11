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

// @brief Computes element-wise: if (operand1 > cmp) operand1 = (operand1 -
// diff) mod modulus
// @param operand1 Vector of elements to compare; stores result
// @param cmp Scalar to compare against
// @param diff Scalar to subtract by
// @param modulus Modulus to redce by
// @param n Number of elements in operand1
void EltwiseCmpSubModAVX512(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                            uint64_t diff, uint64_t modulus, uint64_t n);

}  // namespace lattice
}  // namespace intel
