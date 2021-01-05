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

#include "intel-lattice/util/util.hpp"

namespace intel {
namespace lattice {

/// @brief Computes element-wise conditional addition.
/// @param[in,out] operand1 Vector of elements to compare; stores result
/// @param[in] cmp Comparison operation
/// @param[in] bound Scalar to compare against
/// @param[in] diff Scalar to conditionally add
/// @param[in] n Number of elements in \p operand1
/// @details Computes operand1[i] = cmp(operand1[i], bound) ? operand1[i] +
/// diff : operand1[i] for all \f$i=0, ..., n-1\f$.
void EltwiseCmpAdd(uint64_t* operand1, CMPINT cmp, uint64_t bound,
                   uint64_t diff, uint64_t n);

}  // namespace lattice
}  // namespace intel
