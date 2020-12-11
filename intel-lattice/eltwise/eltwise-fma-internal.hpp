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

#include "number-theory/number-theory.hpp"

namespace intel {
namespace lattice {

void EltwiseFMAModNative(const uint64_t* arg1, uint64_t arg2,
                         const uint64_t* arg3, uint64_t* out, uint64_t n,
                         uint64_t modulus);

}  // namespace lattice
}  // namespace intel
