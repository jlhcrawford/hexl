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

// Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document
void EltwiseMultModAVX512Int(uint64_t* operand1, const uint64_t* operand2,
                             uint64_t n, const uint64_t modulus);

void EltwiseMultModAVX512Int(uint64_t* result, const uint64_t* operand1,
                             const uint64_t* operand2, uint64_t n,
                             const uint64_t modulus);

// From Function 18, page 19 of https://arxiv.org/pdf/1407.3383.pdf
// See also Algorithm 2/3 of
// https://hal.archives-ouvertes.fr/hal-02552673/document
void EltwiseMultModAVX512Float(uint64_t* operand1, const uint64_t* operand2,
                               uint64_t n, const uint64_t modulus);

void EltwiseMultModAVX512Float(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               const uint64_t modulus);

}  // namespace lattice
}  // namespace intel
