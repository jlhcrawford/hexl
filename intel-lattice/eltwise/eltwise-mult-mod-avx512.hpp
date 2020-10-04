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

namespace intel {
namespace lattice {

// Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document
void EltwiseMultModAVX512Int(uint64_t* operand1, const uint64_t* operand2,
                             uint64_t n, const uint64_t modulus);

// From Function 18, page 19 of https://arxiv.org/pdf/1407.3383.pdf
// See also Algorithm 2/3 of
// https://hal.archives-ouvertes.fr/hal-02552673/document
void EltwiseMultModAVX512Float(uint64_t* operand1, const uint64_t* operand2,
                               uint64_t n, const uint64_t modulus);

}  // namespace lattice
}  // namespace intel
