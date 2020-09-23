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

namespace intel {
namespace lattice {

enum class CMPINT {
  EQ = 0,     // Equal
  LT = 1,     // Less than
  LE = 2,     // Less than or equal
  FALSE = 3,  // False
  NE = 4,     // Not equal
  NLT = 5,    // Not less than
  NLE = 6,    // Not less than or equal
  TRUE = 7    // True
};

inline CMPINT Not(CMPINT cmp) {
  switch (cmp) {
    case CMPINT::EQ:
      return CMPINT::NE;
    case CMPINT::LT:
      return CMPINT::NLT;
    case CMPINT::LE:
      return CMPINT::NLE;
    case CMPINT::FALSE:
      return CMPINT::TRUE;
    case CMPINT::NE:
      return CMPINT::EQ;
    case CMPINT::NLT:
      return CMPINT::LT;
    case CMPINT::NLE:
      return CMPINT::LE;
    case CMPINT::TRUE:
      return CMPINT::FALSE;
    default:
      return CMPINT::FALSE;
  }
}

}  // namespace lattice
}  // namespace intel
