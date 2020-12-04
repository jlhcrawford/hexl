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

#include <utility>

#include "intel-lattice/util/util.hpp"

namespace intel {
namespace lattice {

inline bool Compare(CMPINT cmp, uint64_t lhs, uint64_t rhs) {
  switch (cmp) {
    case CMPINT::EQ:
      return lhs == rhs;
    case CMPINT::LT:
      return lhs < rhs;
      break;
    case CMPINT::LE:
      return lhs <= rhs;
      break;
    case CMPINT::FALSE:
      return false;
      break;
    case CMPINT::NE:
      return lhs != rhs;
      break;
    case CMPINT::NLT:
      return lhs >= rhs;
      break;
    case CMPINT::NLE:
      return lhs > rhs;
    case CMPINT::TRUE:
      return true;
    default:
      return true;
  }
}

}  // namespace lattice
}  // namespace intel
