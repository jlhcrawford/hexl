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
