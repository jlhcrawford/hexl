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

namespace intel {
namespace lattice {

/// \enum CMPINT
/// \brief Represents binary operations between two boolean values
enum class CMPINT {
  EQ = 0,     ///< Equal
  LT = 1,     ///< Less than
  LE = 2,     ///< Less than or equal
  FALSE = 3,  ///< False
  NE = 4,     ///< Not equal
  NLT = 5,    ///< Not less than
  NLE = 6,    ///< Not less than or equal
  TRUE = 7    ///< True
};

/// @brief Returns the logical negation of a binary operation
/// @param[in] cmp The binary operation to negate
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
