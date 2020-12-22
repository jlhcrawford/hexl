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

__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;

// Create logging/debug macros with no run-time overhead unless LATTICE_DEBUG is
// enabled
#ifdef LATTICE_DEBUG
#include "logging/logging.hpp"

#define LATTICE_CHECK(cond, expr)                                    \
  if (!(cond)) {                                                     \
    LOG(ERROR) << expr << " in fuction: " << __FUNCTION__            \
               << " in file: " __FILE__ << " at line: " << __LINE__; \
    throw std::runtime_error("Error. Check log output");             \
  }

#define LATTICE_CHECK_BOUNDS3(arg, n, bound)                               \
  for (size_t i = 0; i < n; ++i) {                                         \
    LATTICE_CHECK((arg)[i] < bound, "Arg[" << i << "] = " << (arg)[i]      \
                                           << " exceeds bound " << bound); \
  }

#define LATTICE_CHECK_BOUNDS4(arg, n, bound, expr) \
  for (size_t i = 0; i < n; ++i) {                 \
    LATTICE_CHECK((arg)[i] < bound, expr);         \
  }

#else

#define LATTICE_CHECK(cond, expr) \
  {}
#define LATTICE_CHECK_BOUNDS3(arg, n, bound) \
  {}

#define LATTICE_CHECK_BOUNDS4(arg, n, bound, expr) \
  {}
#endif

// Dispatch LATTICE_CHECK_BOUNDS to proper number of arguments
#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define LATTICE_CHECK_BOUNDS(...)                                      \
  GET_MACRO(__VA_ARGS__, LATTICE_CHECK_BOUNDS4, LATTICE_CHECK_BOUNDS3) \
  (__VA_ARGS__)

// }  // namespace lattice
// }  // namespace intel
