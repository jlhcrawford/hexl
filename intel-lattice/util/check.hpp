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

__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;

// Create logging/debug macros with no run-time overhead unless LATTICE_DEBUG is
// enabled
#ifdef LATTICE_DEBUG
#include "logging/logging.hpp"

#define LATTICE_CHECK(cond, expr)                        \
  if (!(cond)) {                                         \
    LOG(ERROR) << expr;                                  \
    throw std::runtime_error("Error. Check log output"); \
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
