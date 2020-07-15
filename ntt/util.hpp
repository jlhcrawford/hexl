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

#include "logging/logging.hpp"

__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;

// TODO(fboemer): better error logging. It should not evaluate expr if in debug
// mode

#ifdef NTT_DEBUG
#define NTT_CHECK(cond, expr)                            \
  if (!(cond)) {                                         \
    LOG(ERROR) << expr;                                  \
    throw std::runtime_error("Error. Check log output"); \
  }
#else
#define NTT_CHECK(cond, expr) \
  {}
#endif
