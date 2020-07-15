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

// TODO(fboemer) Enable if needed
// #define ELPP_THREAD_SAFE
#define ELPP_CUSTOM_COUT std::cerr
#define ELPP_STL_LOGGING
#define ELPP_LOG_STD_ARRAY
#define ELPP_LOG_UNORDERED_MAP
#define ELPP_LOG_UNORDERED_SET
#define ELPP_NO_LOG_TO_FILE
#define ELPP_DISABLE_DEFAULT_CRASH_HANDLING
#define ELPP_WINSOCK2

#include <easylogging++.h>

#include <algorithm>
#include <complex>
#include <vector>

inline MAKE_LOGGABLE(std::vector<std::complex<double>>, vector, os) {
  size_t size = std::min(vector.size(), 10UL);

  for (size_t i = 0; i < size; ++i) {
    os << vector[i];
    if (i < size - 1) {
      os << ", ";
    }
  }
  if (size < vector.size()) {
    os << "...";
  }
  return os;
}

#define IVLOG(N, rest)   \
  do {                   \
    if (VLOG_IS_ON(N)) { \
      VLOG(N) << rest;   \
    }                    \
  } while (0);
