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

// Wrap IVLOG with LATTICE_DEBUG; this ensures no logging overhead in release
// mode
#ifdef LATTICE_DEBUG
#define IVLOG(N, rest)   \
  do {                   \
    if (VLOG_IS_ON(N)) { \
      VLOG(N) << rest;   \
    }                    \
  } while (0);
#else
#define IVLOG(N, rest) \
  {}
#endif
