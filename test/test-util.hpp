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

#include <limits>
#include <vector>

#include "logging/logging.hpp"
#include "util/check.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include <immintrin.h>

#include "util/avx512-util.hpp"
#endif

namespace intel {
namespace lattice {

// Checks whether x == y.
inline void CheckEqual(const std::vector<uint64_t>& x,
                       const std::vector<uint64_t>& y) {
  EXPECT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(x[i], y[i]);
  }
}

// Asserts x == y
template <typename T>
inline void AssertEqual(const std::vector<T>& x, const std::vector<T>& y) {
  ASSERT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], y[i]);
  }
}

#ifdef LATTICE_HAS_AVX512DQ
inline void CheckEqual(const __m512i a, const __m512i b) {
  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);
  CheckEqual(as, bs);
}

inline void AssertEqual(const __m512i a, const __m512i b) {
  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);
  AssertEqual(as, bs);
}

// Returns true iff a == b
// Logs an error if a != b
inline bool Equals(__m512i a, __m512i b) {
  bool match = true;

  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);

  for (size_t i = 0; i < 8; ++i) {
    if (as[i] != bs[i]) {
      LOG(ERROR) << "Mismatch at index " << i << ": "
                 << "a[" << i << "] = " << as[i] << ", b[" << i
                 << "] = " << bs[i];
      match = false;
    }
  }
  return match;
}
#endif

}  // namespace lattice
}  // namespace intel
