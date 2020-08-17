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

#include <limits>
#include <vector>

#include "logging/logging.hpp"
#include "util/check.hpp"

#ifdef LATTICE_HAS_AVX512F
#include <immintrin.h>

#include "util/avx512_util.hpp"
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
inline void AssertEqual(const std::vector<uint64_t>& x,
                        const std::vector<uint64_t>& y) {
  ASSERT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], y[i]);
  }
}

#ifdef LATTICE_HAS_AVX512F
inline void CheckEqual(const __m512i a, const __m512i b) {
  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);
  CheckEqual(as, bs);
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
