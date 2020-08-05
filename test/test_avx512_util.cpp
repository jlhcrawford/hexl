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

#include <immintrin.h>

#include <chrono>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "util/avx512_util.hpp"

namespace intel {
namespace ntt {

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

#ifdef LATTICE_HAS_AVX512IFMA
TEST(AVX512, avx512_multiply_uint64_hi52) {
  {
    __m512i w = _mm512_set_epi64(90774764920991, 90774764920991, 90774764920991,
                                 90774764920991, 90774764920991, 90774764920991,
                                 90774764920991, 90774764920991);
    __m512i y = _mm512_set_epi64(424, 635, 757, 457, 280, 624, 353, 496);

    __m512i expected = _mm512_set_epi64(8, 12, 15, 9, 5, 12, 7, 9);

    __m512i z = avx512_multiply_uint64_hi<52>(w, y);

    ASSERT_TRUE(Equals(z, expected));
  }
}
#endif

}  // namespace ntt
}  // namespace intel
