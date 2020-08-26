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

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test/test_util.hpp"
#include "util/avx512_util.hpp"

namespace intel {
namespace lattice {

#ifdef LATTICE_HAS_AVX512IFMA
TEST(AVX512, _mm512_il_mulhi_epi52) {
  __m512i w = _mm512_set_epi64(90774764920991, 90774764920991, 90774764920991,
                               90774764920991, 90774764920991, 90774764920991,
                               90774764920991, 90774764920991);
  __m512i y = _mm512_set_epi64(424, 635, 757, 457, 280, 624, 353, 496);

  __m512i expected = _mm512_set_epi64(8, 12, 15, 9, 5, 12, 7, 9);

  __m512i z = _mm512_il_mulhi_epi<52>(w, y);

  ASSERT_TRUE(Equals(z, expected));
}
#endif

TEST(AVX512, _mm512_il_add_epu64) {
  {
    __m512i a = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i b = _mm512_set_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i expected_out = _mm512_set_epi64(8, 10, 12, 14, 16, 18, 20, 22);
    __m512i expected_carry = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0);

    __m512i c;
    __m512i carry = _mm512_il_add_epu<64>(a, b, &c);

    CheckEqual(carry, expected_carry);
    CheckEqual(c, expected_out);
  }

  // Overflow
  {
    __m512i a = _mm512_set_epi64(1UL << 32,         //
                                 1UL << 63,         //
                                 (1UL << 63) + 1,   //
                                 (1UL << 63) + 10,  //
                                 0,                 //
                                 0,                 //
                                 0,                 //
                                 0);
    __m512i b = _mm512_set_epi64(1UL << 32,         //
                                 1UL << 63,         //
                                 1UL << 63,         //
                                 (1UL << 63) + 17,  //
                                 0,                 //
                                 0,                 //
                                 0,                 //
                                 0);
    __m512i expected_out = _mm512_set_epi64(1UL << 33,  //
                                            0,          //
                                            1,          //
                                            27,         //
                                            0,          //
                                            0,          //
                                            0,          //
                                            0);
    __m512i expected_carry = _mm512_set_epi64(0, 1, 1, 1, 0, 0, 0, 0);

    __m512i c;
    __m512i carry = _mm512_il_add_epu<64>(a, b, &c);

    CheckEqual(carry, expected_carry);
    CheckEqual(c, expected_out);
  }
}

TEST(AVX512, _mm512_il_cmplt_epu64) {
  // Small
  {
    uint64_t match_value = 10;
    __m512i a = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i b = _mm512_set_epi64(0, 1, 1, 0, 5, 6, 100, 100);
    __m512i expected_out = _mm512_set_epi64(
        0, 0, 0, 0, match_value, match_value, match_value, match_value);

    __m512i c = _mm512_il_cmplt_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }

  // Large
  {
    uint64_t match_value = 13;
    __m512i a = _mm512_set_epi64(1UL << 32,         //
                                 1UL << 63,         //
                                 (1UL << 63) + 1,   //
                                 (1UL << 63) + 10,  //
                                 0,                 //
                                 0,                 //
                                 0,                 //
                                 0);
    __m512i b = _mm512_set_epi64(1UL << 32,         //
                                 1UL << 63,         //
                                 1UL << 63,         //
                                 (1UL << 63) + 17,  //
                                 0,                 //
                                 0,                 //
                                 0,                 //
                                 0);
    __m512i expected_out = _mm512_set_epi64(0, 0, 0, match_value, 0, 0, 0, 0);

    __m512i c = _mm512_il_cmplt_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }
}

TEST(AVX512, _mm512_il_cmpge_epu64) {
  // Small
  {
    uint64_t match_value = 10;
    __m512i a = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i b = _mm512_set_epi64(0, 1, 1, 0, 5, 6, 100, 100);
    __m512i expected_out = _mm512_set_epi64(
        match_value, match_value, match_value, match_value, 0, 0, 0, 0);

    __m512i c = _mm512_il_cmpge_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }

  // Large
  {
    uint64_t match_value = 13;
    __m512i a = _mm512_set_epi64(1UL << 32,         //
                                 1UL << 63,         //
                                 (1UL << 63) + 1,   //
                                 (1UL << 63) + 10,  //
                                 0,                 //
                                 0,                 //
                                 0,                 //
                                 0);
    __m512i b = _mm512_set_epi64(1UL << 32,         //
                                 1UL << 63,         //
                                 1UL << 63,         //
                                 (1UL << 63) + 17,  //
                                 0,                 //
                                 0,                 //
                                 0,                 //
                                 0);
    __m512i expected_out =
        _mm512_set_epi64(match_value, match_value, match_value, 0, match_value,
                         match_value, match_value, match_value);

    __m512i c = _mm512_il_cmpge_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }
}

TEST(AVX512, _mm512_il_mod_epi64) {
  // Small
  {
    __m512i a = _mm512_set_epi64(0, 2, 4, 6, 8, 10, 11, 12);
    __m512i mods = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
    __m512i expected_out = _mm512_set_epi64(0, 0, 1, 2, 3, 4, 4, 4);

    __m512i c = _mm512_il_mod_epi64(a, mods);

    CheckEqual(c, expected_out);
  }

  // Large
  {
    __m512i a = _mm512_set_epi64(1UL << 32,         //
                                 1UL << 63,         //
                                 (1UL << 63) + 1,   //
                                 (1UL << 63) + 10,  //
                                 0,                 //
                                 0,                 //
                                 0,                 //
                                 0);
    __m512i mods = _mm512_set_epi64(1UL << 32,         //
                                    1UL << 63,         //
                                    1UL << 63,         //
                                    (1UL << 63) + 17,  //
                                    0,                 //
                                    0,                 //
                                    0,                 //
                                    0);
    __m512i expected_out =
        _mm512_set_epi64(0, 0, 1, (1UL << 63) + 10, 0, 0, 0, 0);

    __m512i c = _mm512_il_mod_epi64(a, mods);

    CheckEqual(c, expected_out);
  }
}

}  // namespace lattice
}  // namespace intel
