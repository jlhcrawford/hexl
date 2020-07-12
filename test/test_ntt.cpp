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

#include <chrono>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "ntt.hpp"
#include "number-theory.hpp"

namespace intel::ntt {

// Checks whether x == y.
void CheckNTTResults(const std::vector<uint64_t>& x,
                     const std::vector<uint64_t>& y) {
  IVLOG(5, "Checking NTT results");
  IVLOG(5, "x " << x);
  IVLOG(5, "y " << y);
  EXPECT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  EXPECT_TRUE(IsPowerOfTwo(N));
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(x[i], y[i]);
  }
}

TEST(NTT, Powers) {
  uint64_t modulus = 0xffffffffffc0001ULL;
  {
    uint64_t N = 2;
    NTT ntt(modulus, N);

    ASSERT_EQ(1ULL, ntt.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt.GetRootOfUnityPower(1));
  }

  {
    uint64_t N = 4;
    NTT ntt(modulus, 4);

    ASSERT_EQ(1ULL, ntt.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt.GetRootOfUnityPower(1));
    ASSERT_EQ(178930308976060547ULL, ntt.GetRootOfUnityPower(2));
    ASSERT_EQ(748001537669050592ULL, ntt.GetRootOfUnityPower(3));
  }
}

TEST(NTT, 2a) {
  std::vector<uint64_t> input{0, 0};
  uint64_t prime = 0xffffffffffc0001ULL;
  std::vector<uint64_t> exp_output{0, 0};
  size_t N = input.size();
  NTT ntt(prime, N);
  ntt.ForwardTransformToBitReverse(&input);

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2b) {
  std::vector<uint64_t> input{1, 0};
  uint64_t prime = 0xffffffffffc0001ULL;
  std::vector<uint64_t> exp_output{1, 1};

  size_t N = input.size();
  NTT ntt(prime, N);
  ntt.ForwardTransformToBitReverse(&input);

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2c) {
  std::vector<uint64_t> input{1, 1};
  uint64_t prime = 0xffffffffffc0001ULL;
  std::vector<uint64_t> exp_output{288794978602139553ULL,
                                   864126526004445282ULL};

  size_t N = input.size();
  NTT ntt(prime, N);
  ntt.ForwardTransformToBitReverse(&input);

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4a) {
  uint64_t prime = 113;
  std::vector<uint64_t> input{94, 109, 11, 18};
  std::vector<uint64_t> exp_output{82, 2, 81, 98};

  size_t N = input.size();
  NTT ntt(prime, N);
  ntt.ForwardTransformToBitReverse(&input);

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4b) {
  std::vector<uint64_t> input{281474976710765, 49, 281474976710643, 275};
  uint64_t prime = 281474976710897;
  std::vector<uint64_t> exp_output{12006376116355, 216492038983166,
                                   272441922811203, 62009615510542};

  size_t N = input.size();
  NTT ntt(prime, N);
  ntt.ForwardTransformToBitReverse(&input);

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4c) {
  std::vector<uint64_t> input{59, 50, 98, 50};
  uint64_t prime = 113;
  std::vector<uint64_t> exp_output{1, 2, 3, 4};

  size_t N = input.size();
  NTT ntt(prime, N);
  ntt.ForwardTransformToBitReverse(&input);

  CheckNTTResults(input, exp_output);
}

}  // namespace intel::ntt
