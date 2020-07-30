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
#include "ntt/ntt.hpp"
#include "ntt/number-theory.hpp"

namespace intel {
namespace ntt {

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
    NTT ntt(N, modulus);

    ASSERT_EQ(1ULL, ntt.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt.GetRootOfUnityPower(1));
  }

  {
    uint64_t N = 4;
    NTT ntt(N, modulus);

    ASSERT_EQ(1ULL, ntt.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt.GetRootOfUnityPower(1));
    ASSERT_EQ(178930308976060547ULL, ntt.GetRootOfUnityPower(2));
    ASSERT_EQ(748001537669050592ULL, ntt.GetRootOfUnityPower(3));
  }
}

TEST(NTT, 2a_48bit) {
  std::vector<uint64_t> input{0, 0};
  uint64_t prime = 281474976710897;
  std::vector<uint64_t> exp_output{0, 0};
  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2a_60bit) {
  std::vector<uint64_t> input{0, 0};
  uint64_t prime = 0xffffffffffc0001ULL;
  std::vector<uint64_t> exp_output{0, 0};
  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2b_48bit) {
  std::vector<uint64_t> input{1, 0};
  uint64_t prime = 281474976710897;
  std::vector<uint64_t> exp_output{1, 1};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2b_49bit) {
  std::vector<uint64_t> input{1, 0};
  std::vector<uint64_t> exp_output{1, 1};

  size_t N = input.size();
  uint64_t prime = GeneratePrimes(1, 49, N)[0];
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2b_50bit) {
  std::vector<uint64_t> input{1, 0};
  std::vector<uint64_t> exp_output{1, 1};

  size_t N = input.size();
  uint64_t prime = GeneratePrimes(1, 50, N)[0];
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2b_60bit) {
  std::vector<uint64_t> input{1, 0};
  std::vector<uint64_t> exp_output{1, 1};

  size_t N = input.size();
  uint64_t prime = GeneratePrimes(1, 60, N)[0];

  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2b_60bit_native) {
  std::vector<uint64_t> input{1, 0};
  std::vector<uint64_t> exp_output{1, 1};

  size_t N = input.size();
  uint64_t prime = GeneratePrimes(1, 60, N)[0];
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse64(
      N, prime, ntt.GetRootOfUnityPowers().data(),
      ntt.GetPreconRootOfUnityPowers().data(), input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2c_48bit) {
  std::vector<uint64_t> input{1, 1};
  uint64_t prime = 281474976710897;
  std::vector<uint64_t> exp_output{19842761023586, 261632215687313};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2c_60bit) {
  std::vector<uint64_t> input{1, 1};
  uint64_t prime = 0xffffffffffc0001ULL;
  std::vector<uint64_t> exp_output{288794978602139553, 864126526004445282};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 2c_60bit_native) {
  std::vector<uint64_t> input{1, 1};
  uint64_t prime = 0xffffffffffc0001ULL;
  std::vector<uint64_t> exp_output{288794978602139553, 864126526004445282};
  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse64(
      N, prime, ntt.GetRootOfUnityPowers().data(),
      ntt.GetPreconRootOfUnityPowers().data(), input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4a) {
  uint64_t prime = 113;
  std::vector<uint64_t> input{94, 109, 11, 18};
  std::vector<uint64_t> exp_output{82, 2, 81, 98};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4b) {
  std::vector<uint64_t> input{281474976710765, 49, 281474976710643, 275};
  uint64_t prime = 281474976710897;
  std::vector<uint64_t> exp_output{12006376116355, 216492038983166,
                                   272441922811203, 62009615510542};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4c) {
  std::vector<uint64_t> input{59, 50, 98, 50};
  uint64_t prime = 113;
  std::vector<uint64_t> exp_output{1, 2, 3, 4};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4d) {
  std::vector<uint64_t> input{2, 1, 1, 1};
  uint64_t prime = 73;
  std::vector<uint64_t> exp_output{17, 41, 36, 60};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4e) {
  std::vector<uint64_t> input{31, 21, 15, 34};
  uint64_t prime = 16417;
  std::vector<uint64_t> exp_output{1611, 14407, 14082, 2858};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4f) {
  std::vector<uint64_t> input{0, 0, 0, 0};
  uint64_t prime = 4194353;
  std::vector<uint64_t> exp_output{0, 0, 0, 0};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 4g) {
  std::vector<uint64_t> input{4127, 9647, 1987, 5410};
  uint64_t prime = 4194353;
  std::vector<uint64_t> exp_output{1478161, 3359347, 222964, 3344742};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

TEST(NTT, 32a) {
  std::vector<uint64_t> input{401, 203, 221, 352, 487, 151, 405, 356,
                              343, 424, 635, 757, 457, 280, 624, 353,
                              496, 353, 624, 280, 457, 757, 635, 424,
                              343, 356, 405, 151, 487, 352, 221, 203};
  uint64_t prime = 769;
  std::vector<uint64_t> exp_output{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

  size_t N = input.size();
  NTT ntt(N, prime);
  ntt.ForwardTransformToBitReverse(input.data());

  CheckNTTResults(input, exp_output);
}

}  // namespace ntt
}  // namespace intel
