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

#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "ntt/ntt-internal.hpp"
#include "ntt/ntt.hpp"
#include "number-theory/number-theory.hpp"
#include "test/test-util.hpp"

#ifdef LATTICE_HAS_AVX512DQ
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#endif

namespace intel {
namespace lattice {

TEST(NTT, Powers) {
  uint64_t modulus = 0xffffffffffc0001ULL;
  {
    uint64_t N = 2;
    NTT::NTTImpl ntt_impl(N, modulus);

    ASSERT_EQ(1ULL, ntt_impl.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt_impl.GetRootOfUnityPower(1));
  }

  {
    uint64_t N = 4;
    NTT::NTTImpl ntt_impl(N, modulus);

    ASSERT_EQ(1ULL, ntt_impl.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt_impl.GetRootOfUnityPower(1));
    ASSERT_EQ(178930308976060547ULL, ntt_impl.GetRootOfUnityPower(2));
    ASSERT_EQ(748001537669050592ULL, ntt_impl.GetRootOfUnityPower(3));
  }
}

TEST(NTT, root_of_unity) {
  uint64_t p = 769;
  uint64_t N = 8;
  std::vector<uint64_t> input{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> input2 = input;

  uint64_t root_of_unity = MinimalPrimitiveRoot(2 * N, p);

  NTT ntt1(N, p);
  NTT ntt2(N, p, root_of_unity);

  ntt1.ComputeForward(input.data());
  ntt2.ComputeForward(input2.data());

  CheckEqual(input, input2);
}

TEST(NTTImpl, root_of_unity) {
  uint64_t p = 769;
  uint64_t N = 8;

  NTT::NTTImpl ntt_impl(N, p);

  EXPECT_EQ(ntt_impl.GetMinimalRootOfUnity(), MinimalPrimitiveRoot(2 * N, p));
  EXPECT_EQ(ntt_impl.GetDegree(), N);
  EXPECT_EQ(ntt_impl.GetInvRootOfUnityPower(0),
            ntt_impl.GetInvRootOfUnityPowers()[0]);
}

// Parameters = (degree, prime, input, expected_output)
class NTTAPITest
    : public ::testing::TestWithParam<std::tuple<
          uint64_t, uint64_t, std::vector<uint64_t>, std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test different parts of the API
TEST_P(NTTAPITest, Fwd) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t prime = std::get<1>(GetParam());

  std::vector<uint64_t> input = std::get<2>(GetParam());
  std::vector<uint64_t> input2 = input;
  std::vector<uint64_t> input3 = input;
  std::vector<uint64_t> input4 = input;
  std::vector<uint64_t> input5 = input;
  std::vector<uint64_t> exp_output = std::get<3>(GetParam());

  NTT::NTTImpl ntt_impl(N, prime);
  NTT ntt(N, prime);
  ntt.ComputeForward(input.data());

  // Compute reference
  ReferenceForwardTransformToBitReverse(
      N, prime, ntt_impl.GetRootOfUnityPowers().data(), input2.data());

  CheckEqual(input, exp_output);
  CheckEqual(input2, exp_output);

  // Test round-trip
  ntt.ComputeInverse(input.data());
  CheckEqual(input, input3);

  // Test out-of-place forward
  ntt.ComputeForward(input3.data(), input4.data());
  CheckEqual(input4, input2);

  // Test out-of-place inverse
  ntt.ComputeInverse(input4.data(), input5.data());
  CheckEqual(input5, input3);
}

INSTANTIATE_TEST_SUITE_P(
    NTTAPITest, NTTAPITest,
    ::testing::Values(
        std::make_tuple(2, 281474976710897, std::vector<uint64_t>{0, 0},
                        std::vector<uint64_t>{0, 0}),
        std::make_tuple(2, 0xffffffffffc0001ULL, std::vector<uint64_t>{0, 0},
                        std::vector<uint64_t>{0, 0}),
        std::make_tuple(2, 281474976710897, std::vector<uint64_t>{1, 0},
                        std::vector<uint64_t>{1, 1}),
        std::make_tuple(2, 281474976710897, std::vector<uint64_t>{1, 1},
                        std::vector<uint64_t>{19842761023586, 261632215687313}),
        std::make_tuple(2, 0xffffffffffc0001ULL, std::vector<uint64_t>{1, 1},
                        std::vector<uint64_t>{288794978602139553,
                                              864126526004445282}),
        std::make_tuple(4, 113, std::vector<uint64_t>{94, 109, 11, 18},
                        std::vector<uint64_t>{82, 2, 81, 98}),
        std::make_tuple(4, 281474976710897,
                        std::vector<uint64_t>{281474976710765, 49,
                                              281474976710643, 275},
                        std::vector<uint64_t>{12006376116355, 216492038983166,
                                              272441922811203, 62009615510542}),
        std::make_tuple(4, 113, std::vector<uint64_t>{59, 50, 98, 50},
                        std::vector<uint64_t>{1, 2, 3, 4}),
        std::make_tuple(4, 73, std::vector<uint64_t>{2, 1, 1, 1},
                        std::vector<uint64_t>{17, 41, 36, 60}),
        std::make_tuple(4, 16417, std::vector<uint64_t>{31, 21, 15, 34},
                        std::vector<uint64_t>{1611, 14407, 14082, 2858}),
        std::make_tuple(4, 4194353,
                        std::vector<uint64_t>{4127, 9647, 1987, 5410},
                        std::vector<uint64_t>{1478161, 3359347, 222964,
                                              3344742}),
        std::make_tuple(8, 4194353,
                        std::vector<uint64_t>{1, 0, 0, 0, 0, 0, 0, 0},
                        std::vector<uint64_t>{1, 1, 1, 1, 1, 1, 1, 1}),
        std::make_tuple(8, 4194353,
                        std::vector<uint64_t>{1, 1, 0, 0, 0, 0, 0, 0},
                        std::vector<uint64_t>{132171, 4062184, 2675172, 1519183,
                                              462763, 3731592, 1824324,
                                              2370031}),
        std::make_tuple(
            32, 769,
            std::vector<uint64_t>{401, 203, 221, 352, 487, 151, 405, 356,
                                  343, 424, 635, 757, 457, 280, 624, 353,
                                  496, 353, 624, 280, 457, 757, 635, 424,
                                  343, 356, 405, 151, 487, 352, 221, 203},
            std::vector<uint64_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32})));

class NTTZerosTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Parameters = (degree, prime_bits)
TEST_P(NTTZerosTest, Zeros) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t prime_bits = std::get<1>(GetParam());
  uint64_t prime = GeneratePrimes(1, prime_bits, N)[0];

  std::vector<uint64_t> input(N, 0);
  std::vector<uint64_t> exp_output(N, 0);

  NTT ntt(N, prime);
  ntt.ComputeForward(input.data());

  CheckEqual(input, exp_output);
}

INSTANTIATE_TEST_SUITE_P(
    NTTZerosTest, NTTZerosTest,
    ::testing::Values(
        std::make_tuple(1 << 1, 30), std::make_tuple(1 << 2, 30),
        std::make_tuple(1 << 3, 30), std::make_tuple(1 << 4, 35),
        std::make_tuple(1 << 5, 35), std::make_tuple(1 << 6, 35),
        std::make_tuple(1 << 7, 40), std::make_tuple(1 << 8, 40),
        std::make_tuple(1 << 9, 40), std::make_tuple(1 << 10, 45),
        std::make_tuple(1 << 11, 45), std::make_tuple(1 << 12, 45),
        std::make_tuple(1 << 13, 50), std::make_tuple(1 << 14, 50),
        std::make_tuple(1 << 15, 50), std::make_tuple(1 << 16, 55),
        std::make_tuple(1 << 17, 55)));

#ifdef LATTICE_HAS_AVX512IFMA
class NTTPrimesTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test primes around 50 bits to check IFMA behavior
// Parameters = (degree, prime_bits)
TEST_P(NTTPrimesTest, IFMAPrimes) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t prime_bits = std::get<1>(GetParam());
  uint64_t prime = GeneratePrimes(1, prime_bits, N)[0];

  std::vector<uint64_t> input64(N, 0);
  for (size_t i = 0; i < N; ++i) {
    input64[i] = i % prime;
  }
  std::vector<uint64_t> input_ifma = input64;

  std::vector<uint64_t> exp_output(N, 0);

  // Compute reference
  NTT::NTTImpl ntt64(N, prime);
  ReferenceForwardTransformToBitReverse(
      N, prime, ntt64.GetRootOfUnityPowers().data(), input64.data());

  // Compute with s_ifma_shift_bits-bit bit shift
  NTT::NTTImpl ntt_ifma(N, prime);
  ForwardTransformToBitReverseAVX512<52>(
      N, ntt_ifma.GetModulus(), ntt_ifma.GetRootOfUnityPowers().data(),
      ntt_ifma.GetPrecon52RootOfUnityPowers().data(), input_ifma.data());

  CheckEqual(input64, input_ifma);
}

INSTANTIATE_TEST_SUITE_P(NTTPrimesTest, NTTPrimesTest,
                         ::testing::Values(std::make_tuple(1 << 4, 48),
                                           std::make_tuple(1 << 5, 49),
                                           std::make_tuple(1 << 6, 49),
                                           std::make_tuple(1 << 7, 49),
                                           std::make_tuple(1 << 8, 49)));
#endif

#ifdef LATTICE_HAS_AVX512DQ
// Checks AVX512 and native forward NTT implementations match
TEST(NTT, FwdNTT_AVX512) {
  uint64_t N = 64;
  uint64_t prime = GeneratePrimes(1, 55, N)[0];

  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

  for (size_t trial = 0; trial < 1; ++trial) {
    std::vector<std::uint64_t> input(N, 0);
    for (size_t i = 0; i < N; ++i) {
      input[i] = distrib(gen);
    }
    std::vector<std::uint64_t> input2 = input;
    LOG(INFO) << "inpout " << input;

    NTT::NTTImpl ntt_impl(N, prime);
    ForwardTransformToBitReverse64(
        N, prime, ntt_impl.GetRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64RootOfUnityPowers().data(), input2.data());

    LOG(INFO) << "output native " << input;

    ForwardTransformToBitReverseAVX512<64>(
        N, prime, ntt_impl.GetRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64RootOfUnityPowers().data(), input.data());

    ASSERT_EQ(input, input2);
  }
}

// Checks AVX512 and native InvNTT implementations match
TEST(NTT, InvNTT_AVX512) {
  uint64_t N = 512;
  uint64_t prime = GeneratePrimes(1, 55, N)[0];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

  for (size_t trial = 0; trial < 200; ++trial) {
    std::vector<std::uint64_t> input(N, 0);
    for (size_t i = 0; i < N; ++i) {
      input[i] = distrib(gen);
    }
    std::vector<std::uint64_t> input2 = input;

    NTT::NTTImpl ntt_impl(N, prime);
    InverseTransformFromBitReverseAVX512<64>(
        N, ntt_impl.GetModulus(), ntt_impl.GetInvRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64InvRootOfUnityPowers().data(), input.data());

    InverseTransformFromBitReverse64(
        N, prime, ntt_impl.GetInvRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64InvRootOfUnityPowers().data(), input2.data());

    ASSERT_EQ(input, input2);
  }
}
#endif

}  // namespace lattice
}  // namespace intel
