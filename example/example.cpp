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

#include <cstdint>
#include <iostream>
#include <vector>

#include "intel-lattice/intel-lattice.hpp"

void CheckEqual(const std::vector<uint64_t>& x,
                const std::vector<uint64_t>& y) {
  if (x.size() != y.size()) {
    std::cout << "Not equal in size\n";
  }
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    if (x[i] != y[i]) {
      std::cout << "Not equal at index " << i << "\n";
    }
  }
}

void ExampleCmpGtAdd() {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t cmp = 3;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 9, 10, 11, 12, 13};

  intel::lattice::CmpGtAdd(op1.data(), cmp, diff, op1.size());

  CheckEqual(op1, exp_out);
}

void ExampleCmpSubMod() {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7};
  uint64_t cmp = 4;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2};

  uint64_t modulus = 10;

  intel::lattice::CmpGtSubMod(op1.data(), cmp, diff, modulus, op1.size());
  CheckEqual(op1, exp_out);
}

void ExampleFMAModScalar() {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;

  intel::lattice::FMAModScalar(arg1.data(), arg2, nullptr, arg1.data(),
                               arg1.size(), modulus);
  CheckEqual(arg1, exp_out);
}

void ExampleMultiplyMod() {
  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

  intel::lattice::MultiplyModInPlace(op1.data(), op2.data(), op1.size(),
                                     modulus);
  CheckEqual(op1, exp_out);
}

void ExampleNTT() {
  uint64_t prime = 769;
  uint64_t N = 8;
  std::vector<uint64_t> arg{1, 2, 3, 4, 5, 6, 7, 8};
  auto exp_out = arg;
  intel::lattice::NTT ntt(N, prime);

  ntt.ComputeForward(arg.data());
  ntt.ComputeInverse(arg.data());
  CheckEqual(arg, exp_out);
}

int main() {
  ExampleCmpGtAdd();
  ExampleCmpSubMod();
  ExampleFMAModScalar();
  ExampleMultiplyMod();

  return 0;
}
