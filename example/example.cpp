// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020-2021 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

#include <cstdint>
#include <iostream>
#include <vector>

#include "intel-lattice/intel-lattice.hpp"

bool CheckEqual(const std::vector<uint64_t>& x,
                const std::vector<uint64_t>& y) {
  if (x.size() != y.size()) {
    std::cout << "Not equal in size\n";
    return false;
  }
  uint64_t N = x.size();
  bool is_match = true;
  for (size_t i = 0; i < N; ++i) {
    if (x[i] != y[i]) {
      std::cout << "Not equal at index " << i << "\n";
      is_match = false;
    }
  }
  return is_match;
}

void ExampleEltwiseCmpAdd() {
  std::cout << "Running ExampleEltwiseCmpAdd...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t cmp = 3;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 9, 10, 11, 12, 13};

  intel::lattice::EltwiseCmpAdd(op1.data(), intel::lattice::CMPINT::NLE, cmp,
                                diff, op1.size());

  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseCmpAdd\n";
}

void ExampleEltwiseCmpSubMod() {
  std::cout << "Running ExampleEltwiseCmpSubMod...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7};
  uint64_t bound = 4;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2};

  uint64_t modulus = 10;

  intel::lattice::EltwiseCmpSubMod(op1.data(), intel::lattice::CMPINT::NLE,
                                   bound, diff, modulus, op1.size());
  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseCmpSubMod\n";
}

void ExampleEltwiseFMAMod() {
  std::cout << "Running ExampleEltwiseFMAMod...\n";

  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;

  intel::lattice::EltwiseFMAMod(arg1.data(), arg2, nullptr, arg1.data(),
                                arg1.size(), modulus);
  CheckEqual(arg1, exp_out);
  std::cout << "Done running ExampleEltwiseFMAMod\n";
}

void ExampleEltwiseMultMod() {
  std::cout << "Running ExampleEltwiseMultMod...\n";

  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

  intel::lattice::EltwiseMultMod(op1.data(), op2.data(), op1.size(), modulus);
  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseMultMod\n";
}

void ExampleNTT() {
  std::cout << "Running ExampleNTT...\n";

  uint64_t prime = 769;
  uint64_t N = 8;
  std::vector<uint64_t> arg{1, 2, 3, 4, 5, 6, 7, 8};
  auto exp_out = arg;
  intel::lattice::NTT ntt(N, prime);

  ntt.ComputeForward(arg.data());
  ntt.ComputeInverse(arg.data());

  CheckEqual(arg, exp_out);
  std::cout << "Done running ExampleNTT\n";
}

int main() {
  ExampleEltwiseCmpAdd();
  ExampleEltwiseCmpSubMod();
  ExampleEltwiseFMAMod();
  ExampleEltwiseMultMod();
  ExampleNTT();

  return 0;
}
