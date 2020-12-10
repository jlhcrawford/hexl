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

#include <stdint.h>

#include <memory>
#include <vector>

namespace intel {
namespace lattice {

// Performs negacyclic NTT and inverse NTT
class NTT {
 public:
  // Initializes an empty NTT object
  NTT();
  ~NTT();

  // Initializes an NTT object with degree degree and prime modulus p.
  // @param degree Size of the NTT transform, a.k.a N. Must be a power of 2
  // @param p Prime modulus. Must satisfy p == 1 mod 2N
  // @brief Performs pre-computation necessary for forward and inverse
  // transforms
  NTT(uint64_t degree, uint64_t p);

  // Initializes an NTT object with degree degree and prime modulus p.
  // @param degree Size of the NTT transform, a.k.a N. Must be a power of 2
  // @param p Prime modulus. Must satisfy p == 1 mod 2N
  // @param root_of_unity 2N'th root of unity in F_p
  // @brief Performs pre-computation necessary for forward and inverse
  // transforms
  NTT(uint64_t degree, uint64_t p, uint64_t root_of_unity);

  // Compute in-place NTT.
  // Results are bit-reversed
  // @param full_reduce If true, results are in [0,p); otherwise, outputs are in
  // [0,4*p)
  void ComputeForward(uint64_t* elements, bool full_reduce = true);

  // Compute NTT
  // Results are bit-reversed
  // @param elements Data on which to compute the NTT
  // @param result Stores the result
  // @param full_reduce If true, results are in [0,p); otherwise, outputs are in
  // [0,4*p)
  void ComputeForward(const uint64_t* elements, uint64_t* result,
                      bool full_reduce = true);

  // Compute in-place inverse NTT.
  // Results are bit-reversed
  // @param full_reduce If true, results are in [0,p); otherwise, outputs are in
  // [0,2*p)
  void ComputeInverse(uint64_t* elements, bool full_reduce = true);

  // Compute inverse NTT
  // Results are bit-reversed
  // @param elements Data on which to compute the NTT
  // @param result Stores the result
  // @param full_reduce If true, results are in [0,p); otherwise, outputs are in
  // [0,2*p)
  void ComputeInverse(const uint64_t* elements, uint64_t* result,
                      bool full_reduce = true);

  class NTTImpl;

 private:
  std::shared_ptr<NTTImpl> m_impl;
};

}  // namespace lattice
}  // namespace intel
