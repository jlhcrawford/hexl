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

#include <tuple>
#include <utility>

namespace intel {
namespace lattice {

inline size_t hash_combine(size_t lhs, size_t rhs) {
  lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  return lhs;
}

struct hash_tuple {
  template <class T1, class T2, class T3>
  size_t operator()(const std::tuple<T1, T2, T3>& p) const {
    auto hash1 = std::hash<T1>{}(std::get<0>(p));
    auto hash2 = std::hash<T2>{}(std::get<1>(p));
    auto hash3 = std::hash<T2>{}(std::get<2>(p));
    return hash_combine(hash_combine(hash1, hash2), hash3);
  }
};

}  // namespace lattice
}  // namespace intel
