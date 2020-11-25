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

#include <new>
#include <vector>

namespace intel {
namespace lattice {

template <class T>
struct AlignedAllocator {
  typedef T value_type;
  AlignedAllocator() noexcept {}
  template <class U, int V>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    // TODO(fboemer): Allow configurable alignment
    return static_cast<T*>(
        ::operator new (n * sizeof(T), std::align_val_t{64}));
  }
  void deallocate(T* p, std::size_t n) {
    (void)n;  // Avoid unused variable
    ::delete (p);
  }
};

template <class T, class U>
constexpr bool operator==(const AlignedAllocator<T>&,
                          const AlignedAllocator<U>&) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(const AlignedAllocator<T>&,
                          const AlignedAllocator<U>&) noexcept {
  return false;
}

template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

}  // namespace lattice
}  // namespace intel
