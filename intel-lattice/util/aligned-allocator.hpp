// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2020 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

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
    ::operator delete (p, std::align_val_t{64});
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
