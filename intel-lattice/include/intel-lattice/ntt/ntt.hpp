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

#include <stdint.h>

#include <memory>
#include <vector>

namespace intel {
namespace lattice {

/// @brief Performs negacyclic forward and inverse number-theoretic transform
/// (NTT), commonly used in RLWE cryptography.
/// @details The number-theoretic transform (NTT) specializes the discrete
/// Fourier transform (DFT) to the finite field \f$ \mathbb{Z}_p / (X^N + 1)\f$.
class NTT {
 public:
  /// Initializes an empty NTT object
  NTT();

  /// Destructs the NTT object
  ~NTT();

  /// Initializes an NTT object with degree degree and prime modulus p.
  /// @param[in] degree a.k.a. N. Size of the NTT transform. Must be a power of
  /// 2
  /// @param[in] p Prime modulus. Must satisfy \f$ p == 1 \mod 2N \f$
  /// @brief Performs pre-computation necessary for forward and inverse
  /// transforms
  NTT(uint64_t degree, uint64_t p);

  /// @brief Initializes an NTT object with degree \p N and modulus
  /// \p p.
  /// @param[in] degree a.k.a. N. Size of the NTT transform. Must be a power of
  /// 2
  /// @param[in] p Prime modulus. Must satisfy \f$ p == 1 \mod 2N \f$
  /// @param[in] root_of_unity 2N'th root of unity in \f$ \mathbb{Z_p} \f$.
  /// @brief Initializes an NTT object with degree \p degree and prime modulus
  /// \p p.
  /// @details  Performs pre-computation necessary for forward and inverse
  /// transforms
  NTT(uint64_t degree, uint64_t p, uint64_t root_of_unity);

  /// @brief Computes in-place forward NTT. Results are bit-reversed.
  /// @param[in, out] elements Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p elements are in [0,
  /// input_mod_factor * p). Must be 2 or 4.
  /// @param[in] output_mod_factor Returns output \p elements in [0,
  /// output_mod_factor * p). Must be 1 or 4.
  void ComputeForward(uint64_t* elements, uint64_t input_mod_factor = 2,
                      uint64_t output_mod_factor = 1);

  /// @brief Compute forward NTT. Results are bit-reversed.
  /// @param[in] elements Data on which to compute the NTT
  /// @param[out] result Stores the result
  /// @param[in] input_mod_factor Assume input \p elements are in [0,
  /// input_mod_factor * p). Must be 2 or 4.
  /// @param[in] output_mod_factor Returns output \p elements in [0,
  /// output_mod_factor * p). Must be 1 or 4.
  void ComputeForward(const uint64_t* elements, uint64_t* result,
                      uint64_t input_mod_factor = 2,
                      uint64_t output_mod_factor = 1);

  /// @brief Compute in-place inverse NTT. Results are bit-reversed.
  /// @param[in,out] elements Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p elements are in [0,
  /// input_mod_factor * p). Must be 1 or 2.
  /// @param[in] output_mod_factor Returns output \p elements in [0,
  /// output_mod_factor * p). Must be 1 or 2.
  void ComputeInverse(uint64_t* elements, uint64_t input_mod_factor = 1,
                      uint64_t output_mod_factor = 1);

  /// Compute inverse NTT. Results are bit-reversed.
  /// @param[in] elements Data on which to compute the NTT
  /// @param[out] result Stores the result
  /// @param[in] input_mod_factor Assume input \p elements are in [0,
  /// input_mod_factor * p). Must be 1 or 2.
  /// @param[in] output_mod_factor Returns output \p elements in [0,
  /// output_mod_factor * p). Must be 1 or 2.
  void ComputeInverse(const uint64_t* elements, uint64_t* result,
                      uint64_t input_mod_factor = 1,
                      uint64_t output_mod_factor = 1);

  class NTTImpl;  /// Class implementing the NTT

 private:
  std::shared_ptr<NTTImpl> m_impl;
};

}  // namespace lattice
}  // namespace intel
