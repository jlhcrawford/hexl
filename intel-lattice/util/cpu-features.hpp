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

#pragma once

#include <stdbool.h>

#include "cpu_features/cpuinfo_x86.h"

namespace intel {
namespace lattice {

static const cpu_features::X86Features features =
    cpu_features::GetX86Info().features;
static const bool has_avx512_ifma = features.avx512ifma;
static const bool has_avx512_dq =
    features.avx512f && features.avx512dq && features.avx512vl;

}  // namespace lattice
}  // namespace intel
