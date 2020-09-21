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
