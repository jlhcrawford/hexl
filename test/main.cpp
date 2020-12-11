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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "logging/logging.hpp"

int main(int argc, char** argv) {
  gflags::SetUsageMessage(argv[0]);
  START_EASYLOGGINGPP(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}
