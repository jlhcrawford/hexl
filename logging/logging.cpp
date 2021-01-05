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

#include "logging/logging.hpp"

#include <gflags/gflags.h>

#include <string>

INITIALIZE_EASYLOGGINGPP;

DEFINE_bool(logtofile, false, "enable logfile output");
DEFINE_int32(v, 0, "enable verbose (DEBUG) logging");
DEFINE_string(vmodule, "", "enable verbose (DEBUG) logging");
DEFINE_string(logconf, "", "enable logging configuration from file");

namespace {
const char logDirPrefix[] = "logs/";
}  // namespace

el::Configurations LogConfigurationFromFlags(const std::string& app_name) {
  el::Configurations conf;
  if (FLAGS_logconf.empty()) {
    conf.setToDefault();
  } else {
    conf = el::Configurations(FLAGS_logconf.c_str());
  }
  if (!FLAGS_logtofile) {
    conf.set(el::Level::Global, el::ConfigurationType::ToFile, "false");
  } else {
    conf.set(el::Level::Global, el::ConfigurationType::Filename,
             std::string(logDirPrefix) + app_name + ".log");
  }
  if (!FLAGS_v) {
    conf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  } else {
    el::Loggers::setVerboseLevel(FLAGS_v);
  }
  if (!FLAGS_vmodule.empty()) {
    el::Loggers::setVModules(FLAGS_vmodule.c_str());
  }

  return conf;
}
