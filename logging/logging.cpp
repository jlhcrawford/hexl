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
