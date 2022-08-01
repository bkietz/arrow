// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <string>

#include "arrow/util/visibility.h"

namespace arrow {
namespace util {

ARROW_EXPORT bool StacktraceSupported();

ARROW_EXPORT std::string PrintStacktrace(int skip);

/// Constructing a StacktraceOverride will replace the default line of a stacktrace
/// for the frame in which it is declared. This can be used to mark significant frames
/// with a custom string (for example line="worker thread root frame") or to omit frames
/// which would otherwise clutter the trace (with line="").
class ARROW_EXPORT StacktraceOverride {
 public:
  explicit StacktraceOverride(std::string line = "");
  ~StacktraceOverride();
};

}  // namespace util
}  // namespace arrow
