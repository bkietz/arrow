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

#include "arrow/util/stacktrace.h"

#ifdef ARROW_WITH_BACKTRACE

#include <backtrace.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#endif
#include <cassert>

#include "arrow/util/optional.h"
#include "arrow/util/string.h"

namespace arrow {
namespace util {

struct OverrideLine {
  std::string override_line;
  uintptr_t parent_pc;
  StacktraceOverride* stack_ptr;
};

static thread_local std::vector<OverrideLine> kOverrideLines;

bool StacktraceSupported() { return true; }

enum { kContinue = 0, kStop = 1 };

template <typename Visitor>
void VisitStacktrace(int skip, Visitor visitor) {
  static thread_local auto state = backtrace_create_state(
      // TODO use ArrowLog::app_name_
      /*executable_name=*/nullptr,
      /*threaded=*/false,
      /*error_callback=*/
      [](void*, const char* msg, int errnum) {
        std::cerr << "backtrace_create_state failed #" << errnum << " " << msg
                  << std::endl;
        std::abort();
      },
      nullptr);

  backtrace_full(
      state, skip + 1,
      [](void* visitor_raw, uintptr_t pc, const char* filename, int lineno,
         const char* mangled_name) -> int {
        auto& visitor = *static_cast<Visitor*>(visitor_raw);
        return visitor(pc, filename, lineno, mangled_name);
      },
      /*error_callback=*/
      [](void*, const char* msg, int errnum) {
        std::cerr << "backtrace_full encountered error #" << errnum << " " << msg
                  << std::endl;
        std::abort();
      },
      &visitor);
}

std::string PrintStacktrace(int skip) {
  std::vector<std::string> lines;

  VisitStacktrace(skip + 1, [&](uintptr_t pc, const char* filename, int lineno,
                                const char* mangled_name) {
    // check for an overridden line
    for (auto it = kOverrideLines.begin(); it != kOverrideLines.end(); ++it) {
      if (it->parent_pc != pc) continue;

      if (lines.empty()) break;

      if (it->override_line == "") {
        lines.pop_back();
      } else {
        lines.back() = it->override_line;
      }
      break;
    }

    if (filename == nullptr) {
      // no usable debug information for this frame
      return kContinue;
    }

    std::string line;

    int status;
    if (auto function = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status)) {
      assert(status == 0);
      line += function;
      free(function);
    } else {
      line += mangled_name;
    }

    line += " in " + std::string(filename) + ":" + std::to_string(lineno);

    lines.push_back(std::move(line));
    return kContinue;
  });

  return internal::JoinStrings(lines, "\n");
}

StacktraceOverride::StacktraceOverride(std::string line) {
  VisitStacktrace(/*skip=*/2, [&](uintptr_t pc, ...) {
    kOverrideLines.push_back({std::move(line), pc, this});
    return kStop;
  });
}

StacktraceOverride::~StacktraceOverride() {
  auto it = std::remove_if(kOverrideLines.begin(), kOverrideLines.end(),
                           [&](const OverrideLine& l) { return l.stack_ptr == this; });
  kOverrideLines.erase(it, kOverrideLines.end());
}

}  // namespace util
}  // namespace arrow

#else

namespace arrow {
namespace util {

bool StacktraceSupported() { return false; }
std::string PrintStacktrace(int skip) { return ""; }

StacktraceOverride::StacktraceOverride(std::string) {}
StacktraceOverride::~StacktraceOverride() = default;

}  // namespace util
}  // namespace arrow

#endif
