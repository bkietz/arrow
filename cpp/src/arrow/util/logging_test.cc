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

#include <chrono>
#include <cstdint>
#include <iostream>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>

#include "arrow/util/logging.h"
#include "arrow/util/stacktrace.h"

// This code is adapted from
// https://github.com/ray-project/ray/blob/master/src/ray/util/logging_test.cc.

using testing::MatchesRegex;
using testing::Not;

namespace arrow {
namespace util {

int64_t current_time_ms() {
  std::chrono::milliseconds ms_since_epoch =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch());
  return ms_since_epoch.count();
}

// This is not really test.
// This file just print some information using the logging macro.

void PrintLog() {
  ARROW_LOG(DEBUG) << "This is the"
                   << " DEBUG"
                   << " message\n";

  ARROW_LOG(INFO) << "This is the"
                  << " INFO message\n";

  ARROW_LOG(WARNING) << "This is the"
                     << " WARNING message\n";

  ARROW_LOG(ERROR) << "This is the"
                   << " ERROR message\n";

  ARROW_CHECK(true) << "This is an ARROW_CHECK"
                    << " message but it won't show up\n";

  ASSERT_DEATH({ ARROW_LOG(FATAL) << "This is the FATAL message\n"; },
               StacktraceSupported() ? "This is the FATAL message"
                                       ".*PrintLog.*\n"
                                       ".*PrintLogTest_\\w+_Test.*TestBody\\.*"
                                     : "This is the FATAL message");

  ASSERT_DEATH({ ARROW_CHECK(false) << "This is an ARROW_CHECK message\n"; },
               StacktraceSupported() ? "This is an ARROW_CHECK message"
                                       ".*PrintLog.*\n"
                                       ".*PrintLogTest_\\w+_Test.*TestBody\\.*"
                                     : "This is an ARROW_CHECK message");
}

TEST(PrintLogTest, LogTestWithoutInit) {
  // Without ArrowLog::StartArrowLog, this should also work.
  PrintLog();
}

TEST(PrintLogTest, LogTestWithInit) {
  // Test empty app name.
  ArrowLog::StartArrowLog("", ArrowLogLevel::ARROW_DEBUG);
  PrintLog();
  ArrowLog::ShutDownArrowLog();
}

std::string Foo(int skip, bool omit_starts_with_b) { return PrintStacktrace(skip); }
std::string Bar(int skip, bool omit_starts_with_b) {
  if (omit_starts_with_b) {
    StacktraceOverride omit_this_frame;
    return Foo(skip, omit_starts_with_b);
  }
  return Foo(skip, omit_starts_with_b);
}
std::string Baz(int skip, bool omit_starts_with_b) {
  if (omit_starts_with_b) {
    StacktraceOverride omit_this_frame;
    return Bar(skip, omit_starts_with_b);
  }
  return Bar(skip, omit_starts_with_b);
}
std::string Quux(int skip, bool omit_starts_with_b) {
  return Baz(skip, omit_starts_with_b);
}
std::string Root(int skip, bool omit_starts_with_b = false) {
  return Quux(skip, omit_starts_with_b);
}

std::string StacktraceRegex(std::vector<std::string> names) {
  std::string re;
  for (const auto& name : names) {
    re += ".*" + name + ".* in .*.src.arrow.util.logging_test.cc:[0-9]+\n";
  }
  return re + ".*";
}

TEST(Stacktrace, BasicPrint) {
  if (!StacktraceSupported()) {
    GTEST_SKIP();
  };

  ASSERT_THAT(Root(/*skip=*/0), MatchesRegex(StacktraceRegex({
                                    "Foo",
                                    "Bar",
                                    "Baz",
                                    "Quux",
                                    "Root",
                                })));

  ASSERT_THAT(Root(/*skip=*/2), MatchesRegex(StacktraceRegex({
                                    "Baz",
                                    "Quux",
                                    "Root",
                                })));

  ASSERT_THAT(Root(/*skip=*/3), MatchesRegex(StacktraceRegex({
                                    "Quux",
                                    "Root",
                                })));
}

TEST(Stacktrace, OmitStartsWithB) {
  if (!StacktraceSupported()) {
    GTEST_SKIP();
  };

  ASSERT_THAT(Root(/*skip=*/0, /*omit_starts_with_b=*/true),
              MatchesRegex(StacktraceRegex({
                  "Foo",
                  "Quux",
                  "Root",
              })));

  ASSERT_THAT(Root(/*skip=*/0, /*omit_starts_with_b=*/true),
              Not(MatchesRegex(StacktraceRegex({
                  "Bar",
                  "Baz",
              }))));
}

std::string RootNonEmptyOverride() {
  StacktraceOverride frame_label{"frame_label"};
  return Quux(/*skip=*/0, /*omit_starts_with_b=*/false);
}

TEST(Stacktrace, NonEmptyOverride) {
  if (!StacktraceSupported()) {
    GTEST_SKIP();
  };

  ASSERT_THAT(RootNonEmptyOverride(), MatchesRegex(StacktraceRegex({
                                                       "Foo",
                                                       "Bar",
                                                       "Baz",
                                                       "Quux",
                                                   }) +
                                                   "frame_label.*"));
}

}  // namespace util

TEST(DcheckMacros, DoNotEvaluateReleaseMode) {
#ifdef NDEBUG
  int i = 0;
  auto f1 = [&]() {
    ++i;
    return true;
  };
  DCHECK(f1());
  ASSERT_EQ(0, i);
  auto f2 = [&]() {
    ++i;
    return i;
  };
  DCHECK_EQ(f2(), 0);
  DCHECK_NE(f2(), 0);
  DCHECK_LT(f2(), 0);
  DCHECK_LE(f2(), 0);
  DCHECK_GE(f2(), 0);
  DCHECK_GT(f2(), 0);
  ASSERT_EQ(0, i);
  ARROW_UNUSED(f1);
  ARROW_UNUSED(f2);
#endif
}

}  // namespace arrow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
