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
// under the License. template <typename T>

#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "arrow/util/macros.h"

namespace arrow {
namespace util {

template <typename Stream>
void StreamInto(Stream* stream) {}

template <typename Stream, typename Head, typename... Tail>
void StreamInto(Stream* stream, Head&& head, Tail&&... tail) {
  *stream << std::forward<Head>(head);
  StreamInto(stream, std::forward<Tail>(tail)...);
}

template <typename... Args>
std::string StringBuilder(Args&&... args) {
  std::ostringstream ss;
  StreamInto(&ss, std::forward<Args>(args)...);
  return ss.str();
}

}  // namespace util
}  // namespace arrow
