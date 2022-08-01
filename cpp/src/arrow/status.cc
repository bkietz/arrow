// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// A Status encapsulates the result of an operation.  It may indicate success,
// or it may indicate an error with an associated error message.
//
// Multiple threads can invoke const methods on a Status without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same Status must use
// external synchronization.

#include "arrow/status.h"

#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>

#include "arrow/util/logging.h"
#include "arrow/util/stacktrace.h"

namespace arrow {

class StatusStateFreeList {
 public:
  static Status::State* MakeError(StatusCode code, std::string msg,
                                  std::shared_ptr<StatusDetail> detail) {
    ARROW_CHECK_NE(code, StatusCode::OK) << "Cannot construct ok status with message";

    std::unique_lock<std::mutex> lock{instance().mutex_};
    instance().states_.emplace_back();
    auto state = &instance().states_.back();
    lock.unlock();

    state->code = code;
    state->msg = std::move(msg);
    #ifdef ARROW_EXTRA_ERROR_CONTEXT
    state->stacktrace = util::PrintStacktrace(/*skip=*/2);
    #endif
    state->detail = std::move(detail);
    return state;
  }

  static void GarbageCollect() { instance().states_.clear(); }

 private:
  static StatusStateFreeList& instance() {
    static StatusStateFreeList instance;
    return instance;
  }

  std::mutex mutex_;
  std::deque<Status::State> states_;
};

void Status::GarbageCollect() { StatusStateFreeList::GarbageCollect(); }

Status::Status(StatusCode code, std::string msg)
    : state_{StatusStateFreeList::MakeError(code, std::move(msg), nullptr)} {}

Status::Status(StatusCode code, std::string msg, std::shared_ptr<StatusDetail> detail)
    : state_{StatusStateFreeList::MakeError(code, std::move(msg), std::move(detail))} {}

Status::Permanent::Permanent(StatusCode code, std::string msg,
                             std::shared_ptr<StatusDetail> detail) {
  state_.code = code;
  state_.msg = std::move(msg);
  state_.detail = std::move(detail);
}

Status Status::Permanent::status() const {
  Status out;
  out.state_ = &state_;
  return out;
}

std::string Status::CodeAsString() const {
  if (state_ == nullptr) {
    return "OK";
  }
  return CodeAsString(code());
}

std::string Status::CodeAsString(StatusCode code) {
  const char* type;
  switch (code) {
    case StatusCode::OK:
      type = "OK";
      break;
    case StatusCode::OutOfMemory:
      type = "Out of memory";
      break;
    case StatusCode::KeyError:
      type = "Key error";
      break;
    case StatusCode::TypeError:
      type = "Type error";
      break;
    case StatusCode::Invalid:
      type = "Invalid";
      break;
    case StatusCode::Cancelled:
      type = "Cancelled";
      break;
    case StatusCode::IOError:
      type = "IOError";
      break;
    case StatusCode::CapacityError:
      type = "Capacity error";
      break;
    case StatusCode::IndexError:
      type = "Index error";
      break;
    case StatusCode::UnknownError:
      type = "Unknown error";
      break;
    case StatusCode::NotImplemented:
      type = "NotImplemented";
      break;
    case StatusCode::SerializationError:
      type = "Serialization error";
      break;
    case StatusCode::CodeGenError:
      type = "CodeGenError in Gandiva";
      break;
    case StatusCode::ExpressionValidationError:
      type = "ExpressionValidationError";
      break;
    case StatusCode::ExecutionError:
      type = "ExecutionError in Gandiva";
      break;
    default:
      type = "Unknown";
      break;
  }
  return std::string(type);
}

std::string Status::ToString() const {
  std::string result(CodeAsString());
  if (state_ == nullptr) {
    return result;
  }
  result += ": ";
  result += state_->msg;
  if (state_->detail != nullptr) {
    result += ". Detail: ";
    result += state_->detail->ToString();
  }
  #ifdef ARROW_EXTRA_ERROR_CONTEXT
  result += "\n" + state_->stacktrace;
  #endif

  return result;
}

void Status::Abort() const { Abort(std::string()); }

void Status::Abort(const std::string& message) const {
  std::cerr << "-- Arrow Fatal Error --\n";
  if (!message.empty()) {
    std::cerr << message << "\n";
  }
  std::cerr << ToString() << std::endl;
  std::abort();
}

void Status::Warn() const { ARROW_LOG(WARNING) << ToString(); }

void Status::Warn(const std::string& message) const {
  ARROW_LOG(WARNING) << message << ": " << ToString();
}

}  // namespace arrow
