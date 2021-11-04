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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/util.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/type_fwd.h"
#include "arrow/util/async_util.h"
#include "arrow/util/cancel.h"
#include "arrow/util/macros.h"
#include "arrow/util/optional.h"
#include "arrow/util/unreachable.h"
#include "arrow/util/visibility.h"
#include "arrow/vendored/nanopb/pb_decode.h"
#include "arrow/vendored/nanopb/pb_encode.h"

namespace arrow_vendored {
// Awful kludge. This is the only way nanopb can do namespaces AFAICT
#include "generated/arrow/compute/exec/ir/simple.pb.c"
}  // namespace arrow_vendored

namespace arrow {
namespace compute {

template <typename Msg, typename Desc = arrow_vendored::nanopb::MessageDescriptor<Msg>>
Result<std::unique_ptr<Buffer>> Encode(const Msg& msg,
                                       MemoryPool* pool = default_memory_pool()) {
  size_t encoded_size;
  if (!pb_get_encoded_size(&encoded_size, Desc::fields(), &msg)) {
    return Status::IOError("couldn't get message size");
  }

  ARROW_ASSIGN_OR_RAISE(auto buf,
                        AllocateBuffer(static_cast<int64_t>(encoded_size), pool));

  auto stream = arrow_vendored::pb_ostream_from_buffer(buf->mutable_data(), encoded_size);
  if (!pb_encode(&stream, Desc::fields(), &msg)) {
    return Status::IOError("Encoding failed: ", PB_GET_ERROR(&stream));
  }

  DCHECK_EQ(stream.bytes_written, encoded_size);
  return std::move(buf);
}

template <typename Msg, typename Desc = arrow_vendored::nanopb::MessageDescriptor<Msg>>
Result<Msg> Decode(const Buffer& buf) {
  arrow_vendored::SimpleMessage decoded = {0};

  auto stream =
      arrow_vendored::pb_istream_from_buffer(buf.data(), static_cast<size_t>(buf.size()));
  if (!pb_decode(&stream, Desc::fields(), &decoded)) {
    return Status::IOError("Decoding failed: ", PB_GET_ERROR(&stream));
  }

  return decoded;
}

__attribute__((constructor)) void Usage() {
  // encode the message
  arrow_vendored::SimpleMessage encoded = {0};
  encoded.lucky_number = 13;
  auto buf = *Encode(encoded);

  auto decoded = *Decode<arrow_vendored::SimpleMessage>(*buf);

  DCHECK_EQ(decoded.lucky_number, encoded.lucky_number) << "Format failure?";
}

}  // namespace compute
}  // namespace arrow
