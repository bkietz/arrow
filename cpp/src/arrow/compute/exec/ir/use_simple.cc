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
#include "generated/arrow/compute/exec/ir/simple.pb.h"

namespace arrow {
namespace compute {

__attribute__((constructor)) void Usage() {
  std::vector<uint8_t> buffer;

  // encode the message
  SimpleMessage encoded = {0};
  encoded.lucky_number = 13;

  {
    size_t encoded_size;
    DCHECK(pb_get_encoded_size(&encoded_size, &SimpleMessage_msg, &encoded));
    buffer.resize(encoded_size);
  }

  pb_ostream_t ostream = pb_ostream_from_buffer(buffer.data(), buffer.size());

  DCHECK(pb_encode(&ostream, &SimpleMessage_msg, &encoded))
      << "Encoding failed: " << PB_GET_ERROR(&ostream);

  DCHECK_EQ(ostream.bytes_written, buffer.size());

  // decode the message
  SimpleMessage decoded = {0};

  pb_istream_t istream = pb_istream_from_buffer(buffer.data(), buffer.size());

  DCHECK(pb_decode(&istream, SimpleMessage_fields, &decoded))
      << "Decoding failed: " << PB_GET_ERROR(&istream);

  DCHECK_EQ(decoded.lucky_number, encoded.lucky_number) << "Format failure?";
}

}  // namespace compute
}  // namespace arrow
