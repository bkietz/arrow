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

// C++ object model and user API for interprocess schema messaging

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "arrow/io/type_fwd.h"
#include "arrow/ipc/type_fwd.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace ipc {

struct IpcWriteOptions;

// Read interface classes. We do not fully deserialize the flatbuffers so that
// individual fields metadata can be retrieved from very large schema without
//

/// An IPC message including metadata and body
class ARROW_EXPORT Message {
 public:
  /// Construct message, but do not validate
  ///
  /// Use at your own risk; Message::Open has more metadata validation
  Message(std::shared_ptr<Buffer> metadata, std::shared_ptr<Buffer> body);

  ~Message();

  /// Create and validate a Message instance from two buffers
  ///
  /// :param metadata: a buffer containing the Flatbuffer metadata
  /// :param body: a buffer containing the message body, which may be null
  /// :return: the created message
  static Result<std::unique_ptr<Message>> Open(std::shared_ptr<Buffer> metadata,
                                               std::shared_ptr<Buffer> body);

  /// Read message body and create Message given Flatbuffer metadata
  /// :param metadata: containing a serialized Message flatbuffer
  /// :param stream: an InputStream
  /// :return: the created Message
  ///
  /// ```{note}
  /// If stream supports zero-copy, this is zero-copy
  static Result<std::unique_ptr<Message>> ReadFrom(std::shared_ptr<Buffer> metadata,
                                                   io::InputStream* stream);

  /// Read message body from position in file, and create Message given
  /// the Flatbuffer metadata
  /// :param offset: the position in the file where the message body starts.
  /// :param metadata: containing a serialized Message flatbuffer
  /// :param file: the seekable file interface to read from
  /// :return: the created Message
  ///
  /// ```{note}
  /// If file supports zero-copy, this is zero-copy
  static Result<std::unique_ptr<Message>> ReadFrom(const int64_t offset,
                                                   std::shared_ptr<Buffer> metadata,
                                                   io::RandomAccessFile* file);

  /// Return true if message type and contents are equal
  ///
  /// :param other: another message
  /// :return: true if contents equal
  bool Equals(const Message& other) const;

  /// the Message metadata
  ///
  /// :return: buffer
  std::shared_ptr<Buffer> metadata() const;

  /// Custom metadata serialized in metadata Flatbuffer. Returns nullptr
  /// when none set
  const std::shared_ptr<const KeyValueMetadata>& custom_metadata() const;

  /// the Message body, if any
  ///
  /// :return: buffer is null if no body
  std::shared_ptr<Buffer> body() const;

  /// The expected body length according to the metadata, for
  /// verification purposes
  int64_t body_length() const;

  /// The Message type
  MessageType type() const;

  /// The Message metadata version
  MetadataVersion metadata_version() const;

  const void* header() const;

  /// Write length-prefixed metadata and body to output stream
  ///
  /// :param file: output stream to write to
  /// :param options: IPC writing options including alignment
  /// :param output_length[out]: the number of bytes written
  /// :return: Status
  Status SerializeTo(io::OutputStream* file, const IpcWriteOptions& options,
                     int64_t* output_length) const;

  /// Return true if the Message metadata passes Flatbuffer validation
  bool Verify() const;

  /// Whether a given message type needs a body.
  static bool HasBody(MessageType type) {
    return type != MessageType::NONE && type != MessageType::SCHEMA;
  }

 private:
  // Hide serialization details from user API
  class MessageImpl;
  std::unique_ptr<MessageImpl> impl_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(Message);
};

ARROW_EXPORT std::string FormatMessageType(MessageType type);

/// An abstract class to listen events from MessageDecoder.
///
/// This API is EXPERIMENTAL.
///
/// ```{versionadded} 0.17.0
/// ```
class ARROW_EXPORT MessageDecoderListener {
 public:
  virtual ~MessageDecoderListener() = default;

  /// Called when a message is decoded.
  ///
  /// MessageDecoder calls this method when it decodes a message. This
  /// method is called multiple times when the target stream has
  /// multiple messages.
  ///
  /// :param message: a decoded message
  /// :return: Status
  virtual Status OnMessageDecoded(std::unique_ptr<Message> message) = 0;

  /// Called when the decoder state is changed to
  /// MessageDecoder::State::INITIAL.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// :return: Status
  virtual Status OnInitial();

  /// Called when the decoder state is changed to
  /// MessageDecoder::State::METADATA_LENGTH.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// :return: Status
  virtual Status OnMetadataLength();

  /// Called when the decoder state is changed to
  /// MessageDecoder::State::METADATA.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// :return: Status
  virtual Status OnMetadata();

  /// Called when the decoder state is changed to
  /// MessageDecoder::State::BODY.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// :return: Status
  virtual Status OnBody();

  /// Called when the decoder state is changed to
  /// MessageDecoder::State::EOS.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// :return: Status
  virtual Status OnEOS();
};

/// Assign a message decoded by MessageDecoder.
///
/// This API is EXPERIMENTAL.
///
/// ```{versionadded} 0.17.0
/// ```
class ARROW_EXPORT AssignMessageDecoderListener : public MessageDecoderListener {
 public:
  /// Construct a listener that assigns a decoded message to the
  /// specified location.
  ///
  /// :param message: a location to store the received message
  explicit AssignMessageDecoderListener(std::unique_ptr<Message>* message)
      : message_(message) {}

  virtual ~AssignMessageDecoderListener() = default;

  Status OnMessageDecoded(std::unique_ptr<Message> message) override {
    *message_ = std::move(message);
    return Status::OK();
  }

 private:
  std::unique_ptr<Message>* message_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(AssignMessageDecoderListener);
};

/// Push style message decoder that receives data from user.
///
/// This API is EXPERIMENTAL.
///
/// ```{versionadded} 0.17.0
/// ```
class ARROW_EXPORT MessageDecoder {
 public:
  /// State for reading a message
  enum State {
    /// The initial state. It requires one of the followings as the next data:
    ///
    ///   * int32_t continuation token
    ///   * int32_t end-of-stream mark (== 0)
    ///   * int32_t metadata length (backward compatibility for
    ///     reading old IPC messages produced prior to version 0.15.0
    INITIAL,

    /// It requires int32_t metadata length.
    METADATA_LENGTH,

    /// It requires metadata.
    METADATA,

    /// It requires message body.
    BODY,

    /// The end-of-stream state. No more data is processed.
    EOS,
  };

  /// Construct a message decoder.
  ///
  /// :param listener: a MessageDecoderListener that responds events from
  /// the decoder
  /// :param pool: an optional MemoryPool to copy metadata on the
  /// :param skip_body: if true the body will be skipped even if the message has a body
  /// CPU, if required
  explicit MessageDecoder(std::shared_ptr<MessageDecoderListener> listener,
                          MemoryPool* pool = default_memory_pool(),
                          bool skip_body = false);

  /// Construct a message decoder with the specified state.
  ///
  /// This is a construct for advanced users that know how to decode
  /// Message.
  ///
  /// :param listener: a MessageDecoderListener that responds events from
  /// the decoder
  /// :param initial_state: an initial state of the decode
  /// :param initial_next_required_size: the number of bytes needed
  /// to run the next action
  /// :param pool: an optional MemoryPool to copy metadata on the
  /// CPU, if required
  /// :param skip_body: if true the body will be skipped even if the message has a body
  MessageDecoder(std::shared_ptr<MessageDecoderListener> listener, State initial_state,
                 int64_t initial_next_required_size,
                 MemoryPool* pool = default_memory_pool(), bool skip_body = false);

  virtual ~MessageDecoder();

  /// Feed data to the decoder as a raw data.
  ///
  /// If the decoder can decode one or more messages by the data, the
  /// decoder calls listener->OnMessageDecoded() with a decoded
  /// message multiple times.
  ///
  /// If the state of the decoder is changed, corresponding callbacks
  /// on listener is called:
  ///
  /// * MessageDecoder::State::INITIAL: listener->OnInitial()
  /// * MessageDecoder::State::METADATA_LENGTH: listener->OnMetadataLength()
  /// * MessageDecoder::State::METADATA: listener->OnMetadata()
  /// * MessageDecoder::State::BODY: listener->OnBody()
  /// * MessageDecoder::State::EOS: listener->OnEOS()
  ///
  /// :param data: a raw data to be processed. This data isn't
  /// copied. The passed memory must be kept alive through message
  /// processing.
  /// :param size: raw data size.
  /// :return: Status
  Status Consume(const uint8_t* data, int64_t size);

  /// Feed data to the decoder as a Buffer.
  ///
  /// If the decoder can decode one or more messages by the Buffer,
  /// the decoder calls listener->OnMessageDecoded() with a decoded
  /// message multiple times.
  ///
  /// :param buffer: a Buffer to be processed.
  /// :return: Status
  Status Consume(std::shared_ptr<Buffer> buffer);

  /// Return the number of bytes needed to advance the state of
  /// the decoder.
  ///
  /// This method is provided for users who want to optimize performance.
  /// Normal users don't need to use this method.
  ///
  /// Here is an example usage for normal users:
  ///
  /// ~~~{.cpp}
  /// decoder.Consume(buffer1);
  /// decoder.Consume(buffer2);
  /// decoder.Consume(buffer3);
  /// ~~~
  ///
  /// Decoder has internal buffer. If consumed data isn't enough to
  /// advance the state of the decoder, consumed data is buffered to
  /// the internal buffer. It causes performance overhead.
  ///
  /// If you pass next_required_size() size data to each Consume()
  /// call, the decoder doesn't use its internal buffer. It improves
  /// performance.
  ///
  /// Here is an example usage to avoid using internal buffer:
  ///
  /// ~~~{.cpp}
  /// buffer1 = get_data(decoder.next_required_size());
  /// decoder.Consume(buffer1);
  /// buffer2 = get_data(decoder.next_required_size());
  /// decoder.Consume(buffer2);
  /// ~~~
  ///
  /// Users can use this method to avoid creating small
  /// chunks. Message body must be contiguous data. If users pass
  /// small chunks to the decoder, the decoder needs concatenate small
  /// chunks internally. It causes performance overhead.
  ///
  /// Here is an example usage to reduce small chunks:
  ///
  /// ~~~{.cpp}
  /// buffer = AllocateResizableBuffer();
  /// while ((small_chunk = get_data(&small_chunk_size))) {
  ///   auto current_buffer_size = buffer->size();
  ///   buffer->Resize(current_buffer_size + small_chunk_size);
  ///   memcpy(buffer->mutable_data() + current_buffer_size,
  ///          small_chunk,
  ///          small_chunk_size);
  ///   if (buffer->size() < decoder.next_required_size()) {
  ///     continue;
  ///   }
  ///   std::shared_ptr<arrow::Buffer> chunk(buffer.release());
  ///   decoder.Consume(chunk);
  ///   buffer = AllocateResizableBuffer();
  /// }
  /// if (buffer->size() > 0) {
  ///   std::shared_ptr<arrow::Buffer> chunk(buffer.release());
  ///   decoder.Consume(chunk);
  /// }
  /// ~~~
  ///
  /// :return: the number of bytes needed to advance the state of the
  /// decoder
  int64_t next_required_size() const;

  /// Return the current state of the decoder.
  ///
  /// This method is provided for users who want to optimize performance.
  /// Normal users don't need to use this method.
  ///
  /// Decoder doesn't need Buffer to process data on the
  /// MessageDecoder::State::INITIAL state and the
  /// MessageDecoder::State::METADATA_LENGTH. Creating Buffer has
  /// performance overhead. Advanced users can avoid creating Buffer
  /// by checking the current state of the decoder:
  ///
  /// ~~~{.cpp}
  /// switch (decoder.state()) {
  ///   MessageDecoder::State::INITIAL:
  ///   MessageDecoder::State::METADATA_LENGTH:
  ///     {
  ///       uint8_t data[sizeof(int32_t)];
  ///       auto data_size = input->Read(decoder.next_required_size(), data);
  ///       decoder.Consume(data, data_size);
  ///     }
  ///     break;
  ///   default:
  ///     {
  ///       auto buffer = input->Read(decoder.next_required_size());
  ///       decoder.Consume(buffer);
  ///     }
  ///     break;
  /// }
  /// ~~~
  ///
  /// :return: the current state
  State state() const;

 private:
  class MessageDecoderImpl;
  std::unique_ptr<MessageDecoderImpl> impl_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(MessageDecoder);
};

/// Abstract interface for a sequence of messages
/// ```{versionadded} 0.5.0
/// ```
class ARROW_EXPORT MessageReader {
 public:
  virtual ~MessageReader() = default;

  /// Create MessageReader that reads from InputStream
  static std::unique_ptr<MessageReader> Open(io::InputStream* stream);

  /// Create MessageReader that reads from owned InputStream
  static std::unique_ptr<MessageReader> Open(
      const std::shared_ptr<io::InputStream>& owned_stream);

  /// Read next Message from the interface
  ///
  /// :return: an arrow::ipc::Message instance
  virtual Result<std::unique_ptr<Message>> ReadNextMessage() = 0;
};

// the first parameter of the function should be a pointer to metadata (aka.
// org::apache::arrow::flatbuf::RecordBatch*)
using FieldsLoaderFunction = std::function<Status(const void*, io::RandomAccessFile*)>;

/// Read encapsulated RPC message from position in file
///
/// Read a length-prefixed message flatbuffer starting at the indicated file
/// offset. If the message has a body with non-zero length, it will also be
/// read
///
/// The metadata_length includes at least the length prefix and the flatbuffer
///
/// :param offset: the position in the file where the message starts. The
/// first 4 bytes after the offset are the message length
/// :param metadata_length: the total number of bytes to read from file
/// :param file: the seekable file interface to read from
/// :param fields_loader: the function for loading subset of fields from the given file
/// :return: the message read

ARROW_EXPORT
Result<std::unique_ptr<Message>> ReadMessage(
    const int64_t offset, const int32_t metadata_length, io::RandomAccessFile* file,
    const FieldsLoaderFunction& fields_loader = {});

/// Read encapsulated RPC message from cached buffers
///
/// The buffers should contain an entire message.  Partial reads are not handled.
///
/// This method can be used to read just the metadata by passing in a nullptr for the
/// body.  The body will then be skipped and the body size will not be validated.
///
/// If the body buffer is provided then it must be the complete body buffer
///
/// This is similar to Message::Open but performs slightly more validation (e.g. checks
/// to see that the metadata length is correct and that the body is the size the metadata
/// expected)
///
/// :param metadata: The bytes for the metadata
/// :param body: The bytes for the body
/// :return: The message represented by the buffers
ARROW_EXPORT Result<std::unique_ptr<Message>> ReadMessage(
    std::shared_ptr<Buffer> metadata, std::shared_ptr<Buffer> body);

ARROW_EXPORT
Future<std::shared_ptr<Message>> ReadMessageAsync(
    const int64_t offset, const int32_t metadata_length, const int64_t body_length,
    io::RandomAccessFile* file, const io::IOContext& context = io::default_io_context());

/// Advance stream to an 8-byte offset if its position is not a multiple
/// of 8 already
/// :param stream: an input stream
/// :param alignment: the byte multiple for the metadata prefix, usually 8
/// or 64, to ensure the body starts on a multiple of that alignment
/// :return: Status
ARROW_EXPORT
Status AlignStream(io::InputStream* stream, int32_t alignment = 8);

/// Advance stream to an 8-byte offset if its position is not a multiple
/// of 8 already
/// :param stream: an output stream
/// :param alignment: the byte multiple for the metadata prefix, usually 8
/// or 64, to ensure the body starts on a multiple of that alignment
/// :return: Status
ARROW_EXPORT
Status AlignStream(io::OutputStream* stream, int32_t alignment = 8);

/// Return error Status if file position is not a multiple of the
/// indicated alignment
ARROW_EXPORT
Status CheckAligned(io::FileInterface* stream, int32_t alignment = 8);

/// Read encapsulated IPC message (metadata and body) from InputStream
///
/// Returns null if there are not enough bytes available or the
/// message length is 0 (e.g. EOS in a stream)
///
/// :param stream: an input stream
/// :param pool: an optional MemoryPool to copy metadata on the CPU, if required
/// :return: Message
ARROW_EXPORT
Result<std::unique_ptr<Message>> ReadMessage(io::InputStream* stream,
                                             MemoryPool* pool = default_memory_pool());

/// Feed data from InputStream to MessageDecoder to decode an
/// encapsulated IPC message (metadata and body)
///
/// This API is EXPERIMENTAL.
///
/// :param decoder: a decoder
/// :param stream: an input stream
/// :return: Status
///
/// ```{versionadded} 0.17.0
/// ```
ARROW_EXPORT
Status DecodeMessage(MessageDecoder* decoder, io::InputStream* stream);

/// Write encapsulated IPC message Does not make assumptions about
/// whether the stream is aligned already. Can write legacy (pre
/// version 0.15.0) IPC message if option set
///
/// continuation: 0xFFFFFFFF
/// message_size: int32
/// message: const void*
/// padding
///
///
/// :param message: a buffer containing the metadata to write
/// :param options: IPC writing options, including alignment and
/// legacy message support
/// :param file[in,out]: the OutputStream to write to
/// :param message_length[out]: the total size of the payload written including
/// padding
/// :return: Status
Status WriteMessage(const Buffer& message, const IpcWriteOptions& options,
                    io::OutputStream* file, int32_t* message_length);

}  // namespace ipc
}  // namespace arrow
