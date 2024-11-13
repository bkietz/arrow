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

// Implement Arrow streaming binary format

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "arrow/ipc/dictionary.h"  // IWYU pragma: export
#include "arrow/ipc/message.h"
#include "arrow/ipc/options.h"
#include "arrow/result.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class Buffer;
class MemoryManager;
class MemoryPool;
class RecordBatch;
class Schema;
class Status;
class Table;
class Tensor;
class SparseTensor;

namespace io {

class OutputStream;

}  // namespace io

namespace ipc {

/// Intermediate data structure with metadata header, and zero
/// or more buffers for the message body.
struct IpcPayload {
  MessageType type = MessageType::NONE;
  std::shared_ptr<Buffer> metadata;
  std::vector<std::shared_ptr<Buffer>> body_buffers;
  std::vector<int64_t> variadic_buffer_counts;
  int64_t body_length = 0;      // serialized body length (padded, maybe compressed)
  int64_t raw_body_length = 0;  // initial uncompressed body length
};

struct WriteStats {
  /// Number of IPC messages written.
  int64_t num_messages = 0;
  /// Number of record batches written.
  int64_t num_record_batches = 0;
  /// Number of dictionary batches written.
  ///
  /// Note: num_dictionary_batches >= num_dictionary_deltas + num_replaced_dictionaries
  int64_t num_dictionary_batches = 0;

  /// Number of dictionary deltas written.
  int64_t num_dictionary_deltas = 0;
  /// Number of replaced dictionaries (i.e. where a dictionary batch replaces
  /// an existing dictionary with an unrelated new dictionary).
  int64_t num_replaced_dictionaries = 0;

  /// Total size in bytes of record batches emitted.
  /// The "raw" size counts the original buffer sizes, while the "serialized" size
  /// includes padding and (optionally) compression.
  int64_t total_raw_body_size = 0;
  int64_t total_serialized_body_size = 0;
};

/// Abstract interface for writing a stream of record batches
class ARROW_EXPORT RecordBatchWriter {
 public:
  virtual ~RecordBatchWriter();

  /// Write a record batch to the stream
  ///
  /// :param batch: the record batch to write to the stream
  /// :return: Status
  virtual Status WriteRecordBatch(const RecordBatch& batch) = 0;

  /// Write a record batch with custom metadata to the stream
  ///
  /// :param batch: the record batch to write to the stream
  /// :param custom_metadata: the record batch's custom metadata to write to the stream
  /// :return: Status
  virtual Status WriteRecordBatch(
      const RecordBatch& batch,
      const std::shared_ptr<const KeyValueMetadata>& custom_metadata);

  /// Write possibly-chunked table by creating sequence of record batches
  /// :param table: table to write
  /// :return: Status
  Status WriteTable(const Table& table);

  /// Write Table with a particular chunksize
  /// :param table: table to write
  /// :param max_chunksize: maximum number of rows for table chunks. To
  /// indicate that no maximum should be enforced, pass -1.
  /// :return: Status
  virtual Status WriteTable(const Table& table, int64_t max_chunksize);

  /// Perform any logic necessary to finish the stream
  ///
  /// :return: Status
  virtual Status Close() = 0;

  /// Return current write statistics
  virtual WriteStats stats() const = 0;
};

/// \defgroup record-batch-writer-factories Functions for creating RecordBatchWriter
/// instances
///
/// @{

/// Create a new IPC stream writer from stream sink and schema. User is
/// responsible for closing the actual OutputStream.
///
/// :param sink: output stream to write to
/// :param schema: the schema of the record batches to be written
/// :param options: options for serialization
/// :return: Result<std::shared_ptr<RecordBatchWriter>>
ARROW_EXPORT
Result<std::shared_ptr<RecordBatchWriter>> MakeStreamWriter(
    io::OutputStream* sink, const std::shared_ptr<Schema>& schema,
    const IpcWriteOptions& options = IpcWriteOptions::Defaults());

/// Create a new IPC stream writer from stream sink and schema. User is
/// responsible for closing the actual OutputStream.
///
/// :param sink: output stream to write to
/// :param schema: the schema of the record batches to be written
/// :param options: options for serialization
/// :return: Result<std::shared_ptr<RecordBatchWriter>>
ARROW_EXPORT
Result<std::shared_ptr<RecordBatchWriter>> MakeStreamWriter(
    std::shared_ptr<io::OutputStream> sink, const std::shared_ptr<Schema>& schema,
    const IpcWriteOptions& options = IpcWriteOptions::Defaults());

/// Create a new IPC file writer from stream sink and schema
///
/// :param sink: output stream to write to
/// :param schema: the schema of the record batches to be written
/// :param options: options for serialization, optional
/// :param metadata: custom metadata for File Footer, optional
/// :return: Result<std::shared_ptr<RecordBatchWriter>>
ARROW_EXPORT
Result<std::shared_ptr<RecordBatchWriter>> MakeFileWriter(
    io::OutputStream* sink, const std::shared_ptr<Schema>& schema,
    const IpcWriteOptions& options = IpcWriteOptions::Defaults(),
    const std::shared_ptr<const KeyValueMetadata>& metadata = NULLPTR);

/// Create a new IPC file writer from stream sink and schema
///
/// :param sink: output stream to write to
/// :param schema: the schema of the record batches to be written
/// :param options: options for serialization, optional
/// :param metadata: custom metadata for File Footer, optional
/// :return: Result<std::shared_ptr<RecordBatchWriter>>
ARROW_EXPORT
Result<std::shared_ptr<RecordBatchWriter>> MakeFileWriter(
    std::shared_ptr<io::OutputStream> sink, const std::shared_ptr<Schema>& schema,
    const IpcWriteOptions& options = IpcWriteOptions::Defaults(),
    const std::shared_ptr<const KeyValueMetadata>& metadata = NULLPTR);

/// @}

/// Low-level API for writing a record batch (without schema)
/// to an OutputStream as encapsulated IPC message. See Arrow format
/// documentation for more detail.
///
/// :param batch: the record batch to write
/// :param buffer_start_offset: the start offset to use in the buffer metadata,
/// generally should be 0
/// :param dst: an OutputStream
/// :param metadata_length[out]: the size of the length-prefixed flatbuffer
/// including padding to a 64-byte boundary
/// :param body_length[out]: the size of the contiguous buffer block plus
/// :param options: options for serialization
/// :return: Status
ARROW_EXPORT
Status WriteRecordBatch(const RecordBatch& batch, int64_t buffer_start_offset,
                        io::OutputStream* dst, int32_t* metadata_length,
                        int64_t* body_length, const IpcWriteOptions& options);

/// Serialize record batch as encapsulated IPC message in a new buffer
///
/// :param batch: the record batch
/// :param options: the IpcWriteOptions to use for serialization
/// :return: the serialized message
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> SerializeRecordBatch(const RecordBatch& batch,
                                                     const IpcWriteOptions& options);

/// Serialize record batch as encapsulated IPC message in a new buffer
///
/// :param batch: the record batch
/// :param mm: a MemoryManager to allocate memory from
/// :return: the serialized message
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> SerializeRecordBatch(const RecordBatch& batch,
                                                     std::shared_ptr<MemoryManager> mm);

/// Write record batch to OutputStream
///
/// :param batch: the record batch to write
/// :param options: the IpcWriteOptions to use for serialization
/// :param out: the OutputStream to write the output to
/// :return: Status
///
/// If writing to pre-allocated memory, you can use
/// arrow::ipc::GetRecordBatchSize to compute how much space is required
ARROW_EXPORT
Status SerializeRecordBatch(const RecordBatch& batch, const IpcWriteOptions& options,
                            io::OutputStream* out);

/// Serialize schema as encapsulated IPC message
///
/// :param schema: the schema to write
/// :param pool: a MemoryPool to allocate memory from
/// :return: the serialized schema
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> SerializeSchema(const Schema& schema,
                                                MemoryPool* pool = default_memory_pool());

/// Write multiple record batches to OutputStream, including schema
/// :param batches: a vector of batches. Must all have same schema
/// :param options: options for serialization
/// :param dst[out]: an OutputStream
/// :return: Status
ARROW_EXPORT
Status WriteRecordBatchStream(const std::vector<std::shared_ptr<RecordBatch>>& batches,
                              const IpcWriteOptions& options, io::OutputStream* dst);

/// Compute the number of bytes needed to write an IPC payload
///     including metadata
///
/// :param payload: the IPC payload to write
/// :param options: write options
/// :return: the size of the complete encapsulated message
ARROW_EXPORT
int64_t GetPayloadSize(const IpcPayload& payload,
                       const IpcWriteOptions& options = IpcWriteOptions::Defaults());

/// Compute the number of bytes needed to write a record batch including metadata
///
/// :param batch: the record batch to write
/// :param size[out]: the size of the complete encapsulated message
/// :return: Status
ARROW_EXPORT
Status GetRecordBatchSize(const RecordBatch& batch, int64_t* size);

/// Compute the number of bytes needed to write a record batch including metadata
///
/// :param batch: the record batch to write
/// :param options: options for serialization
/// :param size[out]: the size of the complete encapsulated message
/// :return: Status
ARROW_EXPORT
Status GetRecordBatchSize(const RecordBatch& batch, const IpcWriteOptions& options,
                          int64_t* size);

/// Compute the number of bytes needed to write a tensor including metadata
///
/// :param tensor: the tensor to write
/// :param size[out]: the size of the complete encapsulated message
/// :return: Status
ARROW_EXPORT
Status GetTensorSize(const Tensor& tensor, int64_t* size);

/// EXPERIMENTAL: Convert arrow::Tensor to a Message with minimal memory
/// allocation
///
/// :param tensor: the Tensor to write
/// :param pool: MemoryPool to allocate space for metadata
/// :return: the resulting Message
ARROW_EXPORT
Result<std::unique_ptr<Message>> GetTensorMessage(const Tensor& tensor, MemoryPool* pool);

/// Write arrow::Tensor as a contiguous message.
///
/// The metadata and body are written assuming 64-byte alignment. It is the
/// user's responsibility to ensure that the OutputStream has been aligned
/// to a 64-byte multiple before writing the message.
///
/// The message is written out as followed:
/// ```
/// <metadata size> <metadata> <tensor data>
/// ```
///
/// :param tensor: the Tensor to write
/// :param dst: the OutputStream to write to
/// :param metadata_length[out]: the actual metadata length, including padding
/// :param body_length[out]: the actual message body length
/// :return: Status
ARROW_EXPORT
Status WriteTensor(const Tensor& tensor, io::OutputStream* dst, int32_t* metadata_length,
                   int64_t* body_length);

/// EXPERIMENTAL: Convert arrow::SparseTensor to a Message with minimal memory
/// allocation
///
/// The message is written out as followed:
/// ```
/// <metadata size> <metadata> <sparse index> <sparse tensor body>
/// ```
///
/// :param sparse_tensor: the SparseTensor to write
/// :param pool: MemoryPool to allocate space for metadata
/// :return: the resulting Message
ARROW_EXPORT
Result<std::unique_ptr<Message>> GetSparseTensorMessage(const SparseTensor& sparse_tensor,
                                                        MemoryPool* pool);

/// EXPERIMENTAL: Write arrow::SparseTensor as a contiguous message. The metadata,
/// sparse index, and body are written assuming 64-byte alignment. It is the
/// user's responsibility to ensure that the OutputStream has been aligned
/// to a 64-byte multiple before writing the message.
///
/// :param sparse_tensor: the SparseTensor to write
/// :param dst: the OutputStream to write to
/// :param metadata_length[out]: the actual metadata length, including padding
/// :param body_length[out]: the actual message body length
/// :return: Status
ARROW_EXPORT
Status WriteSparseTensor(const SparseTensor& sparse_tensor, io::OutputStream* dst,
                         int32_t* metadata_length, int64_t* body_length);

/// Compute IpcPayload for the given schema
/// :param schema: the Schema that is being serialized
/// :param options: options for serialization
/// :param mapper: object mapping dictionary fields to dictionary ids
/// :param out[out]: the returned vector of IpcPayloads
/// :return: Status
ARROW_EXPORT
Status GetSchemaPayload(const Schema& schema, const IpcWriteOptions& options,
                        const DictionaryFieldMapper& mapper, IpcPayload* out);

/// Compute IpcPayload for a dictionary
/// :param id: the dictionary id
/// :param dictionary: the dictionary values
/// :param options: options for serialization
/// :param payload[out]: the output IpcPayload
/// :return: Status
ARROW_EXPORT
Status GetDictionaryPayload(int64_t id, const std::shared_ptr<Array>& dictionary,
                            const IpcWriteOptions& options, IpcPayload* payload);

/// Compute IpcPayload for a dictionary
/// :param id: the dictionary id
/// :param is_delta: whether the dictionary is a delta dictionary
/// :param dictionary: the dictionary values
/// :param options: options for serialization
/// :param payload[out]: the output IpcPayload
/// :return: Status
ARROW_EXPORT
Status GetDictionaryPayload(int64_t id, bool is_delta,
                            const std::shared_ptr<Array>& dictionary,
                            const IpcWriteOptions& options, IpcPayload* payload);

/// Compute IpcPayload for the given record batch
/// :param batch: the RecordBatch that is being serialized
/// :param options: options for serialization
/// :param out[out]: the returned IpcPayload
/// :return: Status
ARROW_EXPORT
Status GetRecordBatchPayload(const RecordBatch& batch, const IpcWriteOptions& options,
                             IpcPayload* out);

/// Compute IpcPayload for the given record batch and custom metadata
/// :param batch: the RecordBatch that is being serialized
/// :param custom_metadata: the custom metadata to be serialized with the record batch
/// :param options: options for serialization
/// :param out[out]: the returned IpcPayload
/// :return: Status
ARROW_EXPORT
Status GetRecordBatchPayload(
    const RecordBatch& batch,
    const std::shared_ptr<const KeyValueMetadata>& custom_metadata,
    const IpcWriteOptions& options, IpcPayload* out);

/// Write an IPC payload to the given stream.
/// :param payload: the payload to write
/// :param options: options for serialization
/// :param dst: The stream to write the payload to.
/// :param metadata_length[out]: the length of the serialized metadata
/// :return: Status
ARROW_EXPORT
Status WriteIpcPayload(const IpcPayload& payload, const IpcWriteOptions& options,
                       io::OutputStream* dst, int32_t* metadata_length);

/// Compute IpcPayload for the given sparse tensor
/// :param sparse_tensor: the SparseTensor that is being serialized
/// :param pool[in,out]: for any required temporary memory allocations
/// :param out[out]: the returned IpcPayload
/// :return: Status
ARROW_EXPORT
Status GetSparseTensorPayload(const SparseTensor& sparse_tensor, MemoryPool* pool,
                              IpcPayload* out);

namespace internal {

// These internal APIs may change without warning or deprecation

class ARROW_EXPORT IpcPayloadWriter {
 public:
  virtual ~IpcPayloadWriter();

  // Default implementation is a no-op
  virtual Status Start();

  virtual Status WritePayload(const IpcPayload& payload) = 0;

  virtual Status Close() = 0;
};

/// Create a new IPC payload stream writer from stream sink. User is
/// responsible for closing the actual OutputStream.
///
/// :param sink: output stream to write to
/// :param options: options for serialization
/// :return: Result<std::shared_ptr<IpcPayloadWriter>>
ARROW_EXPORT
Result<std::unique_ptr<IpcPayloadWriter>> MakePayloadStreamWriter(
    io::OutputStream* sink, const IpcWriteOptions& options = IpcWriteOptions::Defaults());

/// Create a new IPC payload file writer from stream sink.
///
/// :param sink: output stream to write to
/// :param schema: the schema of the record batches to be written
/// :param options: options for serialization, optional
/// :param metadata: custom metadata for File Footer, optional
/// :return: Status
ARROW_EXPORT
Result<std::unique_ptr<IpcPayloadWriter>> MakePayloadFileWriter(
    io::OutputStream* sink, const std::shared_ptr<Schema>& schema,
    const IpcWriteOptions& options = IpcWriteOptions::Defaults(),
    const std::shared_ptr<const KeyValueMetadata>& metadata = NULLPTR);

/// Create a new RecordBatchWriter from IpcPayloadWriter and schema.
///
/// The format is implicitly the IPC stream format (allowing dictionary
/// replacement and deltas).
///
/// :param sink: the IpcPayloadWriter to write to
/// :param schema: the schema of the record batches to be written
/// :param options: options for serialization
/// :return: Result<std::unique_ptr<RecordBatchWriter>>
ARROW_EXPORT
Result<std::unique_ptr<RecordBatchWriter>> OpenRecordBatchWriter(
    std::unique_ptr<IpcPayloadWriter> sink, const std::shared_ptr<Schema>& schema,
    const IpcWriteOptions& options = IpcWriteOptions::Defaults());

}  // namespace internal
}  // namespace ipc
}  // namespace arrow
