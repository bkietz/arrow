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

/// Implementation of Flight RPC client.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/ipc/options.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/writer.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/cancel.h"

#include "arrow/flight/type_fwd.h"
#include "arrow/flight/types.h"  // IWYU pragma: keep
#include "arrow/flight/visibility.h"

namespace arrow {

class RecordBatch;
class Schema;

namespace flight {

/// A duration type for Flight call timeouts.
typedef std::chrono::duration<double, std::chrono::seconds::period> TimeoutDuration;

/// Hints to the underlying RPC layer for Arrow Flight calls.
class ARROW_FLIGHT_EXPORT FlightCallOptions {
 public:
  /// Create a default set of call options.
  FlightCallOptions();

  /// An optional timeout for this call. Negative durations
  /// mean an implementation-defined default behavior will be used
  /// instead. This is the default value.
  TimeoutDuration timeout;

  /// IPC reader options, if applicable for the call.
  ipc::IpcReadOptions read_options;

  /// IPC writer options, if applicable for the call.
  ipc::IpcWriteOptions write_options;

  /// Headers for client to add to context.
  std::vector<std::pair<std::string, std::string>> headers;

  /// A token to enable interactive user cancellation of long-running requests.
  StopToken stop_token;

  /// An optional memory manager to control where to allocate incoming data.
  std::shared_ptr<MemoryManager> memory_manager;
};

/// Indicate that the client attempted to write a message
///     larger than the soft limit set via write_size_limit_bytes.
class ARROW_FLIGHT_EXPORT FlightWriteSizeStatusDetail : public arrow::StatusDetail {
 public:
  explicit FlightWriteSizeStatusDetail(int64_t limit, int64_t actual)
      : limit_(limit), actual_(actual) {}
  const char* type_id() const override;
  std::string ToString() const override;
  int64_t limit() const { return limit_; }
  int64_t actual() const { return actual_; }

  /// Extract this status detail from a status, or return
  ///     nullptr if the status doesn't contain this status detail.
  static std::shared_ptr<FlightWriteSizeStatusDetail> UnwrapStatus(
      const arrow::Status& status);

 private:
  int64_t limit_;
  int64_t actual_;
};

struct ARROW_FLIGHT_EXPORT FlightClientOptions {
  /// Root certificates to use for validating server
  /// certificates.
  std::string tls_root_certs;
  /// Override the hostname checked by TLS. Use with caution.
  std::string override_hostname;
  /// The client certificate to use if using Mutual TLS
  std::string cert_chain;
  /// The private key associated with the client certificate for Mutual TLS
  std::string private_key;
  /// A list of client middleware to apply.
  std::vector<std::shared_ptr<ClientMiddlewareFactory>> middleware;
  /// A soft limit on the number of bytes to write in a single
  ///     batch when sending Arrow data to a server.
  ///
  /// Used to help limit server memory consumption. Only enabled if
  /// positive. When enabled, FlightStreamWriter.Write* may yield a
  /// IOError with error detail FlightWriteSizeStatusDetail.
  int64_t write_size_limit_bytes = 0;

  /// Generic connection options, passed to the underlying
  ///     transport; interpretation is implementation-dependent.
  std::vector<std::pair<std::string, std::variant<int, std::string>>> generic_options;

  /// Use TLS without validating the server certificate. Use with caution.
  bool disable_server_verification = false;

  /// Get default options.
  static FlightClientOptions Defaults();
};

/// A RecordBatchReader exposing Flight metadata and cancel
/// operations.
class ARROW_FLIGHT_EXPORT FlightStreamReader : public MetadataRecordBatchReader {
 public:
  /// Try to cancel the call.
  virtual void Cancel() = 0;

  using MetadataRecordBatchReader::ToRecordBatches;
  /// Consume entire stream as a vector of record batches
  virtual arrow::Result<std::vector<std::shared_ptr<RecordBatch>>> ToRecordBatches(
      const StopToken& stop_token) = 0;

  using MetadataRecordBatchReader::ToTable;
  /// Consume entire stream as a Table
  arrow::Result<std::shared_ptr<Table>> ToTable(const StopToken& stop_token);
};

// Silence warning
// "non dll-interface class RecordBatchReader used as base for dll-interface class"
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable : 4275)
#endif

/// A RecordBatchWriter that also allows sending
/// application-defined metadata via the Flight protocol.
class ARROW_FLIGHT_EXPORT FlightStreamWriter : public MetadataRecordBatchWriter {
 public:
  /// Indicate that the application is done writing to this stream.
  ///
  /// The application may not write to this stream after calling
  /// this. This differs from closing the stream because this writer
  /// may represent only one half of a readable and writable stream.
  virtual Status DoneWriting() = 0;
};

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

/// A reader for application-specific metadata sent back to the
/// client during an upload.
class ARROW_FLIGHT_EXPORT FlightMetadataReader {
 public:
  virtual ~FlightMetadataReader();
  /// Read a message from the server.
  virtual Status ReadMetadata(std::shared_ptr<Buffer>* out) = 0;
};

/// Client class for Arrow Flight RPC services.
class ARROW_FLIGHT_EXPORT FlightClient {
 public:
  ~FlightClient();

  /// Connect to an unauthenticated flight service
  /// :param location: the URI
  /// :return: Arrow result with the created FlightClient, OK status may not indicate that
  /// the connection was successful
  static arrow::Result<std::unique_ptr<FlightClient>> Connect(const Location& location);

  /// Connect to an unauthenticated flight service
  /// :param location: the URI
  /// :param options: Other options for setting up the client
  /// :return: Arrow result with the created FlightClient, OK status may not indicate that
  /// the connection was successful
  static arrow::Result<std::unique_ptr<FlightClient>> Connect(
      const Location& location, const FlightClientOptions& options);

  /// Authenticate to the server using the given handler.
  /// :param options: Per-RPC options
  /// :param auth_handler: The authentication mechanism to use
  /// :return: Status OK if the client authenticated successfully
  Status Authenticate(const FlightCallOptions& options,
                      std::unique_ptr<ClientAuthHandler> auth_handler);

  /// Authenticate to the server using basic HTTP style authentication.
  /// :param options: Per-RPC options
  /// :param username: Username to use
  /// :param password: Password to use
  /// :return: Arrow result with bearer token and status OK if client authenticated
  /// successfully
  arrow::Result<std::pair<std::string, std::string>> AuthenticateBasicToken(
      const FlightCallOptions& options, const std::string& username,
      const std::string& password);

  /// Perform the indicated action, returning an iterator to the stream
  /// of results, if any
  /// :param options: Per-RPC options
  /// :param action: the action to be performed
  /// :return: Arrow result with an iterator object for reading the returned results
  arrow::Result<std::unique_ptr<ResultStream>> DoAction(const FlightCallOptions& options,
                                                        const Action& action);
  arrow::Result<std::unique_ptr<ResultStream>> DoAction(const Action& action) {
    return DoAction({}, action);
  }

  /// Perform the CancelFlightInfo action, returning a
  /// CancelFlightInfoResult
  ///
  /// :param options: Per-RPC options
  /// :param request: The CancelFlightInfoRequest
  /// :return: Arrow result with a CancelFlightInfoResult
  arrow::Result<CancelFlightInfoResult> CancelFlightInfo(
      const FlightCallOptions& options, const CancelFlightInfoRequest& request);
  arrow::Result<CancelFlightInfoResult> CancelFlightInfo(
      const CancelFlightInfoRequest& request) {
    return CancelFlightInfo({}, request);
  }

  /// Perform the RenewFlightEndpoint action, returning a renewed
  /// FlightEndpoint
  ///
  /// :param options: Per-RPC options
  /// :param request: The RenewFlightEndpointRequest
  /// :return: Arrow result with a renewed FlightEndpoint
  arrow::Result<FlightEndpoint> RenewFlightEndpoint(
      const FlightCallOptions& options, const RenewFlightEndpointRequest& request);
  arrow::Result<FlightEndpoint> RenewFlightEndpoint(
      const RenewFlightEndpointRequest& request) {
    return RenewFlightEndpoint({}, request);
  }

  /// Retrieve a list of available Action types
  /// :param options: Per-RPC options
  /// :return: Arrow result with the available actions
  arrow::Result<std::vector<ActionType>> ListActions(const FlightCallOptions& options);
  arrow::Result<std::vector<ActionType>> ListActions() {
    return ListActions(FlightCallOptions());
  }

  /// Request access plan for a single flight, which may be an existing
  /// dataset or a command to be executed
  /// :param options: Per-RPC options
  /// :param descriptor: the dataset request, whether a named dataset or
  /// command
  /// :return: Arrow result with the FlightInfo describing where to access the dataset
  arrow::Result<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const FlightCallOptions& options, const FlightDescriptor& descriptor);
  arrow::Result<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const FlightDescriptor& descriptor) {
    return GetFlightInfo({}, descriptor);
  }

  /// Asynchronous GetFlightInfo.
  /// :param options: Per-RPC options
  /// :param descriptor: the dataset request
  /// :param listener: Callbacks for response and RPC completion
  void GetFlightInfoAsync(const FlightCallOptions& options,
                          const FlightDescriptor& descriptor,
                          std::shared_ptr<AsyncListener<FlightInfo>> listener);
  void GetFlightInfoAsync(const FlightDescriptor& descriptor,
                          std::shared_ptr<AsyncListener<FlightInfo>> listener) {
    return GetFlightInfoAsync({}, descriptor, std::move(listener));
  }

  /// Asynchronous GetFlightInfo returning a Future.
  /// :param options: Per-RPC options
  /// :param descriptor: the dataset request
  arrow::Future<FlightInfo> GetFlightInfoAsync(const FlightCallOptions& options,
                                               const FlightDescriptor& descriptor);
  arrow::Future<FlightInfo> GetFlightInfoAsync(const FlightDescriptor& descriptor) {
    return GetFlightInfoAsync({}, descriptor);
  }

  /// Request and poll a long running query
  /// :param options: Per-RPC options
  /// :param descriptor: the dataset request or a descriptor returned by a
  /// prior PollFlightInfo call
  /// :return: Arrow result with the PollInfo describing the status of
  /// the requested query
  arrow::Result<std::unique_ptr<PollInfo>> PollFlightInfo(
      const FlightCallOptions& options, const FlightDescriptor& descriptor);
  arrow::Result<std::unique_ptr<PollInfo>> PollFlightInfo(
      const FlightDescriptor& descriptor) {
    return PollFlightInfo({}, descriptor);
  }

  /// Request schema for a single flight, which may be an existing
  /// dataset or a command to be executed
  /// :param options: Per-RPC options
  /// :param descriptor: the dataset request, whether a named dataset or
  /// command
  /// :return: Arrow result with the SchemaResult describing the dataset schema
  arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightCallOptions& options, const FlightDescriptor& descriptor);

  arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightDescriptor& descriptor) {
    return GetSchema({}, descriptor);
  }

  /// List all available flights known to the server
  /// :return: Arrow result with an iterator that returns a FlightInfo for each flight
  arrow::Result<std::unique_ptr<FlightListing>> ListFlights();

  /// List available flights given indicated filter criteria
  /// :param options: Per-RPC options
  /// :param criteria: the filter criteria (opaque)
  /// :return: Arrow result with an iterator that returns a FlightInfo for each flight
  arrow::Result<std::unique_ptr<FlightListing>> ListFlights(
      const FlightCallOptions& options, const Criteria& criteria);

  /// Given a flight ticket and schema, request to be sent the
  /// stream. Returns record batch stream reader
  /// :param options: Per-RPC options
  /// :param ticket: The flight ticket to use
  /// :return: Arrow result with the returned RecordBatchReader
  arrow::Result<std::unique_ptr<FlightStreamReader>> DoGet(
      const FlightCallOptions& options, const Ticket& ticket);
  arrow::Result<std::unique_ptr<FlightStreamReader>> DoGet(const Ticket& ticket) {
    return DoGet({}, ticket);
  }

  /// DoPut return value
  struct DoPutResult {
    /// a writer to write record batches to
    std::unique_ptr<FlightStreamWriter> writer;
    /// a reader for application metadata from the server
    std::unique_ptr<FlightMetadataReader> reader;
  };
  /// Upload data to a Flight described by the given
  /// descriptor. The caller must call Close() on the returned stream
  /// once they are done writing.
  ///
  /// The reader and writer are linked; closing the writer will also
  /// close the reader. Use \a DoneWriting to only close the write
  /// side of the channel.
  ///
  /// :param options: Per-RPC options
  /// :param descriptor: the descriptor of the stream
  /// :param schema: the schema for the data to upload
  /// :return: Arrow result with a DoPutResult struct holding a reader and a writer
  arrow::Result<DoPutResult> DoPut(const FlightCallOptions& options,
                                   const FlightDescriptor& descriptor,
                                   const std::shared_ptr<Schema>& schema);

  arrow::Result<DoPutResult> DoPut(const FlightDescriptor& descriptor,
                                   const std::shared_ptr<Schema>& schema) {
    return DoPut({}, descriptor, schema);
  }

  struct DoExchangeResult {
    std::unique_ptr<FlightStreamWriter> writer;
    std::unique_ptr<FlightStreamReader> reader;
  };
  arrow::Result<DoExchangeResult> DoExchange(const FlightCallOptions& options,
                                             const FlightDescriptor& descriptor);
  arrow::Result<DoExchangeResult> DoExchange(const FlightDescriptor& descriptor) {
    return DoExchange({}, descriptor);
  }

  /// Set server session option(s) by name/value. Sessions are generally
  /// persisted via HTTP cookies.
  /// :param options: Per-RPC options
  /// :param request: The server session options to set
  ::arrow::Result<SetSessionOptionsResult> SetSessionOptions(
      const FlightCallOptions& options, const SetSessionOptionsRequest& request);

  /// Get the current server session options. The session is generally
  /// accessed via an HTTP cookie.
  /// :param options: Per-RPC options
  /// :param request: The (empty) GetSessionOptions request object.
  ::arrow::Result<GetSessionOptionsResult> GetSessionOptions(
      const FlightCallOptions& options, const GetSessionOptionsRequest& request);

  /// Close/invalidate the current server session. The session is generally
  /// accessed via an HTTP cookie.
  /// :param options: Per-RPC options
  /// :param request: The (empty) CloseSession request object.
  ::arrow::Result<CloseSessionResult> CloseSession(const FlightCallOptions& options,
                                                   const CloseSessionRequest& request);

  /// Explicitly shut down and clean up the client.
  ///
  /// For backwards compatibility, this will be implicitly called by
  /// the destructor if not already called, but this gives the
  /// application no chance to handle errors, so it is recommended to
  /// explicitly close the client.
  ///
  /// ```{versionadded} 8.0.0
  /// ```
  Status Close();

  /// Whether this client supports asynchronous methods.
  bool supports_async() const;

  /// Check whether this client supports asynchronous methods.
  ///
  /// This is like supports_async(), except that a detailed error message
  /// is returned if async support is not available.  If async support is
  /// available, this function returns successfully.
  Status CheckAsyncSupport() const;

 private:
  FlightClient();
  Status CheckOpen() const;
  std::unique_ptr<internal::ClientTransport> transport_;
  bool closed_;
  int64_t write_size_limit_bytes_;
};

}  // namespace flight
}  // namespace arrow
