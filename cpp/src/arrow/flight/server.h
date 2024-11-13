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

// Interfaces to use for defining Flight RPC servers.

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/flight/server_auth.h"
#include "arrow/flight/type_fwd.h"
#include "arrow/flight/types.h"       // IWYU pragma: keep
#include "arrow/flight/visibility.h"  // IWYU pragma: keep
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/options.h"
#include "arrow/record_batch.h"

namespace arrow {

class Schema;
class Status;

namespace flight {

/// Interface that produces a sequence of IPC payloads to be sent in
/// FlightData protobuf messages
class ARROW_FLIGHT_EXPORT FlightDataStream {
 public:
  virtual ~FlightDataStream();

  virtual std::shared_ptr<Schema> schema() = 0;

  /// Compute FlightPayload containing serialized RecordBatch schema
  virtual arrow::Result<FlightPayload> GetSchemaPayload() = 0;

  // When the stream is completed, the last payload written will have null
  // metadata
  virtual arrow::Result<FlightPayload> Next() = 0;

  virtual Status Close();
};

/// A basic implementation of FlightDataStream that will provide
/// a sequence of FlightData messages to be written to a stream
class ARROW_FLIGHT_EXPORT RecordBatchStream : public FlightDataStream {
 public:
  /// :param reader: produces a sequence of record batches
  /// :param options: IPC options for writing
  explicit RecordBatchStream(
      const std::shared_ptr<RecordBatchReader>& reader,
      const ipc::IpcWriteOptions& options = ipc::IpcWriteOptions::Defaults());
  ~RecordBatchStream() override;

  // inherit deprecated API
  using FlightDataStream::GetSchemaPayload;
  using FlightDataStream::Next;

  std::shared_ptr<Schema> schema() override;
  arrow::Result<FlightPayload> GetSchemaPayload() override;

  arrow::Result<FlightPayload> Next() override;
  Status Close() override;

 private:
  class RecordBatchStreamImpl;
  std::unique_ptr<RecordBatchStreamImpl> impl_;
};

/// A reader for IPC payloads uploaded by a client. Also allows
/// reading application-defined metadata via the Flight protocol.
class ARROW_FLIGHT_EXPORT FlightMessageReader : public MetadataRecordBatchReader {
 public:
  /// Get the descriptor for this upload.
  virtual const FlightDescriptor& descriptor() const = 0;
};

/// A writer for application-specific metadata sent back to the
/// client during an upload.
class ARROW_FLIGHT_EXPORT FlightMetadataWriter {
 public:
  virtual ~FlightMetadataWriter();
  /// Send a message to the client.
  virtual Status WriteMetadata(const Buffer& app_metadata) = 0;
};

/// A writer for IPC payloads to a client. Also allows sending
/// application-defined metadata via the Flight protocol.
///
/// This class offers more control compared to FlightDataStream,
/// including the option to write metadata without data and the
/// ability to interleave reading and writing.
class ARROW_FLIGHT_EXPORT FlightMessageWriter : public MetadataRecordBatchWriter {
 public:
  virtual ~FlightMessageWriter() = default;
};

/// Call state/contextual data.
class ARROW_FLIGHT_EXPORT ServerCallContext {
 public:
  virtual ~ServerCallContext() = default;
  /// The name of the authenticated peer (may be the empty string)
  virtual const std::string& peer_identity() const = 0;
  /// The peer address (not validated)
  virtual const std::string& peer() const = 0;
  /// Add a response header.  This is only valid before the server
  /// starts sending the response; generally this isn't an issue unless you
  /// are implementing FlightDataStream, ResultStream, or similar interfaces
  /// yourself, or during a DoExchange or DoPut.
  virtual void AddHeader(const std::string& key, const std::string& value) const = 0;
  /// Add a response trailer.  This is only valid before the server
  /// sends the final status; generally this isn't an issue unless your RPC
  /// handler launches a thread or similar.
  virtual void AddTrailer(const std::string& key, const std::string& value) const = 0;
  /// Look up a middleware by key. Do not maintain a reference
  /// to the object beyond the request body.
  /// :return: The middleware, or nullptr if not found.
  virtual ServerMiddleware* GetMiddleware(const std::string& key) const = 0;
  /// Check if the current RPC has been cancelled (by the client, by
  /// a network error, etc.).
  virtual bool is_cancelled() const = 0;
  /// The headers sent by the client for this call.
  virtual const CallHeaders& incoming_headers() const = 0;
};

class ARROW_FLIGHT_EXPORT FlightServerOptions {
 public:
  explicit FlightServerOptions(const Location& location_);

  ~FlightServerOptions();

  /// The host & port (or domain socket path) to listen on.
  /// Use port 0 to bind to an available port.
  Location location;
  /// The authentication handler to use.
  std::shared_ptr<ServerAuthHandler> auth_handler;
  /// A list of TLS certificate+key pairs to use.
  std::vector<CertKeyPair> tls_certificates;
  /// Enable mTLS and require that the client present a certificate.
  bool verify_client;
  /// If using mTLS, the PEM-encoded root certificate to use.
  std::string root_certificates;
  /// A list of server middleware to apply, along with a key to
  /// identify them by.
  ///
  /// Middleware are always applied in the order provided. Duplicate
  /// keys are an error.
  std::vector<std::pair<std::string, std::shared_ptr<ServerMiddlewareFactory>>>
      middleware;

  /// An optional memory manager to control where to allocate incoming data.
  std::shared_ptr<MemoryManager> memory_manager;

  /// A Flight implementation-specific callback to customize
  /// transport-specific options.
  ///
  /// Not guaranteed to be called. The type of the parameter is
  /// specific to the Flight implementation. Users should take care to
  /// link to the same transport implementation as Flight to avoid
  /// runtime problems. See "Using Arrow C++ in your own project" in
  /// the documentation for more details.
  std::function<void(void*)> builder_hook;
};

/// Skeleton RPC server implementation which can be used to create
/// custom servers by implementing its abstract methods
class ARROW_FLIGHT_EXPORT FlightServerBase {
 public:
  FlightServerBase();
  virtual ~FlightServerBase();

  // Lifecycle methods.

  /// Initialize a Flight server listening at the given location.
  /// This method must be called before any other method.
  /// :param options: The configuration for this server.
  Status Init(const FlightServerOptions& options);

  /// Get the port that the Flight server is listening on.
  /// This method must only be called after Init().  Will return a
  /// non-positive value if no port exists (e.g. when listening on a
  /// domain socket).
  int port() const;

  /// Get the address that the Flight server is listening on.
  /// This method must only be called after Init().
  Location location() const;

  /// Set the server to stop when receiving any of the given signal
  /// numbers.
  /// This method must be called before Serve().
  Status SetShutdownOnSignals(const std::vector<int> sigs);

  /// Start serving.
  /// This method blocks until the server shuts down.
  ///
  /// The server will start to shut down when either Shutdown() is called
  /// or one of the signals registered in SetShutdownOnSignals() is received.
  Status Serve();

  /// Query whether Serve() was interrupted by a signal.
  /// This method must be called after Serve() has returned.
  ///
  /// :return: int the signal number that interrupted Serve(), if any, otherwise 0
  int GotSignal() const;

  /// Shut down the server, blocking until current requests finish.
  ///
  /// Can be called from a signal handler or another thread while Serve()
  /// blocks. Optionally a deadline can be set. Once the deadline expires
  /// server will wait until remaining running calls complete.
  ///
  /// Should only be called once.
  Status Shutdown(const std::chrono::system_clock::time_point* deadline = NULLPTR);

  /// Block until server shuts down with Shutdown.
  ///
  /// Does not respond to signals like Serve().
  Status Wait();

  // Implement these methods to create your own server. The default
  // implementations will return a not-implemented result to the client

  /// Retrieve a list of available fields given an optional opaque
  /// criteria
  /// :param context: The call context.
  /// :param criteria: may be null
  /// :param listings[out]: the returned listings iterator
  /// :return: Status
  virtual Status ListFlights(const ServerCallContext& context, const Criteria* criteria,
                             std::unique_ptr<FlightListing>* listings);

  /// Retrieve the schema and an access plan for the indicated
  /// descriptor
  /// :param context: The call context.
  /// :param request: the dataset request, whether a named dataset or command
  /// :param info[out]: the returned flight info provider
  /// :return: Status
  virtual Status GetFlightInfo(const ServerCallContext& context,
                               const FlightDescriptor& request,
                               std::unique_ptr<FlightInfo>* info);

  /// Retrieve the current status of the target query
  /// :param context: The call context.
  /// :param request: the dataset request or a descriptor returned by a
  /// prior PollFlightInfo call
  /// :param info[out]: the returned retry info provider
  /// :return: Status
  virtual Status PollFlightInfo(const ServerCallContext& context,
                                const FlightDescriptor& request,
                                std::unique_ptr<PollInfo>* info);

  /// Retrieve the schema for the indicated descriptor
  /// :param context: The call context.
  /// :param request: the dataset request, whether a named dataset or command
  /// :param schema[out]: the returned flight schema provider
  /// :return: Status
  virtual Status GetSchema(const ServerCallContext& context,
                           const FlightDescriptor& request,
                           std::unique_ptr<SchemaResult>* schema);

  /// Get a stream of IPC payloads to put on the wire
  /// :param context: The call context.
  /// :param request: an opaque ticket
  /// :param stream[out]: the returned stream provider
  /// :return: Status
  virtual Status DoGet(const ServerCallContext& context, const Ticket& request,
                       std::unique_ptr<FlightDataStream>* stream);

  /// Process a stream of IPC payloads sent from a client
  /// :param context: The call context.
  /// :param reader: a sequence of uploaded record batches
  /// :param writer: send metadata back to the client
  /// :return: Status
  virtual Status DoPut(const ServerCallContext& context,
                       std::unique_ptr<FlightMessageReader> reader,
                       std::unique_ptr<FlightMetadataWriter> writer);

  /// Process a bidirectional stream of IPC payloads
  /// :param context: The call context.
  /// :param reader: a sequence of uploaded record batches
  /// :param writer: send data back to the client
  /// :return: Status
  virtual Status DoExchange(const ServerCallContext& context,
                            std::unique_ptr<FlightMessageReader> reader,
                            std::unique_ptr<FlightMessageWriter> writer);

  /// Execute an action, return stream of zero or more results
  /// :param context: The call context.
  /// :param action: the action to execute, with type and body
  /// :param result[out]: the result iterator
  /// :return: Status
  virtual Status DoAction(const ServerCallContext& context, const Action& action,
                          std::unique_ptr<ResultStream>* result);

  /// Retrieve the list of available actions
  /// :param context: The call context.
  /// :param actions[out]: a vector of available action types
  /// :return: Status
  virtual Status ListActions(const ServerCallContext& context,
                             std::vector<ActionType>* actions);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace flight
}  // namespace arrow
