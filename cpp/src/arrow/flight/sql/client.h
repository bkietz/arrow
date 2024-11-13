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

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "arrow/flight/client.h"
#include "arrow/flight/sql/types.h"
#include "arrow/flight/sql/visibility.h"
#include "arrow/flight/types.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace flight {
namespace sql {

class PreparedStatement;
class Transaction;
class Savepoint;

/// A default transaction to use when the default behavior
///   (auto-commit) is desired.
ARROW_FLIGHT_SQL_EXPORT
const Transaction& no_transaction();

/// Flight client with Flight SQL semantics.
///
/// Wraps a Flight client to provide the Flight SQL RPC calls.
class ARROW_FLIGHT_SQL_EXPORT FlightSqlClient {
  friend class PreparedStatement;

 private:
  std::shared_ptr<FlightClient> impl_;

 public:
  explicit FlightSqlClient(std::shared_ptr<FlightClient> client);

  virtual ~FlightSqlClient() = default;

  /// Execute a SQL query on the server.
  /// :param options:      RPC-layer hints for this call.
  /// :param query:        The UTF8-encoded SQL query to be executed.
  /// :param transaction:  A transaction to associate this query with.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> Execute(
      const FlightCallOptions& options, const std::string& query,
      const Transaction& transaction = no_transaction());

  /// Execute a Substrait plan that returns a result set on the server.
  /// :param options:      RPC-layer hints for this call.
  /// :param plan:         The plan to be executed.
  /// :param transaction:  A transaction to associate this query with.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> ExecuteSubstrait(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// Get the result set schema from the server.
  arrow::Result<std::unique_ptr<SchemaResult>> GetExecuteSchema(
      const FlightCallOptions& options, const std::string& query,
      const Transaction& transaction = no_transaction());

  /// Get the result set schema from the server.
  arrow::Result<std::unique_ptr<SchemaResult>> GetExecuteSubstraitSchema(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// Execute an update query on the server.
  /// :param options:      RPC-layer hints for this call.
  /// :param query:        The UTF8-encoded SQL query to be executed.
  /// :param transaction:  A transaction to associate this query with.
  /// :return: The quantity of rows affected by the operation.
  arrow::Result<int64_t> ExecuteUpdate(const FlightCallOptions& options,
                                       const std::string& query,
                                       const Transaction& transaction = no_transaction());

  /// Execute a Substrait plan that does not return a result set on the server.
  /// :param options:      RPC-layer hints for this call.
  /// :param plan:         The plan to be executed.
  /// :param transaction:  A transaction to associate this query with.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<int64_t> ExecuteSubstraitUpdate(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// Execute a bulk ingestion to the server.
  /// :param options:                   RPC-layer hints for this call.
  /// :param reader:                    The records to ingest.
  /// :param table_definition_options:  The behavior for handling the table definition.
  /// :param table:                     The destination table to load into.
  /// :param schema:                    The DB schema of the destination table.
  /// :param catalog:                   The catalog of the destination table.
  /// :param temporary:                 Use a temporary table.
  /// :param transaction:               Ingest as part of this transaction.
  /// :param ingest_options:            Additional, backend-specific options.
  /// :return: The number of rows ingested to the server.
  arrow::Result<int64_t> ExecuteIngest(
      const FlightCallOptions& options, const std::shared_ptr<RecordBatchReader>& reader,
      const TableDefinitionOptions& table_definition_options, const std::string& table,
      const std::optional<std::string>& schema, const std::optional<std::string>& catalog,
      const bool temporary, const Transaction& transaction = no_transaction(),
      const std::unordered_map<std::string, std::string>& ingest_options = {});

  /// Request a list of catalogs.
  /// :param options:      RPC-layer hints for this call.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetCatalogs(
      const FlightCallOptions& options);

  /// Get the catalogs schema from the server (should be
  ///   identical to SqlSchema::GetCatalogsSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetCatalogsSchema(
      const FlightCallOptions& options);

  /// Request a list of database schemas.
  /// :param options:                   RPC-layer hints for this call.
  /// :param catalog:                   The catalog.
  /// :param db_schema_filter_pattern:  The schema filter pattern.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetDbSchemas(
      const FlightCallOptions& options, const std::string* catalog,
      const std::string* db_schema_filter_pattern);

  /// Get the database schemas schema from the server (should be
  ///   identical to SqlSchema::GetDbSchemasSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetDbSchemasSchema(
      const FlightCallOptions& options);

  /// Given a flight ticket and schema, request to be sent the
  /// stream. Returns record batch stream reader
  /// :param options: Per-RPC options
  /// :param ticket: The flight ticket to use
  /// :return: The returned RecordBatchReader
  virtual arrow::Result<std::unique_ptr<FlightStreamReader>> DoGet(
      const FlightCallOptions& options, const Ticket& ticket);

  /// Request a list of tables.
  /// :param options:                   RPC-layer hints for this call.
  /// :param catalog:                   The catalog.
  /// :param db_schema_filter_pattern:  The schema filter pattern.
  /// :param table_filter_pattern:      The table filter pattern.
  /// :param include_schema:            True to include the schema upon return,
  ///                                      false to not include the schema.
  /// :param table_types:               The table types to include.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetTables(
      const FlightCallOptions& options, const std::string* catalog,
      const std::string* db_schema_filter_pattern,
      const std::string* table_filter_pattern, bool include_schema,
      const std::vector<std::string>* table_types);

  /// Get the tables schema from the server (should be
  ///   identical to SqlSchema::GetTablesSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetTablesSchema(
      const FlightCallOptions& options, bool include_schema);

  /// Request the primary keys for a table.
  /// :param options:          RPC-layer hints for this call.
  /// :param table_ref:        The table reference.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetPrimaryKeys(
      const FlightCallOptions& options, const TableRef& table_ref);

  /// Get the primary keys schema from the server (should be
  ///   identical to SqlSchema::GetPrimaryKeysSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetPrimaryKeysSchema(
      const FlightCallOptions& options);

  /// Retrieves a description about the foreign key columns that reference the
  /// primary key columns of the given table.
  /// :param options:          RPC-layer hints for this call.
  /// :param table_ref:        The table reference.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetExportedKeys(
      const FlightCallOptions& options, const TableRef& table_ref);

  /// Get the exported keys schema from the server (should be
  ///   identical to SqlSchema::GetExportedKeysSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetExportedKeysSchema(
      const FlightCallOptions& options);

  /// Retrieves the foreign key columns for the given table.
  /// :param options:          RPC-layer hints for this call.
  /// :param table_ref:        The table reference.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetImportedKeys(
      const FlightCallOptions& options, const TableRef& table_ref);

  /// Get the imported keys schema from the server (should be
  ///   identical to SqlSchema::GetImportedKeysSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetImportedKeysSchema(
      const FlightCallOptions& options);

  /// Retrieves a description of the foreign key columns in the given foreign key
  ///        table that reference the primary key or the columns representing a unique
  ///        constraint of the parent table (could be the same or a different table).
  /// :param options:        RPC-layer hints for this call.
  /// :param pk_table_ref:   The table reference that exports the key.
  /// :param fk_table_ref:   The table reference that imports the key.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetCrossReference(
      const FlightCallOptions& options, const TableRef& pk_table_ref,
      const TableRef& fk_table_ref);

  /// Get the cross reference schema from the server (should be
  ///   identical to SqlSchema::GetCrossReferenceSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetCrossReferenceSchema(
      const FlightCallOptions& options);

  /// Request a list of table types.
  /// :param options:          RPC-layer hints for this call.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetTableTypes(
      const FlightCallOptions& options);

  /// Get the table types schema from the server (should be
  ///   identical to SqlSchema::GetTableTypesSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetTableTypesSchema(
      const FlightCallOptions& options);

  /// Request the information about all the data types supported.
  /// :param options:          RPC-layer hints for this call.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetXdbcTypeInfo(
      const FlightCallOptions& options);

  /// Request the information about all the data types supported.
  /// :param options:          RPC-layer hints for this call.
  /// :param data_type:        The data type to search for as filtering.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetXdbcTypeInfo(
      const FlightCallOptions& options, int data_type);

  /// Get the type info schema from the server (should be
  ///   identical to SqlSchema::GetXdbcTypeInfoSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetXdbcTypeInfoSchema(
      const FlightCallOptions& options);

  /// Request a list of SQL information.
  /// :param options: RPC-layer hints for this call.
  /// :param sql_info: the SQL info required.
  /// :return: The FlightInfo describing where to access the dataset.
  arrow::Result<std::unique_ptr<FlightInfo>> GetSqlInfo(const FlightCallOptions& options,
                                                        const std::vector<int>& sql_info);

  /// Get the SQL information schema from the server (should be
  ///   identical to SqlSchema::GetSqlInfoSchema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetSqlInfoSchema(
      const FlightCallOptions& options);

  /// Create a prepared statement object.
  /// :param options:              RPC-layer hints for this call.
  /// :param query:                The query that will be executed.
  /// :param transaction:          A transaction to associate this query with.
  /// :return: The created prepared statement.
  arrow::Result<std::shared_ptr<PreparedStatement>> Prepare(
      const FlightCallOptions& options, const std::string& query,
      const Transaction& transaction = no_transaction());

  /// Create a prepared statement object.
  /// :param options:              RPC-layer hints for this call.
  /// :param plan:                 The Substrait plan that will be executed.
  /// :param transaction:          A transaction to associate this query with.
  /// :return: The created prepared statement.
  arrow::Result<std::shared_ptr<PreparedStatement>> PrepareSubstrait(
      const FlightCallOptions& options, const SubstraitPlan& plan,
      const Transaction& transaction = no_transaction());

  /// Call the underlying Flight client's GetFlightInfo.
  virtual arrow::Result<std::unique_ptr<FlightInfo>> GetFlightInfo(
      const FlightCallOptions& options, const FlightDescriptor& descriptor) {
    return impl_->GetFlightInfo(options, descriptor);
  }

  /// Call the underlying Flight client's GetSchema.
  virtual arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightCallOptions& options, const FlightDescriptor& descriptor) {
    return impl_->GetSchema(options, descriptor);
  }

  /// Begin a new transaction.
  ::arrow::Result<Transaction> BeginTransaction(const FlightCallOptions& options);

  /// Create a new savepoint within a transaction.
  /// :param options:      RPC-layer hints for this call.
  /// :param transaction:  The parent transaction.
  /// :param name:         A friendly name for the savepoint.
  ::arrow::Result<Savepoint> BeginSavepoint(const FlightCallOptions& options,
                                            const Transaction& transaction,
                                            const std::string& name);

  /// Commit a transaction.
  ///
  /// After this, the transaction and all associated savepoints will
  /// be invalidated.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param transaction:  The transaction.
  Status Commit(const FlightCallOptions& options, const Transaction& transaction);

  /// Release a savepoint.
  ///
  /// After this, the savepoint (and all savepoints created after it) will be invalidated.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param savepoint:    The savepoint.
  Status Release(const FlightCallOptions& options, const Savepoint& savepoint);

  /// Rollback a transaction.
  ///
  /// After this, the transaction and all associated savepoints will be invalidated.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param transaction:  The transaction.
  Status Rollback(const FlightCallOptions& options, const Transaction& transaction);

  /// Rollback a savepoint.
  ///
  /// After this, the savepoint will still be valid, but all
  /// savepoints created after it will be invalidated.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param savepoint:    The savepoint.
  Status Rollback(const FlightCallOptions& options, const Savepoint& savepoint);

  /// Explicitly cancel a FlightInfo.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param request:      The CancelFlightInfoRequest.
  /// :return: Arrow result with a canceled result.
  ::arrow::Result<CancelFlightInfoResult> CancelFlightInfo(
      const FlightCallOptions& options, const CancelFlightInfoRequest& request) {
    return impl_->CancelFlightInfo(options, request);
  }

  /// Explicitly cancel a query.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param info:         The FlightInfo of the query to cancel.
  ///
  /// \deprecated Deprecated since 13.0.0. Use CancelFlightInfo()
  /// instead. If you can assume that a server requires 13.0.0 or
  /// later, you can always use CancelFlightInfo(). Otherwise, you may
  /// need to use CancelQuery() and/or CancelFlightInfo().
  ARROW_DEPRECATED(
      "Deprecated in 13.0.0. Use CancelFlightInfo() instead. "
      "If you can assume that a server requires 13.0.0 or later, "
      "you can always use CancelFLightInfo(). Otherwise, you "
      "may need to use CancelQuery() and/or CancelFlightInfo()")
  ::arrow::Result<CancelResult> CancelQuery(const FlightCallOptions& options,
                                            const FlightInfo& info);

  /// Sets session options.
  ///
  /// :param options:            RPC-layer hints for this call.
  /// :param request:            The session options to set.
  ::arrow::Result<SetSessionOptionsResult> SetSessionOptions(
      const FlightCallOptions& options, const SetSessionOptionsRequest& request) {
    return impl_->SetSessionOptions(options, request);
  }

  /// Gets current session options.
  ///
  /// :param options:            RPC-layer hints for this call.
  /// :param request:            The (empty) GetSessionOptions request object.
  ::arrow::Result<GetSessionOptionsResult> GetSessionOptions(
      const FlightCallOptions& options, const GetSessionOptionsRequest& request) {
    return impl_->GetSessionOptions(options, request);
  }

  /// Explicitly closes the session if applicable.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param request:      The (empty) CloseSession request object.
  ::arrow::Result<CloseSessionResult> CloseSession(const FlightCallOptions& options,
                                                   const CloseSessionRequest& request) {
    return impl_->CloseSession(options, request);
  }

  /// Extends the expiration of a FlightEndpoint.
  ///
  /// :param options:      RPC-layer hints for this call.
  /// :param request:      The RenewFlightEndpointRequest.
  /// :return: Arrow result with a renewed FlightEndpoint
  ::arrow::Result<FlightEndpoint> RenewFlightEndpoint(
      const FlightCallOptions& options, const RenewFlightEndpointRequest& request) {
    return impl_->RenewFlightEndpoint(options, request);
  }

  /// Explicitly shut down and clean up the client.
  Status Close();

  /// Wrapper around FlightClient::DoGet.
  ///
  /// \internal
  /// Don't call this directly.
  /// \endinternal
  virtual ::arrow::Result<FlightClient::DoPutResult> DoPut(
      const FlightCallOptions& options, const FlightDescriptor& descriptor,
      const std::shared_ptr<Schema>& schema) {
    return impl_->DoPut(options, descriptor, schema);
  }

  /// Wrapper around FlightClient::DoPut. Don't call this directly.
  ///
  /// \internal
  /// Don't call this directly.
  /// \endinternal
  virtual ::arrow::Result<std::unique_ptr<ResultStream>> DoAction(
      const FlightCallOptions& options, const Action& action) {
    return impl_->DoAction(options, action);
  }
};

/// A prepared statement that can be executed.
class ARROW_FLIGHT_SQL_EXPORT PreparedStatement {
 public:
  /// Create a new prepared statement. However, applications
  /// should generally use FlightSqlClient::Prepare.
  ///
  /// :param client:                Client object used to make the RPC requests.
  /// :param handle:                Handle for this prepared statement.
  /// :param dataset_schema:        Schema of the resulting dataset.
  /// :param parameter_schema:      Schema of the parameters (if any).
  PreparedStatement(FlightSqlClient* client, std::string handle,
                    std::shared_ptr<Schema> dataset_schema,
                    std::shared_ptr<Schema> parameter_schema);

  /// Default destructor for the PreparedStatement class.
  /// The destructor will call the Close method from the class in order,
  /// to send a request to close the PreparedStatement.
  /// NOTE: It is best to explicitly close the PreparedStatement, otherwise
  /// errors can't be caught.
  ~PreparedStatement();

  /// Create a PreparedStatement by parsing the server response.
  static arrow::Result<std::shared_ptr<PreparedStatement>> ParseResponse(
      FlightSqlClient* client, std::unique_ptr<ResultStream> results);

  /// Executes the prepared statement query on the server.
  /// :return: A FlightInfo object representing the stream(s) to fetch.
  arrow::Result<std::unique_ptr<FlightInfo>> Execute(
      const FlightCallOptions& options = {});

  /// Executes the prepared statement update query on the server.
  /// :return: The number of rows affected.
  arrow::Result<int64_t> ExecuteUpdate(const FlightCallOptions& options = {});

  /// Retrieve the parameter schema from the query.
  /// :return: The parameter schema from the query.
  const std::shared_ptr<Schema>& parameter_schema() const;

  /// Retrieve the ResultSet schema from the query.
  /// :return: The ResultSet schema from the query.
  const std::shared_ptr<Schema>& dataset_schema() const;

  /// Set a RecordBatch that contains the parameters that will be bound.
  Status SetParameters(std::shared_ptr<RecordBatch> parameter_binding);

  /// Set a RecordBatchReader that contains the parameters that will be bound.
  Status SetParameters(std::shared_ptr<RecordBatchReader> parameter_binding);

  /// Re-request the result set schema from the server (should
  ///   be identical to dataset_schema).
  arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightCallOptions& options = {});

  /// Close the prepared statement so the server can free up any resources.
  ///
  /// After this, the prepared statement may not be used anymore.
  Status Close(const FlightCallOptions& options = {});

  /// Check if the prepared statement is closed.
  /// :return: The state of the prepared statement.
  bool IsClosed() const;

 private:
  FlightSqlClient* client_;
  std::string handle_;
  std::shared_ptr<Schema> dataset_schema_;
  std::shared_ptr<Schema> parameter_schema_;
  std::shared_ptr<RecordBatchReader> parameter_binding_;
  bool is_closed_;
};

/// A handle for a server-side savepoint.
class ARROW_FLIGHT_SQL_EXPORT Savepoint {
 public:
  explicit Savepoint(std::string savepoint_id) : savepoint_id_(std::move(savepoint_id)) {}
  const std::string& savepoint_id() const { return savepoint_id_; }
  bool is_valid() const { return !savepoint_id_.empty(); }

 private:
  std::string savepoint_id_;
};

/// A handle for a server-side transaction.
class ARROW_FLIGHT_SQL_EXPORT Transaction {
 public:
  explicit Transaction(std::string transaction_id)
      : transaction_id_(std::move(transaction_id)) {}
  const std::string& transaction_id() const { return transaction_id_; }
  bool is_valid() const { return !transaction_id_.empty(); }

 private:
  std::string transaction_id_;
};

}  // namespace sql
}  // namespace flight
}  // namespace arrow
