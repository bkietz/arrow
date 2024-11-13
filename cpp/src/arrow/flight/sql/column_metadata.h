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

#include <string>

#include "arrow/flight/sql/visibility.h"
#include "arrow/util/key_value_metadata.h"

namespace arrow {
namespace flight {
namespace sql {

/// Helper class to set column metadata.
class ARROW_FLIGHT_SQL_EXPORT ColumnMetadata {
 private:
  std::shared_ptr<const arrow::KeyValueMetadata> metadata_map_;

 public:
  class ColumnMetadataBuilder;

  explicit ColumnMetadata(std::shared_ptr<const arrow::KeyValueMetadata> metadata_map);

  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kCatalogName;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kSchemaName;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kTableName;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kTypeName;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kPrecision;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kScale;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kIsAutoIncrement;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kIsCaseSensitive;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kIsReadOnly;
  /// Constant variable to hold the value of the key that
  ///        will be used in the KeyValueMetadata class.
  static const char* kIsSearchable;

  /// Static initializer.
  static ColumnMetadataBuilder Builder();

  ///  Return the catalog name set in the KeyValueMetadata.
  /// :return: The catalog name.
  arrow::Result<std::string> GetCatalogName() const;

  ///  Return the schema name set in the KeyValueMetadata.
  /// :return: The schema name.
  arrow::Result<std::string> GetSchemaName() const;

  ///  Return the table name set in the KeyValueMetadata.
  /// :return: The table name.
  arrow::Result<std::string> GetTableName() const;

  ///  Return the data source-specific name for the data type of the column.
  /// :return: The type name.
  arrow::Result<std::string> GetTypeName() const;

  ///  Return the precision set in the KeyValueMetadata.
  /// :return: The precision.
  arrow::Result<int32_t> GetPrecision() const;

  ///  Return the scale set in the KeyValueMetadata.
  /// :return: The scale.
  arrow::Result<int32_t> GetScale() const;

  ///  Return the IsAutoIncrement set in the KeyValueMetadata.
  /// :return: The IsAutoIncrement.
  arrow::Result<bool> GetIsAutoIncrement() const;

  ///  Return the IsCaseSensitive set in the KeyValueMetadata.
  /// :return: The IsCaseSensitive.
  arrow::Result<bool> GetIsCaseSensitive() const;

  ///  Return the IsReadOnly set in the KeyValueMetadata.
  /// :return: The IsReadOnly.
  arrow::Result<bool> GetIsReadOnly() const;

  ///  Return the IsSearchable set in the KeyValueMetadata.
  /// :return: The IsSearchable.
  arrow::Result<bool> GetIsSearchable() const;

  ///  Return the KeyValueMetadata.
  /// :return: The KeyValueMetadata.
  const std::shared_ptr<const arrow::KeyValueMetadata>& metadata_map() const;

  /// A builder class to construct the ColumnMetadata object.
  class ARROW_FLIGHT_SQL_EXPORT ColumnMetadataBuilder {
   public:
    friend class ColumnMetadata;

    /// Set the catalog name in the KeyValueMetadata object.
    /// :param catalog_name: The catalog name.
    /// :return:                 A ColumnMetadataBuilder.
    ColumnMetadataBuilder& CatalogName(const std::string& catalog_name);

    /// Set the schema_name in the KeyValueMetadata object.
    /// :param schema_name:  The schema_name.
    /// :return:                 A ColumnMetadataBuilder.
    ColumnMetadataBuilder& SchemaName(const std::string& schema_name);

    /// Set the table name in the KeyValueMetadata object.
    /// :param table_name:   The table name.
    /// :return:                 A ColumnMetadataBuilder.
    ColumnMetadataBuilder& TableName(const std::string& table_name);

    /// Set the type name in the KeyValueMetadata object.
    /// :param type_name:    The type name.
    /// :return:                 A ColumnMetadataBuilder.
    ColumnMetadataBuilder& TypeName(const std::string& type_name);

    /// Set the precision in the KeyValueMetadata object.
    /// :param precision:    The precision.
    /// :return:                 A ColumnMetadataBuilder.
    ColumnMetadataBuilder& Precision(int32_t precision);

    /// Set the scale in the KeyValueMetadata object.
    /// :param scale:  The scale.
    /// :return:           A ColumnMetadataBuilder.
    ColumnMetadataBuilder& Scale(int32_t scale);

    /// Set the IsAutoIncrement in the KeyValueMetadata object.
    /// :param is_auto_increment:  The IsAutoIncrement.
    /// :return:                       A ColumnMetadataBuilder.
    ColumnMetadataBuilder& IsAutoIncrement(bool is_auto_increment);

    /// Set the IsCaseSensitive in the KeyValueMetadata object.
    /// :param is_case_sensitive: The IsCaseSensitive.
    /// :return:                      A ColumnMetadataBuilder.
    ColumnMetadataBuilder& IsCaseSensitive(bool is_case_sensitive);

    /// Set the IsReadOnly in the KeyValueMetadata object.
    /// :param is_read_only:   The IsReadOnly.
    /// :return:                   A ColumnMetadataBuilder.
    ColumnMetadataBuilder& IsReadOnly(bool is_read_only);

    /// Set the IsSearchable in the KeyValueMetadata object.
    /// :param is_searchable: The IsSearchable.
    /// :return:                  A ColumnMetadataBuilder.
    ColumnMetadataBuilder& IsSearchable(bool is_searchable);

    ColumnMetadata Build() const;

   private:
    std::shared_ptr<arrow::KeyValueMetadata> metadata_map_;

    /// Default constructor.
    ColumnMetadataBuilder();
  };
};
}  // namespace sql
}  // namespace flight
}  // namespace arrow
