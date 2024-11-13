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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/array/data.h"
#include "arrow/device_allocation_type_set.h"
#include "arrow/scalar.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class ChunkedArray;
class RecordBatch;
class Table;

/// Variant type for various Arrow C++ data structures
struct ARROW_EXPORT Datum {
  /// The kind of datum stored
  enum Kind { NONE, SCALAR, ARRAY, CHUNKED_ARRAY, RECORD_BATCH, TABLE };

  /// A placeholder type to represent empty datum
  struct Empty {};

  /// Datums variants may have a length. This special value indicate that the
  /// current variant does not have a length.
  static constexpr int64_t kUnknownLength = -1;

  /// Storage of the actual datum.
  ///
  /// Note: For arrays, ArrayData is stored instead of Array for easier processing
  std::variant<Empty, std::shared_ptr<Scalar>, std::shared_ptr<ArrayData>,
               std::shared_ptr<ChunkedArray>, std::shared_ptr<RecordBatch>,
               std::shared_ptr<Table>>
      value;

  /// Empty datum, to be populated elsewhere
  Datum() = default;

  Datum(const Datum& other) = default;
  Datum& operator=(const Datum& other) = default;
  Datum(Datum&& other) = default;
  Datum& operator=(Datum&& other) = default;

  /// Construct from a Scalar
  Datum(std::shared_ptr<Scalar> value)  // NOLINT implicit conversion
      : value(std::move(value)) {}

  /// Construct from an ArrayData
  Datum(std::shared_ptr<ArrayData> value)  // NOLINT implicit conversion
      : value(std::move(value)) {}

  /// Construct from an ArrayData
  Datum(ArrayData arg)  // NOLINT implicit conversion
      : value(std::make_shared<ArrayData>(std::move(arg))) {}

  /// Construct from an Array
  Datum(const Array& value);  // NOLINT implicit conversion

  /// Construct from an Array
  Datum(const std::shared_ptr<Array>& value);  // NOLINT implicit conversion

  /// Construct from a ChunkedArray
  Datum(std::shared_ptr<ChunkedArray> value);  // NOLINT implicit conversion

  /// Construct from a RecordBatch
  Datum(std::shared_ptr<RecordBatch> value);  // NOLINT implicit conversion

  /// Construct from a Table
  Datum(std::shared_ptr<Table> value);  // NOLINT implicit conversion

  /// Construct from a ChunkedArray.
  ///
  /// This can be expensive, prefer the shared_ptr<ChunkedArray> constructor
  explicit Datum(const ChunkedArray& value);

  /// Construct from a RecordBatch.
  ///
  /// This can be expensive, prefer the shared_ptr<RecordBatch> constructor
  explicit Datum(const RecordBatch& value);

  /// Construct from a Table.
  ///
  /// This can be expensive, prefer the shared_ptr<Table> constructor
  explicit Datum(const Table& value);

  /// Cast from concrete subtypes of Array or Scalar to Datum
  template <typename T, bool IsArray = std::is_base_of_v<Array, T>,
            bool IsScalar = std::is_base_of_v<Scalar, T>,
            typename = enable_if_t<IsArray || IsScalar>>
  Datum(std::shared_ptr<T> value)  // NOLINT implicit conversion
      : Datum(std::shared_ptr<typename std::conditional<IsArray, Array, Scalar>::type>(
            std::move(value))) {}

  /// Cast from concrete subtypes of Array or Scalar to Datum
  template <typename T, typename TV = typename std::remove_reference_t<T>,
            bool IsArray = std::is_base_of_v<Array, T>,
            bool IsScalar = std::is_base_of_v<Scalar, T>,
            typename = enable_if_t<IsArray || IsScalar>>
  Datum(T&& value)  // NOLINT implicit conversion
      : Datum(std::make_shared<TV>(std::forward<T>(value))) {}

  /// Copy from concrete subtypes of Scalar.
  ///
  /// The concrete scalar type must be copyable (not all of them are).
  template <typename T, typename = enable_if_t<std::is_base_of_v<Scalar, T>>>
  Datum(const T& value)  // NOLINT implicit conversion
      : Datum(std::make_shared<T>(value)) {}

  // Convenience constructors
  /// Convenience constructor storing a bool scalar.
  explicit Datum(bool value);
  /// Convenience constructor storing an int8 scalar.
  explicit Datum(int8_t value);
  /// Convenience constructor storing a uint8 scalar.
  explicit Datum(uint8_t value);
  /// Convenience constructor storing an int16 scalar.
  explicit Datum(int16_t value);
  /// Convenience constructor storing a uint16 scalar.
  explicit Datum(uint16_t value);
  /// Convenience constructor storing an int32 scalar.
  explicit Datum(int32_t value);
  /// Convenience constructor storing a uint32 scalar.
  explicit Datum(uint32_t value);
  /// Convenience constructor storing an int64 scalar.
  explicit Datum(int64_t value);
  /// Convenience constructor storing a uint64 scalar.
  explicit Datum(uint64_t value);
  /// Convenience constructor storing a float scalar.
  explicit Datum(float value);
  /// Convenience constructor storing a double scalar.
  explicit Datum(double value);
  /// Convenience constructor storing a string scalar.
  explicit Datum(std::string value);
  /// Convenience constructor storing a string scalar.
  explicit Datum(const char* value);

  /// Convenience constructor for a DurationScalar from std::chrono::duration
  template <template <typename, typename> class StdDuration, typename Rep,
            typename Period,
            typename = decltype(DurationScalar{StdDuration<Rep, Period>{}})>
  explicit Datum(StdDuration<Rep, Period> d) : Datum{DurationScalar(d)} {}

  /// The kind of data stored in Datum
  Datum::Kind kind() const {
    switch (this->value.index()) {
      case 0:
        return Datum::NONE;
      case 1:
        return Datum::SCALAR;
      case 2:
        return Datum::ARRAY;
      case 3:
        return Datum::CHUNKED_ARRAY;
      case 4:
        return Datum::RECORD_BATCH;
      case 5:
        return Datum::TABLE;
      default:
        return Datum::NONE;
    }
  }

  /// Retrieve the stored array as ArrayData
  ///
  /// Use make_array() if an Array is desired (which is more expensive).
  /// :throws std:::bad_variant_access if the datum is not an array
  const std::shared_ptr<ArrayData>& array() const {
    return std::get<std::shared_ptr<ArrayData>>(this->value);
  }

  /// The sum of bytes in each buffer referenced by the datum
  /// Note: Scalars report a size of 0
  /// ```{seealso}
  /// arrow::util::TotalBufferSize for caveats
  /// ```
  int64_t TotalBufferSize() const;

  /// Get the stored ArrayData in mutable form
  ///
  /// For internal use primarily. Keep in mind a shared_ptr<Datum> may have multiple
  /// owners.
  ArrayData* mutable_array() const { return this->array().get(); }

  /// Retrieve the stored array as Array
  /// :throws std:::bad_variant_access if the datum is not an array
  std::shared_ptr<Array> make_array() const;

  /// Retrieve the chunked array stored
  /// :throws std:::bad_variant_access if the datum is not a chunked array
  const std::shared_ptr<ChunkedArray>& chunked_array() const {
    return std::get<std::shared_ptr<ChunkedArray>>(this->value);
  }

  /// Retrieve the record batch stored
  /// :throws std:::bad_variant_access if the datum is not a record batch
  const std::shared_ptr<RecordBatch>& record_batch() const {
    return std::get<std::shared_ptr<RecordBatch>>(this->value);
  }

  /// Retrieve the table stored
  /// :throws std:::bad_variant_access if the datum is not a table
  const std::shared_ptr<Table>& table() const {
    return std::get<std::shared_ptr<Table>>(this->value);
  }

  /// Retrieve the scalar stored
  /// :throws std:::bad_variant_access if the datum is not a scalar
  const std::shared_ptr<Scalar>& scalar() const {
    return std::get<std::shared_ptr<Scalar>>(this->value);
  }

  /// Retrieve the datum as its concrete array type
  /// :throws std:::bad_variant_access if the datum is not an array
  /// :param ExactType: the expected array type, may cause undefined behavior if it is not
  /// the type of the stored array
  template <typename ExactType>
  std::shared_ptr<ExactType> array_as() const {
    return internal::checked_pointer_cast<ExactType>(this->make_array());
  }

  /// Retrieve the datum as its concrete scalar type
  /// :throws std:::bad_variant_access if the datum is not a scalar
  /// :param ExactType: the expected scalar type, may cause undefined behavior if it is
  /// not the type of the stored scalar
  template <typename ExactType>
  const ExactType& scalar_as() const {
    return internal::checked_cast<const ExactType&>(*this->scalar());
  }

  /// True if Datum contains an array
  bool is_array() const { return this->kind() == Datum::ARRAY; }

  /// True if Datum contains a chunked array
  bool is_chunked_array() const { return this->kind() == Datum::CHUNKED_ARRAY; }

  /// True if Datum contains an array or a chunked array
  bool is_arraylike() const {
    return this->kind() == Datum::ARRAY || this->kind() == Datum::CHUNKED_ARRAY;
  }

  /// True if Datum contains a scalar
  bool is_scalar() const { return this->kind() == Datum::SCALAR; }

  /// True if Datum contains a scalar or array-like data
  bool is_value() const { return this->is_arraylike() || this->is_scalar(); }

  /// Return the null count.
  ///
  /// Only valid for scalar and array-like data.
  int64_t null_count() const;

  /// The value type of the variant, if any
  ///
  /// :return: nullptr if no type
  const std::shared_ptr<DataType>& type() const;

  /// The schema of the variant, if any
  ///
  /// :return: nullptr if no schema
  const std::shared_ptr<Schema>& schema() const;

  /// The value length of the variant, if any
  ///
  /// :return: kUnknownLength if no type
  int64_t length() const;

  /// The array chunks of the variant, if any
  ///
  /// :return: empty if not arraylike
  ArrayVector chunks() const;

  DeviceAllocationTypeSet device_types() const;

  /// True if the two data are equal
  bool Equals(const Datum& other) const;

  bool operator==(const Datum& other) const { return Equals(other); }
  bool operator!=(const Datum& other) const { return !Equals(other); }

  std::string ToString() const;
};

ARROW_EXPORT void PrintTo(const Datum&, std::ostream*);

ARROW_EXPORT std::string ToString(Datum::Kind kind);

}  // namespace arrow
