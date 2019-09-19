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

#include "arrow/json/parser.h"

#include <functional>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/json/rapidjson_defs.h"
#include "rapidjson/error/en.h"
#include "rapidjson/reader.h"

#include "arrow/array.h"
#include "arrow/buffer_builder.h"
#include "arrow/builder.h"
#include "arrow/memory_pool.h"
#include "arrow/type.h"
#include "arrow/util/logging.h"
#include "arrow/util/stl.h"
#include "arrow/util/string_view.h"
#include "arrow/util/trie.h"
#include "arrow/visitor_inline.h"

namespace arrow {
namespace json {

namespace rj = arrow::rapidjson;

using internal::BitsetStack;
using internal::checked_cast;
using internal::make_unique;
using util::string_view;

// Let JSON kind be bijective with unconverted type:
// null <-> null
// bool <-> bool
// number <-> dictionary(int32, utf8)
// string <-> dictionary(uint32, utf8)
// array <-> list
// object <-> struct

template <typename... T>
static Status ParseError(T&&... t) {
  return Status::Invalid("JSON parse error: ", std::forward<T>(t)...);
}

static Status KindChangeError(Kind::type from, Kind::type to) {
  return ParseError("A column changed from ", Kind::Name(from), " to ", Kind::Name(to));
}

const std::string& Kind::Name(Kind::type kind) {
  static const std::string names[] = {"null",   "boolean", "number",
                                      "string", "array",   "object"};

  return names[kind];
}

Result<Kind::type> Kind::FromType(const DataType& type) {
  Kind::type kind;
  struct {
    Status Visit(const NullType&) { return SetKind(Kind::kNull); }
    Status Visit(const BooleanType&) { return SetKind(Kind::kBoolean); }
    Status Visit(const NumberType&) { return SetKind(Kind::kNumber); }
    Status Visit(const TimeType&) { return SetKind(Kind::kNumber); }
    Status Visit(const DateType&) { return SetKind(Kind::kNumber); }
    Status Visit(const BinaryType&) { return SetKind(Kind::kString); }
    Status Visit(const FixedSizeBinaryType&) { return SetKind(Kind::kString); }
    Status Visit(const DictionaryType& dict_type) {
      // FIXME(bkietz) this is untested
      ARROW_ASSIGN_OR_RAISE(*kind_, Kind::FromType(*dict_type.value_type()));
      return Status::OK();
    }
    Status Visit(const ListType&) { return SetKind(Kind::kArray); }
    Status Visit(const StructType&) { return SetKind(Kind::kObject); }
    Status Visit(const DataType& not_impl) {
      return Status::NotImplemented("JSON parsing of ", not_impl);
    }
    Status SetKind(Kind::type kind) {
      *kind_ = kind;
      return Status::OK();
    }
    Kind::type* kind_;
  } visitor = {&kind};
  RETURN_NOT_OK(VisitTypeInline(type, &visitor));
  return kind;
}

Kind::type Kind::FromUnconvertedType(const DataType& unconverted_type) {
  struct Impl {
    Status Visit(const NullType&) { return SetKind(Kind::kNull); }

    Status Visit(const BooleanType&) { return SetKind(Kind::kBoolean); }

    Status Visit(const DictionaryType& dict_type) {
      if (dict_type.index_type()->id() == Type::INT32) {
        if (dict_type.value_type()->id() == Type::BINARY) {
          return SetKind(Kind::kNumber);
        }
        if (dict_type.value_type()->id() == Type::STRING) {
          return SetKind(Kind::kString);
        }
      }
      return Invalid(dict_type);
    }

    Status Visit(const ListType& list_type) {
      RETURN_NOT_OK(DoVisit(*list_type.value_type()));
      return SetKind(Kind::kArray);
    }

    Status Visit(const StructType& struct_type) {
      for (const auto& field : struct_type.children()) {
        RETURN_NOT_OK(DoVisit(*field->type()));
      }
      return SetKind(Kind::kObject);
    }

    Status Visit(const DataType& error) { return Invalid(error); }

    Status SetKind(Kind::type kind) {
      *kind_ = kind;
      return Status::OK();
    }

    Status Invalid(const DataType& error) {
      return Status::Invalid(error, " is not a valid type for a parsed JSON array");
    }

    Status DoVisit(const DataType& unconverted_type) {
      return VisitTypeInline(unconverted_type, this);
    }

    Kind::type* kind_;
  };

  Kind::type kind;
  DCHECK_OK(Impl{&kind}.DoVisit(unconverted_type));
  return kind;
}

/// \brief builder for strings or unconverted numbers
///
/// Both of these are represented in the builder as an index only;
/// the actual characters are stored in a single StringArray (into which
/// an index refers). This means building is faster since we don't do
/// allocation for string/number characters but accessing is strided.
///
/// On completion the indices and the character storage are combined
/// into a dictionary-encoded array, which is a convenient container
/// for indices referring into another array.

template <Kind::type>
struct BuilderType;

template <>
struct BuilderType<Kind::kNull> : NullBuilder {
  BuilderType(MemoryPool*) {}
};

template <>
struct BuilderType<Kind::kBoolean> : BooleanBuilder {
  BuilderType(MemoryPool* pool) : BooleanBuilder(pool) {}
};

template <>
struct BuilderType<Kind::kNumber> : Int32Builder {
  BuilderType(MemoryPool* pool) : Int32Builder(pool) {}
};

template <>
struct BuilderType<Kind::kString> : UInt32Builder {
  BuilderType(MemoryPool* pool) : UInt32Builder(pool) {}
};

// XXX do we want to just make these extra methods standard?
template <>
struct BuilderType<Kind::kArray> : ListBuilder {
 public:
  BuilderType(MemoryPool* pool)
      : ListBuilder(pool, std::make_shared<NullBuilder>(), list(null())) {}

  Status value_builder(const std::shared_ptr<Field>& f,
                       const std::shared_ptr<ArrayBuilder>& b) {
    if (b->length() != value_builder_->length()) {
      return Status::Invalid("can't substitute different length value_builder");
    }
    value_builder_ = b;
    value_field_ = f;
    return Status::OK();
  }

  using ListBuilder::value_builder;
};

template <>
struct BuilderType<Kind::kObject> : StructBuilder {
 public:
  BuilderType(MemoryPool* pool) : StructBuilder(struct_({}), pool, {}) {}

  int GetFieldIndex(const std::string& name) const {
    return checked_cast<const StructType&>(*type_).GetFieldIndex(name);
  }

  Status AddField(const std::shared_ptr<Field>& f, const std::shared_ptr<ArrayBuilder>& b,
                  int* index = nullptr) {
    if (index) {
      *index = num_fields();
    }
    children_.push_back(b);
    auto fields = type_->children();
    fields.push_back(f);
    type_ = struct_(std::move(fields));
    return Status::OK();
  }

  Status field_builder(int index, const std::shared_ptr<Field>& f,
                       const std::shared_ptr<ArrayBuilder>& b) {
    children_[index] = b;
    auto fields = type_->children();
    fields[index] = f;
    type_ = struct_(std::move(fields));
    return Status::OK();
  }

  using StructBuilder::field_builder;
};

inline Kind::type BuilderKind(Type::type type_id) {
  switch (type_id) {
    case Type::NA:
      return Kind::kNull;
    case Type::BOOL:
      return Kind::kBoolean;
    case Type::INT32:
      return Kind::kNumber;
    case Type::UINT32:
      return Kind::kString;
    case Type::LIST:
      return Kind::kArray;
    case Type::STRUCT:
      return Kind::kObject;
    default:
      DCHECK(false);
      return Kind::kNull;
  }
}

inline Kind::type BuilderKind(ArrayBuilder* b) { return BuilderKind(b->type()->id()); }

inline Result<std::shared_ptr<ArrayBuilder>> MakeBuilder(MemoryPool* pool,
                                                         Kind::type kind) {
  switch (kind) {
    case Kind::kNull:
      return std::make_shared<BuilderType<Kind::kNull>>(pool);
    case Kind::kBoolean:
      return std::make_shared<BuilderType<Kind::kBoolean>>(pool);
    case Kind::kNumber:
      return std::make_shared<BuilderType<Kind::kNumber>>(pool);
    case Kind::kString:
      return std::make_shared<BuilderType<Kind::kString>>(pool);
    case Kind::kArray:
      return std::make_shared<BuilderType<Kind::kArray>>(pool);
    case Kind::kObject:
      return std::make_shared<BuilderType<Kind::kObject>>(pool);
    default:
      break;
  }
  DCHECK(false);
  return Status::Invalid("");
}

inline Result<std::shared_ptr<ArrayBuilder>> MakeBuilder(
    MemoryPool* pool, const std::shared_ptr<DataType>& converted_type) {
  ARROW_ASSIGN_OR_RAISE(auto kind, Kind::FromType(*converted_type));
  ARROW_ASSIGN_OR_RAISE(auto out, MakeBuilder(pool, kind));

  switch (kind) {
    case Kind::kArray: {
      auto value_field = checked_cast<const ListType&>(*converted_type).value_field();
      ARROW_ASSIGN_OR_RAISE(auto value_builder, MakeBuilder(pool, value_field->type()));

      auto& list_builder = checked_cast<BuilderType<Kind::kArray>&>(*out);
      RETURN_NOT_OK(list_builder.value_builder(value_field, value_builder));
      return std::move(out);
    }

    case Kind::kObject: {
      auto& struct_builder = checked_cast<BuilderType<Kind::kObject>&>(*out);
      for (const auto& child : converted_type->children()) {
        ARROW_ASSIGN_OR_RAISE(auto field_builder, MakeBuilder(pool, child->type()));
        RETURN_NOT_OK(struct_builder.AddField(child, field_builder));
      }
      return std::move(out);
    }

    default:
      break;
  }

  return std::move(out);
}

/// Three implementations are provided for BlockParser, one for each
/// UnexpectedFieldBehavior. However most of the logic is identical in each
/// case, so the majority of the implementation is in this base class
class HandlerBase : public BlockParser,
                    public rj::BaseReaderHandler<rj::UTF8<>, HandlerBase> {
 public:
  explicit HandlerBase(MemoryPool* pool)
      : BlockParser(pool), scalar_values_builder_(pool) {}

  /// Accessor for a stored error Status
  Status Error() { return status_; }

  /// \defgroup rapidjson-handler-interface functions expected by rj::Reader
  ///
  /// bool Key(const char* data, rj::SizeType size, ...) is omitted since
  /// the behavior varies greatly between UnexpectedFieldBehaviors
  ///
  /// @{
  bool Null() {
    status_ = AppendNull(builder_);
    return status_.ok();
  }

  Status AppendNull(ArrayBuilder* b) {
    if (BuilderKind(b) == Kind::kObject) {
      // also append null to any child builders
      for (int i = 0; i < b->num_children(); ++i) {
        RETURN_NOT_OK(AppendNull(b->child(i)));
      }
    }
    return b->AppendNull();
  }

  bool Bool(bool value) {
    if (ARROW_PREDICT_FALSE(BuilderKind(builder_) != Kind::kBoolean)) {
      status_ = IllegallyChangedTo(Kind::kBoolean);
      return status_.ok();
    }
    status_ = Cast<Kind::kBoolean>(builder_)->Append(value);
    return status_.ok();
  }

  bool RawNumber(const char* data, rj::SizeType size, ...) {
    status_ = AppendScalar<Kind::kNumber>(string_view(data, size));
    return status_.ok();
  }

  bool String(const char* data, rj::SizeType size, ...) {
    status_ = AppendScalar<Kind::kString>(string_view(data, size));
    return status_.ok();
  }

  bool StartObject() {
    status_ = StartObjectImpl();
    return status_.ok();
  }

  bool EndObject(...) {
    status_ = EndObjectImpl();
    return status_.ok();
  }

  bool StartArray() {
    status_ = StartArrayImpl();
    return status_.ok();
  }

  bool EndArray(rj::SizeType size) {
    status_ = EndArrayImpl(size);
    return status_.ok();
  }
  /// @}

  /// \brief Set up builders using an expected Schema
  Status Initialize(const std::shared_ptr<Schema>& s) {
    auto root_type = s != nullptr ? struct_(s->fields()) : struct_({});
    ARROW_ASSIGN_OR_RAISE(root_builder_, MakeBuilder(pool_, root_type));
    builder_ = root_builder_.get();
    return Status::OK();
  }

  Status Finish(std::shared_ptr<Array>* parsed) override {
    std::shared_ptr<Array> scalar_values, parsed_without_scalars;
    RETURN_NOT_OK(scalar_values_builder_.Finish(&scalar_values));
    RETURN_NOT_OK(root_builder_->Finish(&parsed_without_scalars));
    // replace scalars (which are currently just int32)
    // with scalars + values (dictionary(int32, utf8))
    return FinishAddScalars(scalar_values, parsed_without_scalars, parsed);
  }

 protected:
  template <Kind::type kind>
  BuilderType<kind>* Cast(ArrayBuilder* builder) {
    DCHECK_EQ(BuilderKind(builder), kind);
    return checked_cast<BuilderType<kind>*>(builder);
  }

  Status FinishAddScalars(const std::shared_ptr<Array>& scalar_values,
                          const std::shared_ptr<Array>& parsed_without_scalars,
                          std::shared_ptr<Array>* parsed) {
    switch (BuilderKind(parsed_without_scalars->type_id())) {
      default:
        *parsed = parsed_without_scalars;
        return Status::OK();

      case Kind::kNumber: {
        std::shared_ptr<Array> binary_scalar_values;
        RETURN_NOT_OK(scalar_values->View(binary(), &binary_scalar_values));
        return DictionaryArray::FromArrays(dictionary(int32(), binary()),
                                           parsed_without_scalars, binary_scalar_values,
                                           parsed);
      }

      case Kind::kString: {
        std::shared_ptr<Array> signed_indices;
        RETURN_NOT_OK(parsed_without_scalars->View(int32(), &signed_indices));
        return DictionaryArray::FromArrays(dictionary(int32(), utf8()), signed_indices,
                                           scalar_values, parsed);
      }

      case Kind::kArray: {
        const auto& list_array = checked_cast<const ListArray&>(*parsed_without_scalars);

        std::shared_ptr<Array> values;
        RETURN_NOT_OK(FinishAddScalars(scalar_values, list_array.values(), &values));

        auto type = list(list_array.type()->child(0)->WithType(values->type()));
        parsed->reset(new ListArray(type, list_array.length(), list_array.value_offsets(),
                                    values, list_array.null_bitmap(),
                                    list_array.null_count()));
        return Status::OK();
      }

      case Kind::kObject: {
        const auto& struct_array =
            checked_cast<const StructArray&>(*parsed_without_scalars);

        ArrayVector children(struct_array.num_fields());
        std::vector<std::shared_ptr<Field>> fields(struct_array.num_fields());

        for (int i = 0; i < struct_array.num_fields(); ++i) {
          RETURN_NOT_OK(
              FinishAddScalars(scalar_values, struct_array.field(i), &children[i]));

          fields[i] = struct_array.type()->child(i)->WithType(children[i]->type());
        }

        auto type = struct_(std::move(fields));
        parsed->reset(new StructArray(type, struct_array.length(), std::move(children),
                                      struct_array.null_bitmap(),
                                      struct_array.null_count()));
        return Status::OK();
      }
    }
  }

  template <typename Handler, typename Stream>
  Status DoParse(Handler& handler, Stream&& json) {
    constexpr auto parse_flags = rj::kParseIterativeFlag | rj::kParseNanAndInfFlag |
                                 rj::kParseStopWhenDoneFlag |
                                 rj::kParseNumbersAsStringsFlag;

    rj::Reader reader;

    for (; num_rows_ < kMaxParserNumRows; ++num_rows_) {
      auto ok = reader.Parse<parse_flags>(json, handler);
      switch (ok.Code()) {
        case rj::kParseErrorNone:
          // parse the next object
          continue;
        case rj::kParseErrorDocumentEmpty:
          // parsed all objects, finish
          return Status::OK();
        case rj::kParseErrorTermination:
          // handler emitted an error
          return handler.Error();
        default:
          // rj emitted an error
          // FIXME(bkietz) report more error data (at least the byte range of
          // the current block, and maybe the path to the most recently parsed
          // value?)
          return ParseError(rj::GetParseError_En(ok.Code()));
      }
    }
    return Status::Invalid("Exceeded maximum rows");
  }

  template <typename Handler>
  Status DoParse(Handler& handler, const std::shared_ptr<Buffer>& json) {
    RETURN_NOT_OK(ReserveScalarStorage(json->size()));
    rj::MemoryStream ms(reinterpret_cast<const char*>(json->data()), json->size());
    using InputStream = rj::EncodedInputStream<rj::UTF8<>, rj::MemoryStream>;
    return DoParse(handler, InputStream(ms));
  }

  /// \defgroup handlerbase-append-methods append non-nested values
  ///
  /// @{

  template <Kind::type kind>
  Status AppendScalar(string_view scalar) {
    // FIXME(bkietz) the builders don't necessarily have the unconverted type.
    if (ARROW_PREDICT_FALSE(BuilderKind(builder_) != kind)) {
      return IllegallyChangedTo(kind);
    }
    auto index = static_cast<int32_t>(scalar_values_builder_.length());
    RETURN_NOT_OK(Cast<kind>(builder_)->Append(index));
    RETURN_NOT_OK(scalar_values_builder_.Append(scalar));
    return Status::OK();
  }

  /// @}

  Status StartObjectImpl() {
    if (ARROW_PREDICT_FALSE(BuilderKind(builder_) != Kind::kObject)) {
      return IllegallyChangedTo(Kind::kObject);
    }
    auto struct_builder = Cast<Kind::kObject>(builder_);
    absent_fields_stack_.Push(struct_builder->num_fields(), true);
    StartNested();
    return struct_builder->Append();
  }

  /// \brief helper for Key() functions
  ///
  /// sets the field builder with name key, or returns false if
  /// there is no field with that name
  bool SetFieldBuilder(string_view key) {
    auto parent = Cast<Kind::kObject>(builder_stack_.back());
    field_index_ = parent->GetFieldIndex(std::string(key));
    if (ARROW_PREDICT_FALSE(field_index_ == -1)) {
      return false;
    }
    builder_ = parent->field_builder(field_index_);
    absent_fields_stack_[field_index_] = false;
    return true;
  }

  Status EndObjectImpl() {
    auto parent = Cast<Kind::kObject>(builder_stack_.back());

    auto expected_count = absent_fields_stack_.TopSize();
    for (int i = 0; i < expected_count; ++i) {
      if (!absent_fields_stack_[i]) {
        continue;
      }
      auto field_builder = parent->field_builder(i);
      RETURN_NOT_OK(field_builder->AppendNull());
    }
    absent_fields_stack_.Pop();
    EndNested();
    return Status::OK();
  }

  Status StartArrayImpl() {
    if (ARROW_PREDICT_FALSE(BuilderKind(builder_) != Kind::kArray)) {
      return IllegallyChangedTo(Kind::kArray);
    }
    StartNested();
    auto list_builder = Cast<Kind::kArray>(builder_);
    RETURN_NOT_OK(list_builder->Append());
    builder_ = list_builder->value_builder();
    return Status::OK();
  }

  Status EndArrayImpl(rj::SizeType size) {
    EndNested();
    return Status::OK();
  }

  /// helper method for StartArray and StartObject
  /// adds the current builder to a stack so its
  /// children can be visited and parsed.
  void StartNested() {
    field_index_stack_.push_back(field_index_);
    field_index_ = -1;
    builder_stack_.push_back(builder_);
  }

  /// helper method for EndArray and EndObject
  /// replaces the current builder with its parent
  /// so parsing of the parent can continue
  void EndNested() {
    field_index_ = field_index_stack_.back();
    field_index_stack_.pop_back();
    builder_ = builder_stack_.back();
    builder_stack_.pop_back();
  }

  Status IllegallyChangedTo(Kind::type illegally_changed_to) {
    return KindChangeError(BuilderKind(builder_), illegally_changed_to);
  }

  /// Reserve storage for scalars, these can occupy almost all of the JSON
  /// buffer
  Status ReserveScalarStorage(int64_t size) override {
    auto available_storage = scalar_values_builder_.value_data_capacity() -
                             scalar_values_builder_.value_data_length();
    if (size <= available_storage) {
      return Status::OK();
    }
    return scalar_values_builder_.ReserveData(size - available_storage);
  }

  Status status_;
  std::shared_ptr<ArrayBuilder> root_builder_;
  ArrayBuilder* builder_ = nullptr;
  // top of this stack is the parent of builder_
  std::vector<ArrayBuilder*> builder_stack_;
  // top of this stack refers to the fields of the highest *StructBuilder*
  // in builder_stack_ (list builders don't have absent fields)
  BitsetStack absent_fields_stack_;
  // index of builder_ within its parent
  int field_index_ = -1;
  // top of this stack == field_index_
  std::vector<int> field_index_stack_;
  StringBuilder scalar_values_builder_;
};

template <UnexpectedFieldBehavior>
class Handler;

template <>
class Handler<UnexpectedFieldBehavior::Error> : public HandlerBase {
 public:
  using HandlerBase::HandlerBase;

  Status Parse(const std::shared_ptr<Buffer>& json) override {
    return DoParse(*this, json);
  }

  /// \ingroup rapidjson-handler-interface
  ///
  /// if an unexpected field is encountered, emit a parse error and bail
  bool Key(const char* key, rj::SizeType len, ...) {
    if (ARROW_PREDICT_TRUE(SetFieldBuilder(string_view(key, len)))) {
      return true;
    }
    status_ = ParseError("unexpected field");
    return false;
  }
};

template <>
class Handler<UnexpectedFieldBehavior::Ignore> : public HandlerBase {
 public:
  using HandlerBase::HandlerBase;

  Status Parse(const std::shared_ptr<Buffer>& json) override {
    return DoParse(*this, json);
  }

  bool Null() {
    if (Skipping()) {
      return true;
    }
    return HandlerBase::Null();
  }

  bool Bool(bool value) {
    if (Skipping()) {
      return true;
    }
    return HandlerBase::Bool(value);
  }

  bool RawNumber(const char* data, rj::SizeType size, ...) {
    if (Skipping()) {
      return true;
    }
    return HandlerBase::RawNumber(data, size);
  }

  bool String(const char* data, rj::SizeType size, ...) {
    if (Skipping()) {
      return true;
    }
    return HandlerBase::String(data, size);
  }

  bool StartObject() {
    ++depth_;
    if (Skipping()) {
      return true;
    }
    return HandlerBase::StartObject();
  }

  /// \ingroup rapidjson-handler-interface
  ///
  /// if an unexpected field is encountered, skip until its value has been
  /// consumed
  bool Key(const char* key, rj::SizeType len, ...) {
    MaybeStopSkipping();
    if (Skipping()) {
      return true;
    }
    if (ARROW_PREDICT_TRUE(SetFieldBuilder(string_view(key, len)))) {
      return true;
    }
    skip_depth_ = depth_;
    return true;
  }

  bool EndObject(...) {
    MaybeStopSkipping();
    --depth_;
    if (Skipping()) {
      return true;
    }
    return HandlerBase::EndObject();
  }

  bool StartArray() {
    if (Skipping()) {
      return true;
    }
    return HandlerBase::StartArray();
  }

  bool EndArray(rj::SizeType size) {
    if (Skipping()) {
      return true;
    }
    return HandlerBase::EndArray(size);
  }

 private:
  bool Skipping() { return depth_ >= skip_depth_; }

  void MaybeStopSkipping() {
    if (skip_depth_ == depth_) {
      skip_depth_ = std::numeric_limits<int>::max();
    }
  }

  int depth_ = 0;
  int skip_depth_ = std::numeric_limits<int>::max();
};

template <>
class Handler<UnexpectedFieldBehavior::InferType> : public HandlerBase {
 public:
  using HandlerBase::HandlerBase;

  Status Parse(const std::shared_ptr<Buffer>& json) override {
    return DoParse(*this, json);
  }

  bool Bool(bool value) {
    if (ARROW_PREDICT_OK(status_ = MaybePromoteFromNull<Kind::kBoolean>())) {
      return HandlerBase::Bool(value);
    }
    return false;
  }

  bool RawNumber(const char* data, rj::SizeType size, ...) {
    if (ARROW_PREDICT_OK(status_ = MaybePromoteFromNull<Kind::kNumber>())) {
      return HandlerBase::RawNumber(data, size);
    }
    return false;
  }

  bool String(const char* data, rj::SizeType size, ...) {
    if (ARROW_PREDICT_OK(status_ = MaybePromoteFromNull<Kind::kString>())) {
      return HandlerBase::String(data, size);
    }
    return false;
  }

  bool StartObject() {
    if (ARROW_PREDICT_OK(status_ = MaybePromoteFromNull<Kind::kObject>())) {
      return HandlerBase::StartObject();
    }
    return false;
  }

  /// \ingroup rapidjson-handler-interface
  ///
  /// If an unexpected field is encountered, add a new builder to
  /// the current parent builder. It is added as a NullBuilder with
  /// (parent.length - 1) leading nulls. The next value parsed
  /// will probably trigger promotion of this field from null
  bool Key(const char* key, rj::SizeType len, ...) {
    if (ARROW_PREDICT_TRUE(SetFieldBuilder(string_view(key, len)))) {
      return true;
    }
    auto struct_builder = Cast<Kind::kObject>(builder_stack_.back());
    auto new_field = field(std::string(key, len), null());
    auto new_builder = std::make_shared<BuilderType<Kind::kNull>>(pool_);
    DCHECK_OK(new_builder->AppendNulls(struct_builder->length() - 1));
    status_ = struct_builder->AddField(new_field, new_builder, &field_index_);
    builder_ = new_builder.get();
    return status_.ok();
  }

  bool StartArray() {
    if (ARROW_PREDICT_OK(status_ = MaybePromoteFromNull<Kind::kArray>())) {
      return HandlerBase::StartArray();
    }
    return false;
  }

 private:
  // return true if a terminal error was encountered
  template <Kind::type kind>
  Status MaybePromoteFromNull() {
    if (ARROW_PREDICT_TRUE(BuilderKind(builder_) == kind)) {
      return Status::OK();
    }
    if (ARROW_PREDICT_FALSE(BuilderKind(builder_) != Kind::kNull)) {
      return Status::Invalid("may only promote from null");
    }

    ARROW_ASSIGN_OR_RAISE(auto promoted, MakeBuilder(pool_, kind));
    RETURN_NOT_OK(promoted->AppendNulls(builder_->length()));

    auto parent = builder_stack_.back();
    if (BuilderKind(parent) == Kind::kArray) {
      auto list_builder = Cast<Kind::kArray>(parent);
      DCHECK_EQ(list_builder->value_builder(), builder_);
      auto new_field = list_builder->type()->child(0)->WithType(promoted->type());
      RETURN_NOT_OK(list_builder->value_builder(new_field, promoted));
    } else {
      auto struct_builder = Cast<Kind::kObject>(parent);
      DCHECK_EQ(struct_builder->field_builder(field_index_), builder_);
      auto new_field =
          struct_builder->type()->child(field_index_)->WithType(promoted->type());
      RETURN_NOT_OK(struct_builder->field_builder(field_index_, new_field, promoted));
    }

    builder_ = promoted.get();
    return Status::OK();
  }
};

Status BlockParser::Make(MemoryPool* pool, const ParseOptions& options,
                         std::unique_ptr<BlockParser>* out) {
  DCHECK(options.unexpected_field_behavior == UnexpectedFieldBehavior::InferType ||
         options.explicit_schema != nullptr);

  switch (options.unexpected_field_behavior) {
    case UnexpectedFieldBehavior::Ignore: {
      *out = make_unique<Handler<UnexpectedFieldBehavior::Ignore>>(pool);
      break;
    }
    case UnexpectedFieldBehavior::Error: {
      *out = make_unique<Handler<UnexpectedFieldBehavior::Error>>(pool);
      break;
    }
    case UnexpectedFieldBehavior::InferType:
      *out = make_unique<Handler<UnexpectedFieldBehavior::InferType>>(pool);
      break;
  }
  return static_cast<HandlerBase&>(**out).Initialize(options.explicit_schema);
}

Status BlockParser::Make(const ParseOptions& options, std::unique_ptr<BlockParser>* out) {
  return BlockParser::Make(default_memory_pool(), options, out);
}

}  // namespace json
}  // namespace arrow
