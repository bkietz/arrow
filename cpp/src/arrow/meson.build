# -*- indent-tabs-mode: nil -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

sources = files(
  'array/array_base.cc',
  'array/array_binary.cc',
  'array/array_decimal.cc',
  'array/array_dict.cc',
  'array/array_nested.cc',
  'array/array_primitive.cc',
  'array/builder_adaptive.cc',
  'array/builder_base.cc',
  'array/builder_binary.cc',
  'array/builder_decimal.cc',
  'array/builder_dict.cc',
  'array/builder_nested.cc',
  'array/builder_primitive.cc',
  'array/builder_union.cc',
  'array/concatenate.cc',
  'array/data.cc',
  'array/diff.cc',
  'array/util.cc',
  'array/validate.cc',
  'builder.cc',
  'buffer.cc',
  'chunked_array.cc',
  'compare.cc',
  'config.cc',
  'datum.cc',
  'device.cc',
  'extension_type.cc',
  'memory_pool.cc',
  'pretty_print.cc',
  'record_batch.cc',
  'result.cc',
  'scalar.cc',
  'sparse_tensor.cc',
  'status.cc',
  'table.cc',
  'table_builder.cc',
  'tensor.cc',
  'tensor/coo_converter.cc',
  'tensor/csf_converter.cc',
  'tensor/csx_converter.cc',
  'type.cc',
  'visitor.cc',
  'c/bridge.cc',
  'io/buffered.cc',
  'io/caching.cc',
  'io/compressed.cc',
  'io/file.cc',
  'io/hdfs.cc',
  'io/hdfs_internal.cc',
  'io/interfaces.cc',
  'io/memory.cc',
  'io/slow.cc',
  'io/stdio.cc',
  'io/transform.cc',
  'util/async_util.cc',
  'util/basic_decimal.cc',
  'util/bit_block_counter.cc',
  'util/bit_run_reader.cc',
  'util/bit_util.cc',
  'util/bitmap.cc',
  'util/bitmap_builders.cc',
  'util/bitmap_ops.cc',
  'util/bpacking.cc',
  'util/cancel.cc',
  'util/compression.cc',
  'util/counting_semaphore.cc',
  'util/cpu_info.cc',
  'util/decimal.cc',
  'util/delimiting.cc',
  'util/formatting.cc',
  'util/future.cc',
  'util/int_util.cc',
  'util/io_util.cc',
  'util/logging.cc',
  'util/key_value_metadata.cc',
  'util/memory.cc',
  'util/mutex.cc',
  'util/string.cc',
  'util/string_builder.cc',
  'util/task_group.cc',
  'util/tdigest.cc',
  'util/thread_pool.cc',
  'util/time.cc',
  'util/trie.cc',
  'util/unreachable.cc',
  'util/uri.cc',
  'util/utf8.cc',
  'util/value_parsing.cc',
  'vendored/base64.cpp',
  'vendored/datetime/tz.cpp',
  'vendored/double-conversion/bignum.cc',
  'vendored/double-conversion/double-conversion.cc',
  'vendored/double-conversion/bignum-dtoa.cc',
  'vendored/double-conversion/fast-dtoa.cc',
  'vendored/double-conversion/cached-powers.cc',
  'vendored/double-conversion/fixed-dtoa.cc',
  'vendored/double-conversion/diy-fp.cc',
  'vendored/double-conversion/strtod.cc',
)

c_sources = files(
  'vendored/musl/strptime.c',
  'vendored/uriparser/UriCommon.c',
  'vendored/uriparser/UriCompare.c',
  'vendored/uriparser/UriEscape.c',
  'vendored/uriparser/UriFile.c',
  'vendored/uriparser/UriIp4Base.c',
  'vendored/uriparser/UriIp4.c',
  'vendored/uriparser/UriMemory.c',
  'vendored/uriparser/UriNormalizeBase.c',
  'vendored/uriparser/UriNormalize.c',
  'vendored/uriparser/UriParseBase.c',
  'vendored/uriparser/UriParse.c',
  'vendored/uriparser/UriQuery.c',
  'vendored/uriparser/UriRecompose.c',
  'vendored/uriparser/UriResolve.c',
  'vendored/uriparser/UriShorten.c',
)

arrow = library('arrow',
  sources: sources + c_sources,
  install: true,
  include_directories: base_include_directories,
  soversion: so_version,
  version: library_version,
  dependencies: [threads] + dl,
)

if get_option('build_tests')
  gtest = dependency('gtest')

  arrow_testing = library('arrow_testing',
    install: true,
    include_directories: base_include_directories,
    soversion: so_version,
    version: library_version,
    dependencies: [dependency('gtest'), threads] + dl,
    link_with: [arrow],
  )

  array_test = executable('array_test',
    'array/array_test.cc',
    'array/array_binary_test.cc',
    'array/array_dict_test.cc',
    'array/array_list_test.cc',
    'array/array_struct_test.cc',
    'array/array_union_test.cc',
    'array/array_view_test.cc',
    dependencies: [dependency('gtest'), threads] + dl,
    link_with: [arrow, arrow_testing],
  )
endif

