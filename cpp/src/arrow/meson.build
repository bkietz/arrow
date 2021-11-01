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

arrow_sources = files(
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
  'buffer.cc',
  'builder.cc',
  'c/bridge.cc',
  'chunked_array.cc',
  'compare.cc',
  'config.cc',
  'datum.cc',
  'device.cc',
  'extension_type.cc',
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
  'util/key_value_metadata.cc',
  'util/logging.cc',
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
  'vendored/double-conversion/bignum-dtoa.cc',
  'vendored/double-conversion/bignum.cc',
  'vendored/double-conversion/cached-powers.cc',
  'vendored/double-conversion/diy-fp.cc',
  'vendored/double-conversion/double-conversion.cc',
  'vendored/double-conversion/fast-dtoa.cc',
  'vendored/double-conversion/fixed-dtoa.cc',
  'vendored/double-conversion/strtod.cc',
  'vendored/musl/strptime.c',
  'vendored/uriparser/UriCommon.c',
  'vendored/uriparser/UriCompare.c',
  'vendored/uriparser/UriEscape.c',
  'vendored/uriparser/UriFile.c',
  'vendored/uriparser/UriIp4.c',
  'vendored/uriparser/UriIp4Base.c',
  'vendored/uriparser/UriMemory.c',
  'vendored/uriparser/UriNormalize.c',
  'vendored/uriparser/UriNormalizeBase.c',
  'vendored/uriparser/UriParse.c',
  'vendored/uriparser/UriParseBase.c',
  'vendored/uriparser/UriQuery.c',
  'vendored/uriparser/UriRecompose.c',
  'vendored/uriparser/UriResolve.c',
  'vendored/uriparser/UriShorten.c',
  'visitor.cc',
)

if get_option('ipc').enabled()
  arrow_sources += files(
    'ipc/dictionary.cc',
    'ipc/feather.cc',
    'ipc/message.cc',
    'ipc/metadata_internal.cc',
    'ipc/options.cc',
    'ipc/reader.cc',
    'ipc/writer.cc',
  )

  if get_option('json').enabled()
    arrow_sources += files(
      'ipc/json_simple.cc',
    )
  endif
endif

if get_option('csv').enabled()
  arrow_sources += files(
    'csv/converter.cc',
    'csv/chunker.cc',
    'csv/column_builder.cc',
    'csv/column_decoder.cc',
    'csv/options.cc',
    'csv/parser.cc',
    'csv/reader.cc',
  )
  if get_option('compute').enabled()
    arrow_sources += files(
      'csv/writer.cc',
    )
  endif
endif

if get_option('json').enabled()
  arrow_sources += files(
    'json/options.cc',
    'json/chunked_builder.cc',
    'json/chunker.cc',
    'json/converter.cc',
    'json/object_parser.cc',
    'json/object_writer.cc',
    'json/parser.cc',
    'json/reader.cc',
  )
endif

if get_option('compute').enabled()
  arrow_sources += files(
    'compute/api_aggregate.cc',
    'compute/api_scalar.cc',
    'compute/api_vector.cc',
    'compute/cast.cc',
    'compute/exec.cc',
    'compute/exec/aggregate_node.cc',
    'compute/exec/exec_plan.cc',
    'compute/exec/expression.cc',
    'compute/exec/filter_node.cc',
    'compute/exec/project_node.cc',
    'compute/exec/source_node.cc',
    'compute/exec/sink_node.cc',
    'compute/exec/order_by_impl.cc',
    'compute/function.cc',
    'compute/function_internal.cc',
    'compute/kernel.cc',
    'compute/registry.cc',
    'compute/kernels/aggregate_basic.cc',
    'compute/kernels/aggregate_mode.cc',
    'compute/kernels/aggregate_quantile.cc',
    'compute/kernels/aggregate_tdigest.cc',
    'compute/kernels/aggregate_var_std.cc',
    'compute/kernels/codegen_internal.cc',
    'compute/kernels/hash_aggregate.cc',
    'compute/kernels/scalar_arithmetic.cc',
    'compute/kernels/scalar_boolean.cc',
    'compute/kernels/scalar_cast_boolean.cc',
    'compute/kernels/scalar_cast_dictionary.cc',
    'compute/kernels/scalar_cast_internal.cc',
    'compute/kernels/scalar_cast_nested.cc',
    'compute/kernels/scalar_cast_numeric.cc',
    'compute/kernels/scalar_cast_string.cc',
    'compute/kernels/scalar_cast_temporal.cc',
    'compute/kernels/scalar_compare.cc',
    'compute/kernels/scalar_nested.cc',
    'compute/kernels/scalar_set_lookup.cc',
    'compute/kernels/scalar_string.cc',
    'compute/kernels/scalar_temporal_binary.cc',
    'compute/kernels/scalar_temporal_unary.cc',
    'compute/kernels/scalar_validity.cc',
    'compute/kernels/scalar_if_else.cc',
    'compute/kernels/util_internal.cc',
    'compute/kernels/vector_array_sort.cc',
    'compute/kernels/vector_hash.cc',
    'compute/kernels/vector_nested.cc',
    'compute/kernels/vector_replace.cc',
    'compute/kernels/vector_selection.cc',
    'compute/kernels/vector_sort.cc',
    'compute/kernels/row_encoder.cc',
    'compute/exec/union_node.cc',
    'compute/exec/key_hash.cc',
    'compute/exec/key_map.cc',
    'compute/exec/key_compare.cc',
    'compute/exec/key_encode.cc',
    'compute/exec/util.cc',
    'compute/exec/hash_join.cc',
    'compute/exec/hash_join_node.cc',
    'compute/exec/task_util.cc',
  )
endif

arrow = library('arrow',
  sources: arrow_sources,
  install: true,
  include_directories: base_include_directories,
  soversion: so_version,
  version: library_version,
  dependencies: base_dependencies,
)

if (
   get_option('tests').enabled() or
   get_option('benchmarks').enabled() or
   get_option('integration').enabled() # or get_option('fuzzing').enabled()
)
  test_dependencies = base_dependencies + [
    dependency('gmock'),
    dependency('gtest'),
  ]

  gtest_main = [dependency('gtest_main')]

  arrow_testing_sources = files(
    'io/test_common.cc',
    'ipc/test_common.cc',
    'testing/json_integration.cc',
    'testing/json_internal.cc',
    'testing/gtest_util.cc',
    'testing/random.cc',
    'testing/generator.cc',
    'testing/util.cc',
  )

  if get_option('csv').enabled()
    arrow_testing_sources += files(
      'csv/test_common.cc',
    )
  endif

  if get_option('compute').enabled()
    arrow_testing_sources += files(
      'compute/exec/test_util.cc',
    )
  endif

  arrow_testing = library('arrow_testing',
    sources: arrow_testing_sources,
    install: true,
    include_directories: base_include_directories,
    soversion: so_version,
    version: library_version,
    dependencies: test_dependencies,
    link_with: [arrow],
  )
endif

# basic unit tests can just be added to arrow_tests
arrow_tests = {
  # [$name: name of unit test]: files(
  #   files to compile into this unit test
  # )
  'buffer': files('buffer_test.cc'),

  'misc': files(
    'datum_test.cc',
    'memory_pool_test.cc',
    'result_test.cc',
    'pretty_print_test.cc',
    'status_test.cc',
  ),

  'scalar': files('scalar_test.cc'),

  'stl': files(
    'stl_iterator_test.cc',
  ) + (get_option('compute').enabled() ? files(
    # This unit test uses compute code
    'stl_test.cc',
  ) : []),

  'table': files(
    'chunked_array_test.cc',
    'record_batch_test.cc',
    'table_test.cc',
    'table_builder_test.cc',
  ),

  'tensor': files(
    'sparse_tensor_test.cc',
    'tensor_test.cc',
  ),

  'type': files(
    'type_test.cc',
  ) + (get_option('ipc').enabled() ? files(
    # The extension type unit tests require IPC / Flatbuffers support
    'extension_type_test.cc',
  ) : []),
}

# basic benchmarks can just be added to arrow_benchmarks
arrow_benchmarks = {
  # [$name: name of benchmark]: files(
  #   files to compile into this benchmark
  # )
  'builder': files('builder_benchmark.cc'),
  'compare': files('compare_benchmark.cc'),
  'memory_pool': files('memory_pool_benchmark.cc'),
  'type': files('type_benchmark.cc'),
}

subdir('array')
subdir('c')
subdir('compute')
subdir('io')
subdir('ipc')
subdir('json')
subdir('testing')

if get_option('tests').enabled()
  foreach name, sources : arrow_tests
    test(name + '_test',
      executable(name + '_test',
        sources,
        dependencies: test_dependencies + gtest_main,
        link_with: [arrow, arrow_testing],
        include_directories: base_include_directories,
        cpp_pch: 'testing/pch.h',
      ),
      suite: 'core',
    )
  endforeach
endif

# json_integration_test is two things at the same time:
# - an executable that can be called to answer integration test requests
# - a self-(unit)test for the C++ side of integration testing
if get_option('tests').enabled()
  test('json_integration_test',
    executable('json_integration_test',
      'testing/json_integration_test.cc',
      dependencies: test_dependencies + [dependency('gflags')] + gtest_main,
      link_with: [arrow, arrow_testing],
      include_directories: base_include_directories,
    ),
    suite: 'core',
  )
endif

if get_option('benchmarks').enabled()
  bench = [
    dependency('benchmark'),
    meson.get_compiler('cpp').find_library('benchmark_main'),
  ]
  foreach name, sources : arrow_benchmarks
    benchmark(name + '_benchmark',
      executable(name + '_benchmark',
        sources,
        dependencies: test_dependencies + bench,
        link_with: [arrow, arrow_testing],
        include_directories: base_include_directories,
        cpp_pch: 'testing/pch.h',
      ),
      suite: 'core',
    )
  endforeach
endif
