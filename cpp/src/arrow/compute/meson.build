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

if not get_option('compute').enabled()
  subdir_done()
endif

arrow_tests += {
  'compute_internals': files(
    'function_test.cc',
    'exec_test.cc',
    'kernel_test.cc',
    'registry_test.cc',
    'kernels/codegen_internal_test.cc',
    'exec/util_test.cc',
  ),

  'aggregate_kernels': files(
    'kernels/aggregate_test.cc',
    'kernels/hash_aggregate_test.cc',
    'kernels/test_util.cc',
  ),

  'expression': files(
    'exec/expression_test.cc',
    'exec/subtree_test.cc',
  ),

  'hash_join_node': files('exec/hash_join_node_test.cc'),

  'plan': files('exec/plan_test.cc'),

  'scalar_kernels': files(
    'kernels/scalar_arithmetic_test.cc',
    'kernels/scalar_boolean_test.cc',
    'kernels/scalar_cast_test.cc',
    'kernels/scalar_compare_test.cc',
    'kernels/scalar_nested_test.cc',
    'kernels/scalar_set_lookup_test.cc',
    'kernels/scalar_string_test.cc',
    'kernels/scalar_temporal_test.cc',
    'kernels/scalar_validity_test.cc',
    'kernels/scalar_if_else_test.cc',
    'kernels/test_util.cc',
  ),

  'union_node': files('exec/union_node_test.cc'),

  'vector_kernels': files(
    'kernels/vector_hash_test.cc',
    'kernels/vector_nested_test.cc',
    'kernels/vector_replace_test.cc',
    'kernels/vector_selection_test.cc',
    'kernels/vector_sort_test.cc',
    'kernels/select_k_test.cc',
    'kernels/test_util.cc',
  ),
}

arrow_benchmarks += {
  'expression': files('exec/expression_benchmark.cc'),
  'function': files('function_benchmark.cc'),
}
