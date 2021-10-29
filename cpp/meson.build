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

project('arrow', 'cpp', 'c',
        license: 'Apache-2.0',
        default_options: [
          'cpp_std=c++11',
        ])

version = '7.0.0-SNAPSHOT'
if version.endswith('-SNAPSHOT')
  version_numbers = version.split('-')[0].split('.')
  version_tag = version.split('-')[1]
else
  version_numbers = version.split('.')
  version_tag = ''
endif
version_major = version_numbers[0].to_int()
version_minor = version_numbers[1].to_int()
version_micro = version_numbers[2].to_int()

so_version = version_major * 100 + version_minor
so_version_patch = version_micro
library_version = '@0@.@1@.@2@'.format(so_version, so_version_patch, 0)

prefix = get_option('prefix')

build_tests = get_option('build_tests')

base_include_directories = [
  include_directories('src'),
  include_directories('src/generated'),
  include_directories('thirdparty/hadoop/include'),
  include_directories('thirdparty/flatbuffers/include'),
]

build_support_dir = join_paths(meson.source_root(), 'build-support')
src_dir = join_paths(meson.source_root(), 'src')

python = find_program('python3')
threads = dependency('threads')
dl = meson.get_compiler('c').find_library('dl', required: false)
dl = dl.found() ? [dl] : []

lint_exclusions = join_paths(build_support_dir, 'lint_exclusions.txt')
lint_quiet = get_option('quiet_lint') ? ['--quiet'] : []

custom_target(
  output: 'lint',
  command: [
    python.path(),
    join_paths(build_support_dir, 'run_cpplint.py'),
    '--cpplint_binary', join_paths(build_support_dir, 'cpplint.py'),
    '--exclude_globs', lint_exclusions,
    '--source_dir', src_dir,
  ] + lint_quiet,
)

custom_target(
  output: 'lint-cpp-cli',
  command: [
    python.path(),
    join_paths(build_support_dir, 'lint_cpp_cli.py'),
    join_paths(build_support_dir, 'cpplint.py'),
  ],
)

clang_format = find_program('clang-format-8')
if clang_format.found()
  clang_format_cmd = [
    python.path(),
    join_paths(build_support_dir, 'run_clang_format.py'),
    '--clang_format_binary', clang_format.path(),
    '--exclude_globs', lint_exclusions,
    '--source_dir', src_dir,
  ] + lint_quiet

  custom_target(
    output: 'format',
    command: clang_format_cmd + ['--fix'],
  )

  custom_target(
    output: 'check-format',
    command: clang_format_cmd,
  )
endif

subdir('src/arrow')
