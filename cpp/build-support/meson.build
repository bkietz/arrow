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

build_support_dir = meson.source_root() / 'build-support'
python = find_program('python3')
lint_quiet = get_option('verbose_lint').disabled() ? ['--quiet'] : []

custom_target(
  output: 'lint',
  command: [
    python.path(),
    join_paths(build_support_dir, 'run_cpplint.py'),
    '--cpplint_binary', join_paths(build_support_dir, 'cpplint.py'),
    '--exclude_globs', build_support_dir / 'lint_exclusions.txt',
    '--source_dir', meson.source_root() / 'src',
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
    '--exclude_globs', build_support_dir / 'lint_exclusions.txt',
    '--source_dir', meson.source_root() / 'src',
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
