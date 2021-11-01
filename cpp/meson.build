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

# Meson style guide:
# - name variables in snake_case
# - prefer dependency() or a wrap file for dependencies
#   https://mesonbuild.com/Wrap-dependency-system-manual.html
# - name options minimally- don't call something 'enable-ipc'
#   when 'ipc' will do.
# - prefer add_project_arguments to manual flags
# - prefer sorted lists
# - don't alias get_option
# - subdir() calls should never be conditional. Instead if a
#   subdir should be skipped then put that check at the top of
#   its meson.build and invoke subdir_done(). This makes
#   preconditions obvious in the file to which they apply instead
#   of implicitly handled at a higher level
# - don't automatically resolve inter-option dependencies,
#   just error helpfully

option_interdependencies = {
  # [$option: name of an option]: {
  #   [name of an option required to enable $option]: extra error message
  # },

  'tests': {
    'json': 'unit tests use JSON to generate test data',
    # XXX is this required for anything other than json_integration_test?
    'ipc': '',
  },

  'large_memory_tests': { 'tests': '' },
  'memcheck_tests': { 'tests': '' },
  'timing_tests': { 'tests': '' },

  'integration': {
    'ipc': '',
  },

  'benchmarks': {
    'json': 'benchmarks use JSON to generate test data',
  },
}

foreach opt, deps : option_interdependencies
  foreach dep, err : deps
    get_option(opt).require(
      get_option(dep).enabled(),
      error_message: err + ' (use -D ' + dep + '=enabled)'
    )
  endforeach
endforeach

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

base_include_directories = [
  include_directories('src'),
  include_directories('src/generated'),
  include_directories('thirdparty/hadoop/include'),
  include_directories('thirdparty/flatbuffers/include'),
]

base_dependencies = [
  dependency('threads'),
  dependency('RapidJSON', required: get_option('json')),
  meson.get_compiler('c').find_library('dl', required: false),
]

msvc = meson.get_compiler('cpp').get_id() == 'msvc'
def = msvc ? '/D' : '-D'

ARROW_GIT_ID = get_option('git_id')
if ARROW_GIT_ID == ''
  ARROW_GIT_ID = run_command('git', 'log', 'n1', '--format=%H').stdout().strip()
endif
add_project_arguments(def + 'ARROW_GIT_ID=' + ARROW_GIT_ID, language: 'cpp')

ARROW_GIT_DESCRIPTION = get_option('git_description')
if ARROW_GIT_DESCRIPTION == ''
  ARROW_GIT_DESCRIPTION = run_command('git', 'describe', '--tags', '--dirty').stdout().strip()
endif
add_project_arguments(def + 'ARROW_GIT_DESCRIPTION=' + ARROW_GIT_DESCRIPTION, language: 'cpp')

add_project_arguments(def + 'ARROW_PACKAGE_KIND=' + get_option('package_kind'), language: 'cpp')

if get_option('deprecated_api').disabled()
  add_project_arguments(def + 'ARROW_NO_DEPRECATED_API', language: 'cpp')
endif

if target_machine.cpu_family() in ['x86', 'x86_64']
  assert(get_option('simd_level') in ['DEFAULT', 'NONE', 'SSE4_2', 'AVX2', 'AVX512'])

  if msvc and get_option('simd_level') in ['AVX2', 'AVX512']
    add_project_arguments('/arch:' + get_option('simd_level'), language: 'cpp')
  else
    if get_option('simd_level') in ['DEFAULT', 'SSE4_2']
      add_project_arguments('-msse4.2', language: 'cpp')
    elif get_option('simd_level') == 'AVX2'
      # Append the avx2/avx512 subset option also, fix issue ARROW-9877 for homebrew-cpp
      add_project_arguments(
        '-march=haswell',
        '-mavx2',
        language: 'cpp'
      )
    elif get_option('simd_level') == 'AVX512'
      add_project_arguments(
        '-march=skylake-avx512',
        '-mbmi2',
        '-mavx512f',
        '-mavx512cd',
        '-mavx512vl',
        '-mavx512dq',
        '-mavx512bw',
        language: 'cpp',
      )
    endif
  endif
elif target_machine.cpu_family() in ['arm', 'aarch64']
  assert(get_option('simd_level') in ['DEFAULT', 'NONE', 'NEON'])

  if get_option('simd_level') in ['DEFAULT', 'NEON']
    add_project_arguments(def + 'ARROW_HAVE_NEON', language: 'cpp')
  endif
elif target_machine.cpu_family() in ['ppc', 'ppc64']
  assert(get_option('simd_level') in ['DEFAULT', 'NONE', 'ALTIVEC'])

  if get_option('simd_level') == 'ALTIVEC'
    add_project_arguments('-maltivec', language: 'cpp')
  endif
endif

if get_option('large_memory_tests').enabled()
  add_project_arguments(def + 'ARROW_LARGE_MEMORY_TESTS', language: 'cpp')
endif

if get_option('memcheck_tests').enabled()
  add_project_arguments(def + 'ARROW_LARGE_MEMORY_TESTS', language: 'cpp')
endif

if get_option('timing_tests').enabled()
  add_project_arguments(def + 'ARROW_WITH_TIMING_TESTS', language: 'cpp')
endif

if get_option('b_sanitize').contains('address')
  add_project_arguments(def + 'ADDRESS_SANITIZER', language: 'cpp')
endif

if get_option('b_sanitize').contains('undefined')
  add_project_arguments(def + 'ARROW_UBSAN', language: 'cpp')
endif

if get_option('b_sanitize').contains('thread')
  add_project_arguments(def + 'THREAD_SANITIZER', language: 'cpp')
endif

subdir('build-support') # lint jobs etc

subdir('src/arrow')

subdir('examples/arrow')

