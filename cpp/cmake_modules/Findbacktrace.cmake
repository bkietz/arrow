# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tries to find backtrace header and library.
#
# Usage of this module as follows:
#
#  find_package(backtrace)
#
# This module defines
#  backtrace::backtrace, target to use backtrace

set(backtrace_LIB_NAMES
    "${CMAKE_STATIC_LIBRARY_PREFIX}backtrace${CMAKE_STATIC_LIBRARY_SUFFIX}")

find_library(backtrace_LIB
             NAMES ${backtrace_LIB_NAMES}
             PATH_SUFFIXES ${ARROW_LIBRARY_PATH_SUFFIXES})

find_path(backtrace_INCLUDE_DIR
          NAMES backtrace.h
          PATH_SUFFIXES ${ARROW_INCLUDE_PATH_SUFFIXES})

find_package_handle_standard_args(backtrace REQUIRED_VARS backtrace_LIB
                                                          backtrace_INCLUDE_DIR)

if(backtrace_FOUND)
  if(NOT TARGET backtrace::backtrace)
    add_library(backtrace::backtrace STATIC IMPORTED)
    set_target_properties(backtrace::backtrace
                          PROPERTIES IMPORTED_LOCATION "${backtrace_LIB}"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                     "${backtrace_INCLUDE_DIR}")
  endif()
endif()
