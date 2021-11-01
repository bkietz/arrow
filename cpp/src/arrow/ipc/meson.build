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

if not get_option('ipc').enabled()
  subdir_done()
endif

arrow_tests += {
  'feather': files('feather_test.cc'),
  'ipc_json_simple': files('json_simple_test.cc'),
  'ipc_read_write': files('read_write_test.cc'),
  'ipc_tensor': files('tensor_test.cc'),
}

# FIXME add command line utilities
# FIXME add fuzz corpus generators
