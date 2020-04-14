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
#
# This file is generated by archery, to update it use for example
#
# $ archery bundled-thirdparty versions --update=BOOST@1.71.0

if(DEFINED ENV{ARROW_AWSSDK_URL})
  set(AWSSDK_SOURCE_URL "$ENV{ARROW_AWSSDK_URL}")
else()
  set_urls(
    AWSSDK_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/aws-sdk-cpp-1.7.160.tar.gz"
    "https://github.com/aws/aws-sdk-cpp/archive/1.7.160.tar.gz"
    "https://dl.bintray.com/ursalabs/arrow-awssdk/aws-sdk-cpp-1.7.160.tar.gz/aws-sdk-cpp-1.7.160.tar.gz")
endif()

set(ARROW_AWSSDK_BUILD_VERSION "1.7.160")

set(AWSSDK_BUILD_MD5_CHECKSUM "MD5=b9d39a6d14f26ea4890b0437509b4c9f")


if(DEFINED ENV{ARROW_BOOST_URL})
  set(BOOST_SOURCE_URL "$ENV{ARROW_BOOST_URL}")
else()
  set_urls(
    BOOST_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/boost_1_71_0.tar.gz"
    "https://dl.bintray.com/ursalabs/arrow-boost/boost_1_71_0.tar.gz"
    "https://github.com/boostorg/boost/archive/boost-1.71.0.tar.gz"
    "https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz")
endif()

set(ARROW_BOOST_BUILD_VERSION "1.71.0")

set(BOOST_BUILD_MD5_CHECKSUM "MD5=056bbc673d60814c0fb0eeeeaf12392b")


if(DEFINED ENV{ARROW_BROTLI_URL})
  set(BROTLI_SOURCE_URL "$ENV{ARROW_BROTLI_URL}")
else()
  set_urls(
    BROTLI_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/brotli-v1.0.7.tar.gz"
    "https://github.com/google/brotli/archive/v1.0.7.tar.gz")
endif()

set(ARROW_BROTLI_BUILD_VERSION "1.0.7")

set(BROTLI_BUILD_MD5_CHECKSUM "MD5=7b6edd4f2128f22794d0ca28c53898a5")


if(DEFINED ENV{ARROW_BZIP2_URL})
  set(BZIP2_SOURCE_URL "$ENV{ARROW_BZIP2_URL}")
else()
  set_urls(
    BZIP2_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/bzip2-1.0.8.tar.gz"
    "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz")
endif()

set(ARROW_BZIP2_BUILD_VERSION "1.0.8")

set(BZIP2_BUILD_MD5_CHECKSUM "MD5=67e051268d0c475ea773822f7500d0e5")


if(DEFINED ENV{ARROW_CARES_URL})
  set(CARES_SOURCE_URL "$ENV{ARROW_CARES_URL}")
else()
  set_urls(
    CARES_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/cares-1.15.0.tar.gz"
    "https://c-ares.haxx.se/download/c-ares-1.15.0.tar.gz")
endif()

set(ARROW_CARES_BUILD_VERSION "1.15.0")

set(CARES_BUILD_MD5_CHECKSUM "MD5=d2391da274653f7643270623e822dff7")


if(DEFINED ENV{ARROW_GBENCHMARK_URL})
  set(GBENCHMARK_SOURCE_URL "$ENV{ARROW_GBENCHMARK_URL}")
else()
  set_urls(
    GBENCHMARK_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/gbenchmark-v1.5.0.tar.gz"
    "https://github.com/google/benchmark/archive/v1.5.0.tar.gz")
endif()

set(ARROW_GBENCHMARK_BUILD_VERSION "1.5.0")

set(GBENCHMARK_BUILD_MD5_CHECKSUM "MD5=eb1466370f3ae31e74557baa29729e9e")


if(DEFINED ENV{ARROW_GFLAGS_URL})
  set(GFLAGS_SOURCE_URL "$ENV{ARROW_GFLAGS_URL}")
else()
  set_urls(
    GFLAGS_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/gflags-v2.2.0.tar.gz"
    "https://github.com/gflags/gflags/archive/v2.2.0.tar.gz")
endif()

set(ARROW_GFLAGS_BUILD_VERSION "2.2.0")

set(GFLAGS_BUILD_MD5_CHECKSUM "MD5=b99048d9ab82d8c56e876fb1456c285e")


if(DEFINED ENV{ARROW_GLOG_URL})
  set(GLOG_SOURCE_URL "$ENV{ARROW_GLOG_URL}")
else()
  set_urls(
    GLOG_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/glog-v0.3.5.tar.gz"
    "https://github.com/google/glog/archive/v0.3.5.tar.gz")
endif()

set(ARROW_GLOG_BUILD_VERSION "0.3.5")

set(GLOG_BUILD_MD5_CHECKSUM "MD5=5df6d78b81e51b90ac0ecd7ed932b0d4")


if(DEFINED ENV{ARROW_GRPC_URL})
  set(GRPC_SOURCE_URL "$ENV{ARROW_GRPC_URL}")
else()
  set_urls(
    GRPC_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/grpc-v1.25.0.tar.gz"
    "https://github.com/grpc/grpc/archive/v1.25.0.tar.gz")
endif()

set(ARROW_GRPC_BUILD_VERSION "1.25.0")

set(GRPC_BUILD_MD5_CHECKSUM "MD5=3a875f7b3f0e3bdd3a603500bcef3d41")


if(DEFINED ENV{ARROW_GTEST_URL})
  set(GTEST_SOURCE_URL "$ENV{ARROW_GTEST_URL}")
else()
  set_urls(
    GTEST_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/gtest-1.8.1.tar.gz"
    "https://github.com/google/googletest/archive/release-1.8.1.tar.gz"
    "https://dl.bintray.com/ursalabs/arrow-gtest/gtest-1.8.1.tar.gz"
    "https://chromium.googlesource.com/external/github.com/google/googletest/+archive/release-1.8.1.tar.gz")
endif()

set(ARROW_GTEST_BUILD_VERSION "1.8.1")

set(GTEST_BUILD_MD5_CHECKSUM "MD5=2e6fbeb6a91310a16efe181886c59596")


if(DEFINED ENV{ARROW_JEMALLOC_URL})
  set(JEMALLOC_SOURCE_URL "$ENV{ARROW_JEMALLOC_URL}")
else()
  set_urls(
    JEMALLOC_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/jemalloc-5.2.1.tar.bz2"
    "https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2")
endif()

set(ARROW_JEMALLOC_BUILD_VERSION "5.2.1")

set(JEMALLOC_BUILD_MD5_CHECKSUM "MD5=3d41fbf006e6ebffd489bdb304d009ae")


if(DEFINED ENV{ARROW_LZ4_URL})
  set(LZ4_SOURCE_URL "$ENV{ARROW_LZ4_URL}")
else()
  set_urls(
    LZ4_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/lz4-v1.9.2.tar.gz"
    "https://github.com/lz4/lz4/archive/v1.9.2.tar.gz")
endif()

set(ARROW_LZ4_BUILD_VERSION "1.9.2")

set(LZ4_BUILD_MD5_CHECKSUM "MD5=3898c56c82fb3d9455aefd48db48eaad")


if(DEFINED ENV{ARROW_MIMALLOC_URL})
  set(MIMALLOC_SOURCE_URL "$ENV{ARROW_MIMALLOC_URL}")
else()
  set_urls(
    MIMALLOC_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/mimalloc-270e765454f98e8bab9d42609b153425f749fff6.tar.gz"
    "https://github.com/microsoft/mimalloc/archive/270e765454f98e8bab9d42609b153425f749fff6.tar.gz")
endif()

set(ARROW_MIMALLOC_BUILD_VERSION "270e765454f98e8bab9d42609b153425f749fff6")

set(MIMALLOC_BUILD_MD5_CHECKSUM "MD5=ff46129827bfc5f35dedbf5a6f301aa2")


if(DEFINED ENV{ARROW_ORC_URL})
  set(ORC_SOURCE_URL "$ENV{ARROW_ORC_URL}")
else()
  set_urls(
    ORC_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/orc-1.6.2.tar.gz"
    "https://github.com/apache/orc/archive/rel/release-1.6.2.tar.gz")
endif()

set(ARROW_ORC_BUILD_VERSION "1.6.2")

set(ORC_BUILD_MD5_CHECKSUM "MD5=b17e027fab4c82a19929b272fd6b4269")


if(DEFINED ENV{ARROW_PROTOBUF_URL})
  set(PROTOBUF_SOURCE_URL "$ENV{ARROW_PROTOBUF_URL}")
else()
  set_urls(
    PROTOBUF_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/protobuf-3.7.1.tar.gz"
    "https://github.com/google/protobuf/releases/download/v3.7.1/protobuf-all-3.7.1.tar.gz")
endif()

set(ARROW_PROTOBUF_BUILD_VERSION "3.7.1")

set(PROTOBUF_BUILD_MD5_CHECKSUM "MD5=cda6ae370a5df941f8aa837c8a0292ba")


if(DEFINED ENV{ARROW_RAPIDJSON_URL})
  set(RAPIDJSON_SOURCE_URL "$ENV{ARROW_RAPIDJSON_URL}")
else()
  set_urls(
    RAPIDJSON_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/rapidjson-2bbd33b33217ff4a73434ebf10cdac41e2ef5e34.tar.gz"
    "https://github.com/miloyip/rapidjson/archive/2bbd33b33217ff4a73434ebf10cdac41e2ef5e34.tar.gz")
endif()

set(ARROW_RAPIDJSON_BUILD_VERSION "2bbd33b33217ff4a73434ebf10cdac41e2ef5e34")

set(RAPIDJSON_BUILD_MD5_CHECKSUM "MD5=f1db3c5449677f148f60425d88481d76")


if(DEFINED ENV{ARROW_RE2_URL})
  set(RE2_SOURCE_URL "$ENV{ARROW_RE2_URL}")
else()
  set_urls(
    RE2_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/re2-2019-08-01.tar.gz"
    "https://github.com/google/re2/archive/2019-08-01.tar.gz")
endif()

set(ARROW_RE2_BUILD_VERSION "2019-08-01")

set(RE2_BUILD_MD5_CHECKSUM "MD5=b38416cfa4c9dc68651292b5137989e0")


if(DEFINED ENV{ARROW_SNAPPY_URL})
  set(SNAPPY_SOURCE_URL "$ENV{ARROW_SNAPPY_URL}")
else()
  set_urls(
    SNAPPY_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/snappy-1.1.7.tar.gz"
    "https://github.com/google/snappy/archive/1.1.7.tar.gz")
endif()

set(ARROW_SNAPPY_BUILD_VERSION "1.1.7")

set(SNAPPY_BUILD_MD5_CHECKSUM "MD5=ee9086291c9ae8deb4dac5e0b85bf54a")


if(DEFINED ENV{ARROW_THRIFT_URL})
  set(THRIFT_SOURCE_URL "$ENV{ARROW_THRIFT_URL}")
else()
  set_urls(
    THRIFT_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/thrift-0.12.0.tar.gz"
    "https://dl.bintray.com/ursalabs/arrow-thrift/thrift-0.12.0.tar.gz"
    "https://archive.apache.org/dist/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://downloads.apache.org/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://apache.claz.org/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://apache.cs.utah.edu/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://apache.mirrors.lucidnetworks.net/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://apache.osuosl.org/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://ftp.wayne.edu/apache/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://mirror.olnevhost.net/pub/apache/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://mirrors.gigenet.com/apache/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://mirrors.koehn.com/apache/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://mirrors.ocf.berkeley.edu/apache/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://mirrors.sonic.net/apache/thrift/0.12.0/thrift-0.12.0.tar.gz"
    "https://us.mirrors.quenda.co/apache/thrift/0.12.0/thrift-0.12.0.tar.gz")
endif()

set(ARROW_THRIFT_BUILD_VERSION "0.12.0")

set(THRIFT_BUILD_MD5_CHECKSUM "MD5=3deebbb4d1ca77dd9c9e009a1ea02183")


if(DEFINED ENV{ARROW_ZLIB_URL})
  set(ZLIB_SOURCE_URL "$ENV{ARROW_ZLIB_URL}")
else()
  set_urls(
    ZLIB_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/zlib-1.2.11.tar.gz"
    "https://zlib.net/fossils/zlib-1.2.11.tar.gz")
endif()

set(ARROW_ZLIB_BUILD_VERSION "1.2.11")

set(ZLIB_BUILD_MD5_CHECKSUM "MD5=1c9f62f0778697a09d36121ead88e08e")


if(DEFINED ENV{ARROW_ZSTD_URL})
  set(ZSTD_SOURCE_URL "$ENV{ARROW_ZSTD_URL}")
else()
  set_urls(
    ZSTD_SOURCE_URL
    "https://github.com/ursa-labs/thirdparty/releases/download/latest/zstd-v1.4.3.tar.gz"
    "https://github.com/facebook/zstd/archive/v1.4.3.tar.gz")
endif()

set(ARROW_ZSTD_BUILD_VERSION "1.4.3")

set(ZSTD_BUILD_MD5_CHECKSUM "MD5=dd73dd3b6e5efd8946e6c5e7fe7cb1d2")

