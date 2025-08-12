#!/bin/bash
set -x
clear

# set cross build toolchain and k230 sdk(Linux)
export PATH=$PATH:`pwd`/../toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V2.10.1/bin
build_type=Release
# build_type=Debug

build=build
rm -rf ${build}
mkdir ${build}
pushd ${build}
cmake -DCMAKE_BUILD_TYPE=${build_type}           \
      -DCMAKE_INSTALL_PREFIX=`pwd`               \
      -DCMAKE_TOOLCHAIN_FILE=cmake/Riscv64.cmake \
      -DDUMP_PROFILE_INFO=1 \
      ..

make -j
popd
