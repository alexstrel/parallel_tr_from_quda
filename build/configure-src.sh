#!/bin/bash

export QUDA_GPU_ARCH=sm_75

export CXX=/usr/bin/g++
export CC=/usr/bin/gcc

cmake .. \
        -DQUDA_BUILD_SHAREDLIB=ON \
        -DQUDA_MPI=ON \
        -DCMAKE_EXE_LINKER_FLAGS_SANITIZE="-fsanitize=address,undefined" \
        -DCMAKE_BUILD_TYPE=HOSTDEBUG


