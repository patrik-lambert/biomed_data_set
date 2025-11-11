#!/bin/bash
pkg install python --yes
apt update
apt install git cmake vulkan-tools vulkan-headers shaderc vulkan-loader-android --yes
apt install libc++

git clone https://github.com/tetherto/qvac-ext-lib-llama.cpp #
cd qvac-ext-lib-llama.cpp #
git checkout temp-latest #
#git config --global core.pager cat

cmake -B build -DGGML_VULKAN=1 &> log.cmake.txt #
cmake --build build --config Debug -j2 &> log.build.txt #

less log.cmake.txt #
tail log.build.txt #