apt update
apt install git cmake vulkan-tools vulkan-headers shaderc vulkan-loader-android --yes
apt install libc++

git clone https://github.com/tetherto/qvac-ext-lib-llama.cpp
cd qvac-ext-lib-llama.cpp
git fetch origin pull/33/head:pr33
git checkout pr33
git config --global core.pager cat

cmake -B build -DGGML_VULKAN=1
cmake --build build --config Debug -j2

