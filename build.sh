#!/bin/bash

# 定义根目录
root_dir=$(pwd)

# 清理并创建输出目录
rm -rf out
mkdir out

# 创建 k230_bin 目录
k230_bin="${root_dir}/k230_bin"
rm -rf ${k230_bin}
mkdir -p ${k230_bin}

# 编译 ui_and_ai
pushd out
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=`pwd` \
      -DCMAKE_C_COMPILER=/opt/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.0.2/bin/riscv64-unknown-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=/opt/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.0.2/bin/riscv64-unknown-linux-gnu-c++ \
      "${root_dir}/src/ui_and_ai"
make -j && make install
popd

# 复制 ui_and_ai 到 k230_bin
if [ -f out/bin/ui_and_ai ]; then
    cp out/bin/ui_and_ai ${k230_bin}
fi

# 复制 utils 目录到 k230_bin
cp -r "${root_dir}/src/ui_and_ai/utils"/* ${k230_bin}

# 确保情绪模型存在（复用 ai_demo 下载的资源）
emo_model_src="${root_dir}/../ai_demo/kmodel/ai_poc/kmodel/face_emotion.kmodel"
if [ -f "${emo_model_src}" ]; then
    cp "${emo_model_src}" "${k230_bin}/"
else
    echo "warning: face_emotion.kmodel not found at ${emo_model_src}"
fi

# 清理 ui_and_ai 的输出目录
rm -rf out

# 重新创建输出目录
mkdir out

# 编译 xiaozhi_client
pushd out
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=`pwd` \
      -DCMAKE_C_COMPILER=/opt/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.0.2/bin/riscv64-unknown-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=/opt/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.0.2/bin/riscv64-unknown-linux-gnu-g++ \
      "${root_dir}/src/xiaozhi_client"
make -j && make install
popd

# 复制 xiaozhi_client 到 k230_bin
if [ -f out/bin/xiaozhi_client ]; then
    cp out/bin/xiaozhi_client ${k230_bin}
    cp ${root_dir}/src/xiaozhi_client/snd_core/wakeup_audio.pcm ${k230_bin}
fi

# 清理 xiaozhi_client 的输出目录
rm -rf out