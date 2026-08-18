#!/usr/bin/env bash
# 容器内命令前置：配置含光 PPU（CUDA-compat）环境，再 exec 实际命令
source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1 || true
export UMD_PLATFORM_TYPE=1
export HGGC_DRIVER_CANDIDATE=UMD
export PJRT_DEVICE=CUDA
exec "$@"
