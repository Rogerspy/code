#!/usr/bin/env bash
# 启动含光 PPU 的 PyTorch 容器（Aliyun/T-Head inference-xpu-pytorch 镜像）
# 镜像内置 torch2.9 + transformers4.57 + triton3.5 + numpy1.26 + vllm0.18(ppu)
# PPU 经 PJRT_DEVICE=CUDA + UMD 驱动暴露为 CUDA。需 docker 免密 sudo（已配置）。
set -euo pipefail
IMAGE="egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun/inference-xpu-pytorch:26.01-v2.0.0-vllm0.18.0-torch2.9-cu129-20260330"
NAME="flagos-ppu"
HOSTROOT="/home/qiyiyan/songchao"
DEV="${CUDA_VISIBLE_DEVICES:-2}"   # PPU0 被同事 vllm 占用，用空闲的 2
mkdir -p "$HOSTROOT/.cache"
[ -f "$HOSTROOT/dnshost" ] || : > "$HOSTROOT/dnshost"   # entrypoint.sh 会 cat 它

sudo docker rm -f "$NAME" 2>/dev/null || true
sudo docker run -d --name "$NAME" \
  --network=host --privileged --init --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES="$DEV" \
  -e UMD_PLATFORM_TYPE=1 \
  -e HGGC_DRIVER_CANDIDATE=UMD \
  -e PJRT_DEVICE=CUDA \
  -e CHECKPOINTS_DIR=/workspace/checkpoints \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HOME=/cache/hf \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e OMP_NUM_THREADS=8 \
  -v "${HOSTROOT}:/workspace" \
  -v "${HOSTROOT}/.cache:/cache" \
  "$IMAGE" sleep infinity

echo "=== wait + validate torch on PPU (device $DEV) ==="
sleep 4
sudo docker exec -i "$NAME" /workspace/code/_common/crun.sh python <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = float((x @ x).sum())
    print("matmul on PPU OK:", y)
PY
echo "container $NAME ready. exec: sudo docker exec -it $NAME bash"
