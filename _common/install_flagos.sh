#!/usr/bin/env bash
# FlagOS 安装：按 flagos-ai/skills install-stack-flagos 文档的 5 包顺序，针对含光 PPU(CUDA-compat) 适配。
# 容器内运行： sudo docker exec flagos-ppu bash /workspace/code/_common/install_flagos.sh
set -euo pipefail
PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
GH_PREFIX="https://ghfast.top/https://github.com"

echo "### baseline ###"
python -c "import torch,transformers; print('torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available())"

echo "### Step 1: model deps (sentence-transformers / FlagEmbedding / transformers>=4.51) ###"
python -m pip install $PIP_MIRROR -U "sentence-transformers>=3.0" "transformers>=4.51" FlagEmbedding einops
# jina-embeddings-v3 要求 numpy<2；与其它包可能冲突，best-effort 单独装
python -m pip install $PIP_MIRROR "numpy<2" || echo "[warn] numpy<2 与其它包冲突，运行 jina 时按需单独环境处理"

echo "### Step 2: FlagOS install-stack (adapted for 含光/CUDA-compat) ###"
# 2.1 vLLM：容器自带 vllm 0.18（阿里云 PPU 补丁版）；不降级到 0.13.0（会破坏 PPU 支持）。偏差已记录。
echo "[vLLM] container ships $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo n/a); keep as-is (downgrade to 0.13.0 breaks PPU patches)"

# 2.2 FlagTree：预编译 wheel 来自 FlagOS PyPI(resource.flagos.net)，按 vendor+python+glibc 匹配；含光不在厂商表 -> NOT_FOUND -> 跳过（per skill NOT_FOUND policy）
echo "[FlagTree] 含光无 vendor wheel in FlagOS PyPI -> skip (NOT_FOUND policy)"

# 2.3 FlagGems：FlagOS 核心算子库（Triton），从源码装
echo "[FlagGems] install from source ..."
cd /tmp && rm -rf FlagGems
if timeout 150 git clone ${GH_PREFIX}/FlagOpen/FlagGems; then
  cd FlagGems && python -m pip install $PIP_MIRROR -e . || echo "[warn] FlagGems install partial"
else
  echo "[warn] FlagGems clone failed (network) -> skip"
fi

# 2.4 FlagCX：通信库，make 按 vendor flag；含光无 FLAGCX_ADAPTOR -> 跳过
echo "[FlagCX] 含光无 FLAGCX_ADAPTOR in vendor-mappings -> skip"

# 2.5 vllm-plugin-FL：绑定 vllm 0.13.0；容器为 0.18 -> 跳过
echo "[vllm-plugin-FL] pinned vllm 0.13.0 conflicts container 0.18 -> skip"

echo "### Step 3: validate ###"
python - <<'PY'
import importlib
for m in ['torch','transformers','sentence_transformers','FlagEmbedding']:
    try:
        importlib.import_module(m); print('OK', m)
    except Exception as e:
        print('FAIL', m, e)
import torch
print('cuda?', torch.cuda.is_available(), 'count', torch.cuda.device_count())
try:
    import flag_gems; print('OK flag_gems', getattr(flag_gems,'__version__','?'))
except Exception as e:
    print('flag_gems not importable:', e)
PY
echo "FlagOS install finished. Status: PARTIAL (FlagGems core installed; FlagTree/FlagCX/vllm-plugin-FL skipped per 含光 vendor policy)."
