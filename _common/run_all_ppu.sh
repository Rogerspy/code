#!/usr/bin/env bash
# 在含光 PPU 上重新训练+推理 5 个模型,全程记录清晰日志,并采样 ppu-smi
# 证明计算发生在物理 PPU2(CUDA_VISIBLE_DEVICES=2),非 nvidia、非 CPU。
set -uo pipefail
HOSTROOT=/home/qiyiyan/songchao
DEV=2   # 容器 CUDA_VISIBLE_DEVICES=2 -> 物理 PPU 2

# ppu_mem N  -> "xMiB / 98304MiB"(物理 PPU N 的显存)
ppu_mem() { ppu-smi 2>/dev/null | grep -A1 "^| $1  PPU" | tail -1 | grep -oE "[0-9]+MiB / [0-9]+MiB" | head -1; }
ppu_mb()  { ppu_mem "$1" | grep -oE "^[0-9]+"; }

MODELS=(Qwen3-Embedding-0.6B bge-large-zh-v1.5 Qwen3-Reranker-0.6B bge-reranker-large jina-embeddings-v3)
SUMMARY="$HOSTROOT/code/_common/run_ppu_summary.txt"; : > "$SUMMARY"

echo "ppu-smi: $(ppu-smi 2>/dev/null | grep -iE 'PPU-SMI|Driver Version' | head -1)" | tee -a "$SUMMARY"
echo "physical PPU $DEV idle before: $(ppu_mem $DEV)" | tee -a "$SUMMARY"
echo "(nvidia-smi check: $(command -v nvidia-smi || echo 'no nvidia-smi -> 非 NVIDIA 机器'))" | tee -a "$SUMMARY"

for m in "${MODELS[@]}"; do
  d="$HOSTROOT/code/$m"; tlog="$d/train.log"; ilog="$d/infer.log"; smp="$d/ppu_smi.train.sample"
  echo "########## $m ##########" | tee -a "$SUMMARY"

  echo "[train] ppu$DEV mem before: $(ppu_mem $DEV)" | tee -a "$SUMMARY"
  rm -f "$smp"
  # 后台采样:每 1.5s 记录物理 ppu$DEV 的已用显存(MiB)
  ( for i in $(seq 1 50); do ts=$(date +%H:%M:%S); echo "$ts $(ppu_mb $DEV)"; sleep 1.5; done ) > "$smp" 2>&1 &
  SMIPID=$!

  sudo docker exec -i -w "/workspace/code/$m" flagos-ppu \
    /workspace/code/_common/crun.sh python train.py > "$tlog" 2>&1
  texit=$?
  kill "$SMIPID" 2>/dev/null; wait "$SMIPID" 2>/dev/null
  tok=$(grep -c "TRAIN_OK" "$tlog")
  peak=$(sort -k2 -n "$smp" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "[train] exit=$texit TRAIN_OK=$tok  ppu$DEV mem after: $(ppu_mem $DEV)  peak_during_train=${peak}MiB" | tee -a "$SUMMARY"
  grep -E "PPU-EVIDENCE|device\(0\)\.name|first_param|PASS|mem after|TRAIN_OK|epoch|loss" "$tlog" 2>/dev/null | head -30 | sed 's/^/    /' | tee -a "$SUMMARY"

  echo "[infer] ppu$DEV mem before: $(ppu_mem $DEV)" | tee -a "$SUMMARY"
  sudo docker exec -i -w "/workspace/code/$m" flagos-ppu \
    /workspace/code/_common/crun.sh python infer.py > "$ilog" 2>&1
  iexit=$?; iok=$(grep -c "INFER_OK" "$ilog")
  echo "[infer] exit=$iexit INFER_OK=$iok  ppu$DEV mem after: $(ppu_mem $DEV)" | tee -a "$SUMMARY"
  grep -E "PPU-EVIDENCE|first_param|INFER_OK|embedding dim|dim" "$ilog" 2>/dev/null | head -15 | sed 's/^/    /' | tee -a "$SUMMARY"
done
echo "########## ALL DONE ##########" | tee -a "$SUMMARY"
