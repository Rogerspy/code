#!/usr/bin/env bash
# 顺序运行剩余 4 个模型的 train+infer，每个捕获状态与日志。
# 容器 flagos-ppu 内通过 crun.sh 配置 PPU(CUDA-compat)环境后执行。
set -uo pipefail
HOSTROOT=/home/qiyiyan/songchao
MODELS=(Qwen3-Embedding-0.6B Qwen3-Reranker-0.6B bge-reranker-large jina-embeddings-v3)
SUMMARY="$HOSTROOT/code/_common/run_summary.txt"
: > "$SUMMARY"

for m in "${MODELS[@]}"; do
  d="$HOSTROOT/code/$m"
  echo "########## $m ##########" | tee -a "$SUMMARY"

  tlog="$d/run.train.log"
  echo "[train] $m ..." | tee -a "$SUMMARY"
  if sudo docker exec -i -w "/workspace/code/$m" flagos-ppu \
       /workspace/code/_common/crun.sh python train.py > "$tlog" 2>&1; then
    tst=ok
  else
    tst=FAIL
  fi
  tok=$(grep -c "TRAIN_OK" "$tlog" 2>/dev/null || echo 0)
  echo "$m train: $tst (TRAIN_OK=$tok)" | tee -a "$SUMMARY"
  echo "  tail train:" | tee -a "$SUMMARY"
  tail -3 "$tlog" 2>/dev/null | sed 's/^/    /' | tee -a "$SUMMARY"

  ilog="$d/run.infer.log"
  echo "[infer] $m ..." | tee -a "$SUMMARY"
  if sudo docker exec -i -w "/workspace/code/$m" flagos-ppu \
       /workspace/code/_common/crun.sh python infer.py > "$ilog" 2>&1; then
    ist=ok
  else
    ist=FAIL
  fi
  iok=$(grep -c "INFER_OK" "$ilog" 2>/dev/null || echo 0)
  echo "$m infer: $ist (INFER_OK=$iok)" | tee -a "$SUMMARY"
  echo "  tail infer:" | tee -a "$SUMMARY"
  tail -3 "$ilog" 2>/dev/null | sed 's/^/    /' | tee -a "$SUMMARY"
done
echo "########## ALL DONE ##########" | tee -a "$SUMMARY"
