#!/usr/bin/env bash
# Serve a model directory from this project (e.g. a merged GRPO checkpoint) for the eval harness.
#   scripts/serve_local_model.sh $PWD/runs/X/merged NAME [PORT] [GPU_UTIL]
set -euo pipefail
DIR=${1:?model dir}; NAME=${2:?served name}; PORT=${3:-8010}; UTIL=${4:-0.25}
P=${P:-$(cd "$(dirname "$0")/.." && pwd)}
docker rm -f smol-eval >/dev/null 2>&1 || true
docker run -d --name smol-eval --gpus all --ipc host -p "${PORT}:8000" -e HF_HUB_OFFLINE=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$P:$P:ro" \
  vllm/vllm-openai:v0.27.1-aarch64 \
  --model "$DIR" --served-model-name "$NAME" --gpu-memory-utilization "$UTIL" \
  --max-model-len 32768 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && { echo "ready after $((i*5))s"; exit 0; }
  docker ps -q --filter name=smol-eval | grep -q . || { echo "container exited"; docker logs --tail 60 smol-eval; exit 1; }
  sleep 5
done
echo "not ready after 15 min"; exit 1
