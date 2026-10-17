#!/usr/bin/env bash
# Agent-mode pre-screen on one shard of train tasks: 4 samples per task at the training temperature (0.8)
# and turn budget (10 tool calls + 1), so build_pool.py can keep the tasks the model solves sometimes but
# not always (the only groups that carry a GRPO gradient). Screens the untrained 4B unless MODEL_DIR is set.
#   scripts/prescreen_agent.sh data/prescreen_agent_shard1.json runs/prescreen-4b-agent-node1 [UTIL] [CONCURRENCY]
#   MODEL_DIR=$PWD/milestones/M1-agent-grpo-4b-step120/model/merged MODEL_NAME=smol-4b-m1 \
#     scripts/prescreen_agent.sh data/m2_rl_candidates.json runs/prescreen-m1-agent 0.4 40
set -uo pipefail
cd "$(dirname "$0")/.."
SHARD=${1:?shard json}; OUT=${2:?out dir}; UTIL=${3:-0.35}; CONC=${4:-32}
if [ -n "${MODEL_DIR:-}" ]; then
  NAME=${MODEL_NAME:?set MODEL_NAME with MODEL_DIR}
  scripts/serve_local_model.sh "$MODEL_DIR" "$NAME" 8010 "$UTIL" || exit 1
else
  NAME=Qwen/Qwen3.5-4B
  scripts/serve_vllm.sh "$NAME" 8010 "$UTIL" || exit 1
fi
.venv/bin/python eval_harness.py run --endpoint http://127.0.0.1:8010/v1 --model "$NAME" \
  --split train --task-ids "$SHARD" --n-samples 4 --temperature 0.8 --mode agent --max-turns 11 \
  --no-think --sandbox docker --concurrency "$CONC" --out "$OUT"
rc=$?
docker rm -f smol-eval >/dev/null 2>&1
exit $rc
