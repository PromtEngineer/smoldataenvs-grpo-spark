#!/usr/bin/env bash
# One baseline end to end: wait for weights, serve (or wait for a remote server), smoke test, full runs.
#   scripts/baseline_chain.sh MODEL TAG [DL_LOG]
# Env knobs:
#   EP=http://host:port/v1   endpoint (default http://127.0.0.1:8010/v1, served here by serve_vllm.sh)
#   REMOTE=1                 do not serve here; wait until EP lists MODEL (another node serves it)
#   SERVE_EXTRA="..."        extra vllm args for serve_vllm.sh, e.g. "--dtype bfloat16"
#   GPU_UTIL=0.25            vllm --gpu-memory-utilization when serving here
#   THINK=0|low              0 (default): --no-think; otherwise sent as reasoning_effort
#   MAX_TOKENS=1024          per model call
#   MODES="program agent"    which full runs to do
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?model}; TAG=${2:?tag}; DL_LOG=${3:-}
EP=${EP:-http://127.0.0.1:8010/v1}; GPU_UTIL=${GPU_UTIL:-0.25}; THINK=${THINK:-0}
MAX_TOKENS=${MAX_TOKENS:-1024}; MODES=${MODES:-"program agent"}
PY=.venv/bin/python
log() { echo "[$(date +%T)] $*"; }
if [ -n "$DL_LOG" ]; then
  log "waiting for download ($DL_LOG)"
  until [ -f "$DL_LOG" ] && grep -q DL_EXIT "$DL_LOG"; do sleep 15; done
  grep -q "DL_EXIT=0" "$DL_LOG" || { log "download failed"; exit 1; }
fi
if [ "${REMOTE:-0}" = 1 ]; then
  log "waiting for $MODEL at $EP"
  until curl -sf "$EP/models" | grep -q "\"$MODEL\""; do sleep 30; done
elif curl -sf "$EP/models" | grep -q "\"$MODEL\""; then
  log "$MODEL already served at $EP"
else
  log "serving $MODEL"
  # shellcheck disable=SC2086
  scripts/serve_vllm.sh "$MODEL" 8010 "$GPU_UTIL" ${SERVE_EXTRA:-} || { log "serve failed"; exit 1; }
fi
COMMON=(--endpoint "$EP" --model "$MODEL" --slice eval_plus_test --sandbox docker --max-tokens "$MAX_TOKENS")
if [ "$THINK" = 0 ]; then COMMON+=(--no-think); else COMMON+=(--extra-body "{\"reasoning_effort\":\"$THINK\"}"); fi
log "smoke test (8 tasks, program mode)"
$PY eval_harness.py run "${COMMON[@]}" --mode program --limit 8 --concurrency 8 --out "runs/smoke-$TAG" || { log "smoke run crashed"; exit 1; }
# plumbing check, not a quality bar (a base 2B scores ~5%, so 0/8 correct is normal):
# the model must return code and that code must actually execute in the sandbox
executed=$($PY -c "import json;r=[json.loads(l) for l in open('runs/smoke-$TAG/results.jsonl')];print(sum(1 for x in r if (x.get('completion') or '').strip() and ((x.get('prediction') or '').strip() or 'Error' in (x.get('stderr') or ''))))")
log "smoke: $executed/8 completions produced code that executed in the sandbox"
[ "$executed" -ge 6 ] || { log "smoke: too few programs executed; stopping"; exit 1; }
for mode in $MODES; do
  log "full run: $mode mode (394 tasks)"
  if [ "$mode" = agent ]; then
    $PY eval_harness.py run "${COMMON[@]}" --mode agent --max-turns 12 --concurrency 16 --out "runs/A0-$TAG-agent"
  else
    $PY eval_harness.py run "${COMMON[@]}" --mode program --concurrency 16 --out "runs/A0-$TAG-program"
    $PY checker_v2.py --jsonl "runs/A0-$TAG-program/results.jsonl" > "runs/A0-$TAG-program/checker_v2.txt" 2>&1
  fi
done
log "CHAIN_DONE"
