#!/usr/bin/env bash
# Examine checkpoints of a training run on THIS node (no second node): merge, serve, 394-task agent exam,
# compare vs the untrained 4B and vs milestone M1, write the dashboard's heldout_step file.
#   scripts/exam_ckpts_local.sh runs/grpo-agent-4b-m2 4b-m2rl 30 41
set -uo pipefail
cd "$(dirname "$0")/.."
P=${P:-$(cd "$(dirname "$0")/.." && pwd)}
RUN=${1:?run dir}; TAG=${2:?exam tag}; shift 2
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for s in "$@"; do
  EX=runs/T${s}-$TAG-agent
  { [ -f $EX/EXAM_DONE ] || [ -f $EX/EXAM_FAILED ]; } && { log "skip $s (done)"; continue; }
  mkdir -p $EX
  log "STAGE exam checkpoint $s"
  if [ ! -f $RUN/checkpoint-$s/merged/model.safetensors ]; then
    docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HOME=${HF_HOME_HOST:-$HOME/.cache/huggingface} -e HF_HUB_OFFLINE=1 \
      -v $P:$P -v ${HF_HOME_HOST:-$HOME/.cache/huggingface}:${HF_HOME_HOST:-$HOME/.cache/huggingface} -w $P smol-train:v2 \
      python3 scripts/merge_lora.py $RUN/checkpoint-$s $RUN/checkpoint-$s/merged > logs/merge-$TAG-ckpt$s.log 2>&1 \
      || { log "FAILED merge $s"; touch $EX/EXAM_FAILED; continue; }
  fi
  scripts/serve_local_model.sh $P/$RUN/checkpoint-$s/merged smol-$TAG-$s 8010 0.4 > logs/serve-$TAG-ckpt$s.log 2>&1 \
    || { log "FAILED serve $s"; docker rm -f smol-eval >/dev/null 2>&1; touch $EX/EXAM_FAILED; continue; }
  .venv/bin/python eval_harness.py run --endpoint http://127.0.0.1:8010/v1 --model smol-$TAG-$s \
    --slice eval_plus_test --no-think --sandbox docker --mode agent --max-turns 12 --concurrency 16 \
    --out $EX > logs/exam-$TAG-ckpt$s.log 2>&1
  docker rm -f smol-eval >/dev/null 2>&1
  .venv/bin/python compare_runs.py runs/A0-qwen35-4b-agent $EX > $EX/compare_vs_base.json 2>&1
  .venv/bin/python compare_runs.py runs/T120-4b-agent-long-agent $EX > $EX/compare_vs_m1.json 2>&1
  .venv/bin/python scripts/ckpt_heldout.py $EX $RUN $s
  touch $EX/EXAM_DONE
  log "exam checkpoint $s done"
done
log "LOCAL_EXAMS_DONE $RUN"
