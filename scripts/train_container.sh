#!/usr/bin/env bash
# Run train_grpo_local.py in the smol-train image on this Spark. Sandboxes still start on the HOST
# (docker socket), with the host venv as their interpreter. A memory watchdog stops the run if
# MemAvailable stays under 5 GB (GB10 hard-freezes instead of OOM-killing).
#   scripts/train_container.sh --profile spark_4b_lora --pilot
set -uo pipefail
P=${P:-$(cd "$(dirname "$0")/.." && pwd)}
docker rm -f smol-train >/dev/null 2>&1
( while sleep 5; do
    docker ps -q --filter name=^smol-train$ | grep -q . || { sleep 30; docker ps -q --filter name=^smol-train$ | grep -q . || exit 0; }
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt 5 ]; then sleep 5; a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
      [ "$a" -lt 5 ] && { echo "[watchdog] MemAvailable ${a} GB: stopping smol-train"; docker rm -f smol-train; exit 1; }; fi
  done ) &
docker run --rm --name smol-train --gpus all --ipc host --network host \
  --user "$(id -u):$(id -g)" --group-add "$(getent group docker | cut -d: -f3)" \
  -e HOME=/tmp -e HF_HOME=${HF_HOME_HOST:-$HOME/.cache/huggingface} -e HF_HUB_OFFLINE=1 \
  -e SMOLENV_SANDBOX_PYTHON=$P/.venv/bin/python -e PYTHONUNBUFFERED=1 \
  -v $P:$P -v ${HF_HOME_HOST:-$HOME/.cache/huggingface}:${HF_HOME_HOST:-$HOME/.cache/huggingface} \
  -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker:ro \
  -w $P smol-train:local \
  ${LAUNCH:-torchrun --standalone --nproc_per_node 1} train_grpo_local.py "$@"
