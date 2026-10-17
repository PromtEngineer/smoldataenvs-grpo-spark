#!/usr/bin/env bash
# Gemini reference run on this Spark. Reads the API key from stdin (one line), so it never
# lands in argv, the environment of other processes, or a file.
#   printf '%s\n' "$GEMINI_API_KEY" | scripts/run_gemini.sh MODE OUT [extra eval_harness args...]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?program|agent}; OUT=${2:?out dir}; shift 2
IFS= read -r GEMINI_API_KEY; export GEMINI_API_KEY
exec .venv/bin/python eval_harness.py run \
  --endpoint https://generativelanguage.googleapis.com/v1beta/openai --model gemini-3.8-flash \
  --api-key-env GEMINI_API_KEY --extra-body '{"reasoning_effort":"low"}' --max-tokens 4096 \
  --slice eval_plus_test --mode "$MODE" --sandbox docker --concurrency 8 --out "$OUT" "$@"
