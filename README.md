# GRPO on a DGX Spark: open RL environments for a small model

This repository trains a small open model, Qwen3.5-4B, with reinforcement learning (GRPO) on
[SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs), Hugging Face FineEnvs' set of
5,394 verifiable data-analysis tasks. Everything runs locally on two NVIDIA DGX Sparks. It contains the
eval harness, the sandbox, a strict answer checker, the training scripts, and the exam results behind the
video.

The short version: RL raised the 4B model from 51.8% to 58.4% on a 394-task held-out exam (paired
McNemar p = 0.001) and closed more than half the gap to Qwen3.5-27B (64.0%). After that, four
different training routes all landed at about 58%.

## Results

All scores are strict-checker pass@1 on the same 394 held-out tasks, in agent mode (the model works in a
`bash` terminal and writes its answer to a file), with greedy decoding.

| Model | Exam | Unseen datasets (87 tasks) |
|---|---|---|
| Qwen3.5-4B, untrained | 51.8% | 52.9% |
| RL on random tasks (40 steps) | 54.1% (p = 0.29) | |
| RL on pre-screened "edge" tasks (120 steps, 19.1 h) | **58.4%** (p = 0.001) | 58.6% |
| More RL from there | 58.1% | 54.0% |
| Distilled from the 27B's transcripts | 58.1% | 57.5% |
| Distilled, then RL | 58.6% | **62.1%** (p = 0.04 vs untrained) |
| Qwen3.5-27B-FP8, untrained (target) | 64.0% | 64.4% |

What the numbers say:

- **Task selection matters more than steps.** On random tasks, 55% to 70% of GRPO groups were all right
  or all wrong, so they carried no gradient. Pre-screening 999 tasks (4 tries each) and keeping the 240
  the model solved only sometimes is what made RL work.
- **RL taught finishing, not data analysis.** The main change was fewer episodes ending without an
  answer (99 to 72) and fewer answers in the wrong form, such as a number where a name was asked
  (17 to 6 on 140 name questions).
- **The ceiling is the base model.** Untrained, the 4B solves 62.3% of training tasks in one try and
  72.9% within four, so RL sharpening alone tops out in the low 60s.
- **No reward hacking.** The reward is the strict checker. Answers that pass only the dataset's lenient
  grader stayed at plain near-miss numbers, with none padded with words.

`results/exams/` has the summary of every exam in the table, plus the checkpoint curve of the 120-step
run (`T20` to `T120`). Each summary breaks scores down by tier and by seen or unseen dataset.

## How it works

| Piece | File |
|---|---|
| Local environment: sandboxed executor, rollouts, reward | `smolenv_local.py` |
| Eval harness for any OpenAI-compatible endpoint | `eval_harness.py` |
| Strict checker (the reward) | `checker_v2.py` |
| Agent-mode GRPO with TRL `environment_factory` | `train_grpo_agent.py` |
| Program-mode GRPO and the shared training plumbing | `train_grpo_local.py` |
| Distillation (SFT on the 27B's transcripts) | `train_sft_agent.py`, `scripts/build_sft_27b.py` |
| Pre-screen and task pool | `scripts/prescreen_agent.sh`, `scripts/build_pool.py` |
| Training profiles and hyperparameters | `train_plan.yaml` |
| Live training dashboard | `dashboard/` |
| Transcript audits for leakage and grader gaming | `scripts/audit_agent.py`, `scripts/audit_flags.py` |

The sandbox replaces FineEnvs' hosted sandboxes with a local one. On the Sparks it uses Docker:
every model command runs in a throwaway container with no network, no capabilities, a read-only root,
and only the task's input files mounted. On macOS it uses `sandbox-exec`.

The dataset ships with a lenient grader that takes the first number it finds in an answer, so
`42, yes` passes when the answer is `42`. `checker_v2.py` is an independent, stricter checker; training
uses it as the reward, and logs the lenient grader at weight 0 as a reward-hacking alarm.

## Reproduce it

You need an NVIDIA GPU machine with Docker (the training images are `aarch64` builds for the DGX Spark),
Python 3.11 or later, and about 20 GB of disk for the task files.

1. Install the host dependencies and download the dataset:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install pandas numpy pyarrow huggingface_hub
   .venv/bin/hf download FineEnvs/SmolDataEnvs --repo-type dataset --local-dir data/SmolDataEnvs
   ```

2. Build the sandbox image and check that the sandbox works. The self-test uses mock models, so it
   needs no GPU and no network:

   ```bash
   .venv/bin/python smolenv_local.py build-docker-image
   .venv/bin/python eval_harness.py selftest --sandbox docker --out runs/selftest
   ```

3. Download the input files the tasks use. The command without `--yes` is a dry run that prints the
   size:

   ```bash
   .venv/bin/python scripts/stage_task_files.py --splits eval,test
   .venv/bin/python scripts/stage_task_files.py --splits eval,test,train --yes
   ```

4. Serve a model and run the 394-task exam:

   ```bash
   scripts/serve_vllm.sh Qwen/Qwen3.5-4B 8010 0.35
   .venv/bin/python eval_harness.py run --endpoint http://127.0.0.1:8010/v1 --model Qwen/Qwen3.5-4B \
     --slice eval_plus_test --mode agent --sandbox docker --out runs/base-4b-agent
   ```

5. Build the two training images. `smol-train:v2` builds on `smol-train:local`:

   ```bash
   docker build -f docker/Dockerfile.train -t smol-train:local docker/
   docker build -f docker/Dockerfile.train2 -t smol-train:v2 docker/
   ```

6. Pre-screen 1,000 candidate training tasks, keep the ones the model solves only sometimes, and train:

   ```bash
   .venv/bin/python scripts/make_prescreen_shards.py --profile agent_4b_lora --n 1000 --shards 1
   scripts/prescreen_agent.sh data/prescreen_agent_shard1.json runs/prescreen-4b
   .venv/bin/python scripts/build_pool.py runs/prescreen-4b data/grpo_pool_4b_agent.json --min-samples 4
   SCRIPT=train_grpo_agent.py scripts/train_in_container.sh --profile agent_4b_lora
   ```

   Merge the LoRA adapter with `scripts/merge_lora.py`, serve it with `scripts/serve_local_model.sh`, and
   run the exam from step 4 again.

`results/versions.txt` pins the exact versions used: TRL 1.14.0, PEFT 0.21.0, Transformers 5.15.0,
vLLM 0.27.1, flash-linear-attention 0.5.2, the FineEnvs commit, the SmolDataEnvs revision, and the
Qwen3.5-4B snapshot.

## Lessons from the DGX Spark

- **Match the trainer to vLLM.** Without `flash-linear-attention`, Transformers runs Qwen3.5's Gated
  DeltaNet layers on a slow PyTorch fallback whose log-probs differ from vLLM's. TRL's default
  importance sampling (`sequence_mask`) multiplies that gap across the sequence, so long completions
  got almost no weight. `Dockerfile.train2` installs it, and the plan sets `is_mode: token_truncate`.
- **Use `repetition_penalty: 1.0`.** vLLM applies the penalty when it samples, but the trainer's
  log-probs don't include it. With 1.05 (the FineEnvs default) the per-token gap was 0.06 nats; with
  1.0 it was 0.012. A healthy run keeps the gap well under 0.02.
- **Watch memory.** The GB10's unified memory doesn't OOM-kill cleanly. `train_in_container.sh` stops
  the run when available memory stays under 5 GB, and `scripts/cache_evict.py` keeps the page cache in
  check.

## Data and licenses

- Our code is Apache-2.0 (see `LICENSE`).
- `vendor/` holds code from [FineEnvs](https://github.com/adithya-s-k/FineEnvs) (Apache-2.0, see
  `vendor/LICENSE-FineEnvs.txt`) and the SmolDataEnvs grader (MIT). See `NOTICE`.
- The SmolDataEnvs task metadata is MIT. The input files come from Kaggle datasets that each have their
  own license, so this repository doesn't include them; `scripts/stage_task_files.py` downloads them
  from the dataset's source.
- Qwen3.5-4B and Qwen3.5-27B are Apache-2.0.

`data/slices.json` lists the task IDs of each exam slice, and `data/bucket_listing.json` lists the input
files each task needs (names and sizes only).
