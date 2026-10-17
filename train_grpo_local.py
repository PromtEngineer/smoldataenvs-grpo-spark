"""GRPO on SmolDataEnvs on hardware you own (DGX Spark GB10, later an RTX 4090).

OUTLINE + WIRING. The reward path is complete and dry-runnable on a laptop; the TRL
part is the FineEnvs train_grpo.py configuration, parameterised by a profile in
train_plan.yaml. Nothing here has been run on a GPU yet.

Same reward as the eval harness, by construction: both call
smolenv_local.program_rollout(), which grades with the vendored dataset grader (v1)
and the strict checker (v2). The profile's `optimise` picks which one is the reward
(default strict: FineEnvs' published v0, trained on v1, learned to print
"<value> label Yes"); the other, plus "ran", command-filter and timeout rates, are
logged at weight 0 so reward hacking shows up in the curves.

    # laptop, no GPU, no TRL: prompts + local sandbox + rewards on 4 train tasks
    .venv/bin/python train_grpo_local.py --profile spark_2b_full --dry-run

    # Spark, 40-step throughput pilot (needs train files staged + TRL/vLLM env)
    python train_grpo_local.py --profile spark_2b_full --pilot

    # full run
    python train_grpo_local.py --profile spark_2b_full --init-from runs/sft-2b-program/final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import smolenv_local as env  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # FineEnvs: fragmentation OOM


def load_profile(name: str) -> dict:
    import yaml

    plan = yaml.safe_load((ROOT / "train_plan.yaml").read_text())
    prof = dict(plan["grpo"]["common"])
    prof.update(plan["grpo"]["profiles"][name])
    prof["_plan"] = plan
    return prof


# ── data ─────────────────────────────────────────────────────────────────────

def build_train_rows(prof: dict, n_tasks: int | None = None) -> list[dict]:
    from eval_harness import load_tasks

    rows = load_tasks("train", tiers=prof.get("difficulty", ""), limit=0)
    pool = prof.get("task_pool_file")
    if pool and (ROOT / pool).exists():  # pre-screened: tasks the base model solves 1..7 of 8 times
        keep = set(json.loads((ROOT / pool).read_text()))
        rows = [r for r in rows if r["task_id"] in keep]
    if prof.get("max_input_mb"):
        listing = json.loads((ROOT / "data/bucket_listing.json").read_text())
        size = {(k.split("::")[1], f["path"].split("/")[-1]): f["size"] for k, v in listing.items() for f in (v or [])}
        whole = {k.split("::")[1]: sum(f["size"] for f in (v or [])) for k, v in listing.items()}

        def need(r):  # empty `files` -> the whole prefix is staged
            return sum(size.get((r["bucket_prefix"], f), 0) for f in r["files"]) if r["files"] else whole.get(r["bucket_prefix"], 0)

        rows = [r for r in rows if need(r) <= prof["max_input_mb"] * 1e6]
    import random

    random.Random(prof.get("seed", 42)).shuffle(rows)
    return rows[: (n_tasks or prof["num_tasks"])]


def to_hf_dataset(rows: list[dict]):
    from datasets import Dataset

    keep = ("task_id", "answer", "reward_mode", "atol", "rtol", "bucket_prefix", "files", "difficulty_tier", "question")
    return Dataset.from_list([{**{k: r[k] for k in keep}, "prompt": env.program_messages(r)} for r in rows])


# ── reward: one sandbox run per completion, shared by all reward functions ────

class RewardEngine:
    """TRL passes each reward function (completions, **dataset_columns). All reward
    functions for a batch see the same completions, so execute once and cache."""

    def __init__(self, data_root: Path, work_dir: Path, workers: int, timeout_s: float, allow_missing: bool = False):
        self.cfg = env.SandboxConfig(timeout_s=timeout_s)
        self.allow_missing = allow_missing  # dry run only: canned completions do not read data
        self.data_root, self.work_dir = data_root, work_dir
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self._key, self._res = None, None
        self._lock = threading.Lock()
        self.stats = {"batches": 0, "exec_seconds": 0.0, "wall_seconds": 0.0}

    def _one(self, row: dict, completion) -> dict:
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        try:
            sb = env.LocalSandbox(row, self.data_root, self.work_dir / "sandboxes", self.cfg)
            if sb.missing_files and not self.allow_missing:
                return {"reward_v1": None, "reward_v2": None, "ran": 0.0, "error": "missing_data"}
            return env.program_rollout(sb, row, text)
        except Exception as e:  # the sandbox, not the policy: ungraded (None), never 0.0
            return {"reward_v1": None, "reward_v2": None, "ran": 0.0, "error": f"{type(e).__name__}: {e}"}

    def results(self, completions, **cols) -> list[dict]:
        with self._lock:
            if self._key is not completions:
                t0 = time.time()
                rows = [{k: cols[k][i] for k in cols if isinstance(cols[k], list)} for i in range(len(completions))]
                for r in rows:
                    r.setdefault("task_id", "task")
                self._res = list(self.pool.map(self._one, rows, completions))
                self._key = completions
                self.stats["batches"] += 1
                self.stats["wall_seconds"] += time.time() - t0
                self.stats["exec_seconds"] += sum(x.get("exec_seconds", 0.0) or 0.0 for x in self._res)
            return self._res


def make_reward_funcs(engine: RewardEngine):
    def reward_correct(completions, **cols):
        """Optimised. 1.0 iff the dataset grader (v1) accepts the last printed line."""
        return [r["reward_v1"] for r in engine.results(completions, **cols)]

    def reward_correct_strict(completions, **cols):
        """Logged at weight 0: checker_v2 on the same prediction. v1 rising while this
        stays flat is the reward-hacking alarm."""
        return [r["reward_v2"] for r in engine.results(completions, **cols)]

    def reward_ran(completions, **cols):
        """Logged at weight 0 (FineEnvs: as a 0.1 shaping term it was exploited)."""
        return [r.get("ran", 0.0) for r in engine.results(completions, **cols)]

    def reward_command_filtered(completions, **cols):
        """Logged at weight 0: share of answers rejected as command-shaped."""
        return [1.0 if r.get("filtered") == "command_shaped" else 0.0 for r in engine.results(completions, **cols)]

    def reward_timed_out(completions, **cols):
        return [1.0 if r.get("timed_out") else 0.0 for r in engine.results(completions, **cols)]

    return [reward_correct, reward_correct_strict, reward_ran, reward_command_filtered, reward_timed_out]


# ── callbacks (throughput + held-out eval under both checkers) ────────────────

def make_callbacks(prof: dict, engine: RewardEngine, run_dir: Path):
    from transformers import TrainerCallback

    class Throughput(TrainerCallback):
        """tokens/s = completion tokens generated in the step / step wall time."""

        def on_step_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            if logs is None or not hasattr(self, "t0"):
                return
            dt = time.time() - self.t0
            n_compl = prof["per_device_train_batch_size"] * prof["gradient_accumulation_steps"]
            mean_len = logs.get("completions/mean_length")
            if mean_len:
                logs["perf/gen_tokens_per_step"] = mean_len * n_compl
                logs["perf/tokens_per_s_end_to_end"] = mean_len * n_compl / max(dt, 1e-6)
            logs["perf/step_seconds"] = dt
            logs["perf/sandbox_wall_s_total"] = engine.stats["wall_seconds"]
            logs["perf/sandbox_exec_s_total"] = engine.stats["exec_seconds"]

    class JsonlLog(TrainerCallback):
        """Every logged step appended to run_dir/train_log.jsonl (no tracker service needed)."""

        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                with open(run_dir / "train_log.jsonl", "a") as fh:
                    fh.write(json.dumps({"step": state.global_step, "t": round(time.time(), 1), **logs}, default=str) + "\n")

    class HeldoutEval(TrainerCallback):
        """Every `eval_every` steps: greedy pass@1 on the eval split (144) with the
        trainer's colocated vLLM, graded by v1 AND v2, split by seen/unseen dataset.
        The test split (250) is NOT touched during training."""

        def __init__(self, trainer_ref):
            self.trainer_ref = trainer_ref

        def on_step_end(self, args, state, control, **kw):
            every = prof.get("eval_every", 100)
            if state.global_step == 0 or state.global_step % every:
                return
            from eval_harness import load_tasks, summarize
            from vllm import SamplingParams

            trainer = self.trainer_ref[0]
            llm = getattr(trainer, "llm", None) or getattr(getattr(trainer, "vllm_generation", None), "llm", None)
            if llm is None:  # TRL moved the colocated engine: evaluate afterwards with eval_harness.py instead
                print(f"heldout eval skipped at step {state.global_step}: no colocated vLLM handle on the trainer")
                return
            rows = load_tasks(prof.get("eval_split", "eval"))
            tok = trainer.processing_class
            prompts = [tok.apply_chat_template(env.program_messages(r), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for r in rows]
            outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=prof["max_completion_length"]))
            recs = []
            for r, o in zip(rows, outs):
                res = engine._one(r, o.outputs[0].text)
                recs.append({"task_id": r["task_id"], "tier": r["difficulty_tier"], "level": r["difficulty_level"],
                             "reward_mode": r["reward_mode"], "seen_dataset_in_train": r["seen_dataset_in_train"],
                             "split": r["split"], "model": "policy", "mode": "program",
                             "completion_tokens": len(o.outputs[0].token_ids), **res})
            s = summarize(recs)
            (run_dir / f"heldout_step{state.global_step}.json").write_text(json.dumps(s, indent=1))
            logs = {"heldout/pass@1_v1": s["overall"]["pass@1_v1"], "heldout/pass@1_v2": s["overall"]["pass@1_v2"],
                    "heldout/v1_minus_v2": s["overall"]["v1_minus_v2"]}
            for k, b in s["by_dataset_seen_in_train"].items():
                logs[f"heldout/seen_{k}_v1"] = b["pass@1_v1"]
                logs[f"heldout/seen_{k}_v2"] = b["pass@1_v2"]
            for k, b in s["by_tier"].items():
                logs[f"heldout/{k}_v1"] = b["pass@1_v1"]
            trainer.log(logs)

    return Throughput, HeldoutEval, JsonlLog


def write_meta(run_dir: Path, prof: dict, n_tasks: int, mode: str, init_from=None) -> None:
    """run_dir/meta.json: what the live dashboard needs to show progress and an ETA."""
    keep = ("model", "lora", "optimise", "max_steps", "num_generations", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "temperature", "max_completion_length", "eval_every",
            "save_steps", "task_pool_file", "max_tool_calling_iterations")
    meta = {k: prof.get(k) for k in keep if k in prof}
    meta.update(mode=mode, n_tasks=n_tasks, init_from=init_from, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="a key under grpo.profiles in train_plan.yaml")
    ap.add_argument("--pilot", action="store_true", help="30-50 step throughput pilot (pilot: block in the plan)")
    ap.add_argument("--dry-run", action="store_true", help="no GPU, no TRL: prompts + sandbox + rewards")
    ap.add_argument("--init-from", default=None, help="SFT checkpoint to start from (default: the profile's model)")
    ap.add_argument("--data-root", default=str(ROOT / "data/task_files"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a profile value (YAML-parsed), e.g. --set max_steps=2; applied last")
    a = ap.parse_args()

    prof = load_profile(a.profile)
    if a.pilot:
        prof.update(prof["_plan"]["pilot"]["overrides"])
    import yaml

    for kv in a.set:
        k, v = kv.split("=", 1)
        prof[k] = yaml.safe_load(v)
    if a.set:
        print("overrides:", {kv.split("=", 1)[0]: prof[kv.split("=", 1)[0]] for kv in a.set})
    run_dir = Path(a.run_dir or ROOT / "runs" / f"grpo-{a.profile}{'-pilot' if a.pilot else ''}")
    run_dir.mkdir(parents=True, exist_ok=True)
    engine = RewardEngine(Path(a.data_root), run_dir, prof.get("sandbox_workers", 8), prof.get("exec_timeout_s", 90),
                          allow_missing=a.dry_run)
    funcs = make_reward_funcs(engine)

    if a.dry_run:
        rows = build_train_rows(prof, n_tasks=4)
        cols = {k: [r[k] for r in rows] for k in ("task_id", "answer", "reward_mode", "atol", "rtol", "bucket_prefix", "files")}
        completions = [
            [{"role": "assistant", "content": f"```python\nprint({rows[0]['answer']!r})\n```"}],        # correct
            [{"role": "assistant", "content": "```python\n1/0\n```"}],                                  # crash
            [{"role": "assistant", "content": f"```python\nprint('echo -n \"{rows[2]['answer']}\" > /workdir/answer.txt')\n```"}],  # command-shaped
            [{"role": "assistant", "content": f"```python\nprint('The answer is {rows[3]['answer']}')\n```"}],  # v1-lenient format
        ]
        print("prompt:\n" + env.program_messages(rows[0])[-1]["content"][:500] + "\n" + "-" * 60)
        out = {f.__name__: f(completions, **cols) for f in funcs}
        for i, r in enumerate(rows):
            print(f"  completion {i} gold={r['answer']!r:24} " + " ".join(f"{k}={v[i]}" for k, v in out.items()))
        assert out["reward_correct"][0] == 1.0 and out["reward_correct"][1] == 0.0 and out["reward_correct"][2] == 0.0
        print(f"\ndry run ok ({env.detect_backend()} sandbox). completion 3 shows what v1 forgives and v2 does not.")
        (run_dir / "dry_run.json").write_text(json.dumps(out, indent=1, default=str))
        return

    # ── the TRL part (GPU) ─────────────────────────────────────────────────
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    rows = build_train_rows(prof)
    ds = to_hf_dataset(rows)
    print(f"{len(ds)} training tasks, {prof['num_generations']} generations each, profile={a.profile}")
    write_meta(run_dir, {**prof, "optimise": prof.get("optimise", "strict")}, len(ds), "program", a.init_from)
    cfg = GRPOConfig(
        output_dir=str(run_dir),
        use_vllm=True,
        vllm_mode=prof.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=prof["vllm_gpu_memory_utilization"],
        # Match vLLM's precision (TRL otherwise loads the policy in float32) and correct the remaining
        # trainer/vLLM mismatch per token: TRL's default "sequence_mask" multiplies per-token log-prob gaps
        # across the whole sequence, which down-weighted long completions (corr(length, IS ratio) = -0.77).
        model_init_kwargs={"dtype": prof.get("model_dtype", "bfloat16")},
        vllm_importance_sampling_mode=prof.get("is_mode", "token_truncate"),
        vllm_importance_sampling_clip_max=prof.get("is_clip_max", 2.0),
        # Qwen3.5 defaults to 262,144 tokens of context; prompts here are ~300-900 tokens + 1,024 generated
        vllm_max_model_length=prof.get("vllm_max_model_length", 4096),
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=[0.0, 1.0, 0.0, 0.0, 0.0] if prof.get("optimise", "strict") == "strict" else [1.0, 0.0, 0.0, 0.0, 0.0],
        num_generations=prof["num_generations"],
        max_completion_length=prof["max_completion_length"],
        mask_truncated_completions=True,
        max_steps=prof["max_steps"],
        learning_rate=prof["learning_rate"],
        temperature=prof["temperature"],
        top_p=prof["top_p"],
        repetition_penalty=prof["repetition_penalty"],
        per_device_train_batch_size=prof["per_device_train_batch_size"],
        gradient_accumulation_steps=prof["gradient_accumulation_steps"],
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=prof["save_steps"],
        log_completions=True,
        report_to=prof.get("report_to", "none"),
        run_name=run_dir.name,
        seed=prof.get("seed", 42),
    )
    lora = None
    if prof.get("lora"):
        L = prof["lora"]
        lora = LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                          target_modules="all-linear", exclude_modules="visual.*")
    trainer_ref: list = []
    Throughput, HeldoutEval, JsonlLog = make_callbacks(prof, engine, run_dir)
    trainer = GRPOTrainer(model=a.init_from or prof["model"], reward_funcs=funcs, train_dataset=ds, args=cfg,
                          peft_config=lora, callbacks=[Throughput(), JsonlLog(), HeldoutEval(trainer_ref)])
    trainer_ref.append(trainer)
    trainer.train()
    trainer.save_model(str(run_dir / "final"))
    print(f"saved -> {run_dir / 'final'}")


if __name__ == "__main__":
    main()
