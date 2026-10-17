"""SFT warm start for SmolDataEnvs (arm A2: program format; arm A3: FineEnvs agent format).

OUTLINE. Hyperparameters come from the `sft:` block of train_plan.yaml. The program
format trains the exact completion shape GRPO samples and the eval harness grades:
build_prompt(row) -> one ```python block whose last printed line is the bare answer.

    .venv/bin/python train_sft_local.py --format program --dry-run      # no GPU: data + length checks
    python train_sft_local.py --format program --model Qwen/Qwen3.5-2B   # Spark / 4090
    python train_sft_local.py --format agent   --model Qwen/Qwen3.5-2B   # FineEnvs train_sft.py equivalent
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_rows(fmt: str, sft_cfg: dict, include_rft: bool) -> list[dict]:
    if fmt == "agent":
        import pandas as pd

        df = pd.read_parquet(ROOT / "data/SmolDataEnvs-sft/data/train-00000-of-00001.parquet")
        return [{"messages": [json.loads(m) for m in r.messages], "tools": list(r.tools)} for r in df.itertuples()]
    rows = [json.loads(l) for l in (ROOT / sft_cfg["data"]).read_text().splitlines() if l.strip()]
    rft = ROOT / sft_cfg["data_optional_rft"]
    if include_rft and rft.exists():
        rows += [json.loads(l) for l in rft.read_text().splitlines() if l.strip()]
    return [{"messages": r["messages"]} for r in rows]


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=("program", "agent"), default="program")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--no-rft", action="store_true", help="program format: skip the teacher RFT file")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    plan = yaml.safe_load((ROOT / "train_plan.yaml").read_text())
    cfg = plan["sft"]["program_format" if a.format == "program" else "agent_format"]
    rows = load_rows(a.format, cfg, not a.no_rft)
    chars = sorted(sum(len(json.dumps(m)) for m in r["messages"]) for r in rows)
    print(f"{a.format}: {len(rows)} examples; chars p50={chars[len(chars)//2]} p99={chars[int(len(chars)*0.99)]} "
          f"(~{chars[int(len(chars)*0.99)]//3.5:.0f} tokens p99 vs max_length {cfg['max_length']})")
    if a.format == "program":
        bad = [r for r in rows if not r["messages"][-1]["content"].startswith("```python")]
        assert not bad, f"{len(bad)} targets are not a single python block"
    if a.dry_run:
        print("dry run ok")
        return

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list(rows).train_test_split(test_size=cfg["eval_holdout"], seed=42)
    out = a.out or str(ROOT / "runs" / f"sft-{a.format}-{a.model.split('/')[-1]}")
    L = cfg["lora"]
    trainer = SFTTrainer(
        model=a.model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                               target_modules=L["target_modules"], exclude_modules=L["exclude_modules"]),
        args=SFTConfig(
            output_dir=out,
            chat_template_kwargs={"enable_thinking": False},
            num_train_epochs=cfg["epochs"],
            max_length=cfg["max_length"],
            per_device_train_batch_size=cfg["per_device_batch"],
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["learning_rate"],
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=100,
            # program format: loss on the assistant program only (VERIFY flag name in the installed TRL)
            assistant_only_loss=a.format == "program",
            report_to="trackio",
            run_name=Path(out).name,
        ),
    )
    trainer.train()
    trainer.save_model(out + "/final")
    print(f"saved -> {out}/final  (merge the LoRA before serving with vLLM, or pass --enable-lora)")


if __name__ == "__main__":
    main()
