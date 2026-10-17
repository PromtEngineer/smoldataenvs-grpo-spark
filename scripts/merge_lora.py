"""Merge a GRPO LoRA adapter into its base model for serving (run inside smol-train:local).
Loads the base exactly as TRL did (trl.trainer.utils.create_model_from_path -> the config's
architecture class), attaches the adapter, checks every adapter tensor attached, merges, saves bf16.
    python scripts/merge_lora.py ADAPTER_DIR OUT_DIR"""
import glob, os, shutil, sys
import torch
from peft import PeftModel
from safetensors import safe_open
from trl.trainer.utils import create_model_from_path
from huggingface_hub import snapshot_download

adapter, out = sys.argv[1], sys.argv[2]
import json
base = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
model = create_model_from_path(base, dtype="bfloat16", device_map="cpu")
ref_name, ref = next((n, p.detach().clone()) for n, p in model.named_parameters() if "language_model" in n and n.endswith("q_proj.weight"))
peft_model = PeftModel.from_pretrained(model, adapter)
n_file = len(list(safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt").keys()))
n_attached = sum(1 for n, _ in peft_model.named_parameters() if "lora_" in n)
print(f"adapter tensors in file: {n_file}, lora params attached: {n_attached}")
assert n_attached == n_file, "adapter did not fully attach (module names differ from training)"
merged = peft_model.merge_and_unload()
delta = (dict(merged.named_parameters())[ref_name].float() - ref.float()).abs().max().item()
print(f"max |delta| on {ref_name}: {delta:.3e}")
assert delta > 0, "merge changed nothing"
merged.save_pretrained(out, safe_serialization=True)
snap = base if os.path.isdir(base) else snapshot_download(base, local_files_only=True)  # base may be a merged dir (M1)
for f in ("preprocessor_config.json", "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
          "chat_template.jinja", "vocab.json", "merges.txt", "generation_config.json"):
    src = os.path.join(snap, f)
    if os.path.exists(src) and not os.path.exists(os.path.join(out, f)):
        shutil.copy(src, os.path.join(out, f))
print(f"merged -> {out}: {sorted(os.listdir(out))}")
