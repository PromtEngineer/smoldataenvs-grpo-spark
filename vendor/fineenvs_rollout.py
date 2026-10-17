# Vendored excerpt of adithya-s-k/FineEnvs 04-smoldataenvs/scripts/rollout.py
# (commit 08a5622ff3164bb72cd57d6360c7edcce6247bf8, Apache-2.0, see LICENSE-FineEnvs.txt).
#
# Only the pure, local functions are copied, verbatim: the prompt, code extraction,
# the command-shaped-answer filter, last-line extraction, and the grade() wrapper.
# SandboxRunner (Hugging Face Sandboxes, network + HF token) is deliberately NOT
# vendored; eval_harness.py replaces it with a local sandboxed subprocess.
# The grader itself is vendor/smoldataenvs_grader.py, a byte-identical copy of
# FineEnvs/SmolDataEnvs grader.py (sha256 bbcbf4e5...ac84a, MIT).
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

SYSTEM = (
    "You are a data analyst. You answer questions about CSV files by writing a short "
    "Python program and reading what it prints."
)

PROMPT = """{question}

The files are in /home/user/input and your program runs in that directory:
{files}

Write one Python program in a ```python block, then stop.

- Look at the data if you need to, then compute the answer.
- The LAST thing the program prints must be the answer on its own: a bare number
  (no commas or units), a short label, yes/no, or a comma-separated list.
- Keep it under 40 lines. pandas, numpy, scipy, sklearn and statsmodels are installed."""


def build_prompt(row: dict) -> list[dict]:
    """The one prompt shape, so training and eval cannot drift apart."""
    files = "\n".join(f"- {f}" for f in row["files"])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(question=row["question"], files=files)},
    ]


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(completion: str | list[dict]) -> str:
    """Last fenced block wins: models often think in one block and answer in the next."""
    if isinstance(completion, list):  # conversational completions
        completion = completion[-1]["content"] if completion else ""
    blocks = CODE_RE.findall(completion or "")
    if blocks:
        return blocks[-1].strip()
    return (completion or "").strip()


_COMMAND_SHAPED = re.compile(
    r"(^|\s)(echo|printf|cat|python3?|bash|sh|tee|awk|sed)\b|[>|]{1,2}\s*\S+|\$\(|`",
)


def looks_like_a_command(answer: str) -> bool:
    """True when the 'answer' is really the line that would produce it."""
    return bool(answer) and bool(_COMMAND_SHAPED.search(answer.strip()))


def last_line(stdout: str) -> str:
    """The answer is the last thing printed. Anything above it is the model thinking."""
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


_grader = None


def grader():
    """The dataset's own grader, loaded from the vendored copy (no network)."""
    global _grader
    if _grader is None:
        path = Path(__file__).with_name("smoldataenvs_grader.py")
        spec = importlib.util.spec_from_file_location("smoldataenvs_grader", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["smoldataenvs_grader"] = mod  # its dataclasses need this
        spec.loader.exec_module(mod)
        _grader = mod
    return _grader


def grade(row: dict, prediction: str) -> float:
    if not prediction:
        return 0.0
    r = grader().grade(
        row["answer"],
        prediction,
        reward_mode=row["reward_mode"],
        abs_tol=row["atol"],
        rel_tol=row["rtol"],
    )
    return float(r.reward)


def grade_with_method(row: dict, prediction: str) -> tuple[float, str]:
    """Same as grade(), but also returns which rung of the ladder matched."""
    if not prediction:
        return 0.0, "empty"
    r = grader().grade(row["answer"], prediction, reward_mode=row["reward_mode"],
                       abs_tol=row["atol"], rel_tol=row["rtol"])
    return float(r.reward), r.method
