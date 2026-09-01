"""按顺序执行所有章节的快速配置；任一失败时返回非零状态码。"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CASES = [
    ("01_tensor_autograd.py", []),
    ("02_linear_regression.py", ["--epochs", "15"]),
    ("03_mlp_classification.py", ["--steps", "80"]),
    ("04_optimization_regularization.py", ["--steps", "40"]),
    ("05_embeddings.py", []),
    ("06_rnn_language_model.py", ["--steps", "15"]),
    ("07_self_attention.py", []),
    ("08_transformer_block.py", []),
    ("09_tokenizer_dataset.py", []),
    ("10_mini_gpt.py", ["--steps", "10", "--new-tokens", "5"]),
    ("11_lora_finetune.py", ["--steps", "50"]),
    ("12_generation_evaluation.py", ["--steps", "10"]),
    ("13_training_engineering.py", ["--updates", "2"]),
]


def main() -> None:
    failures: list[str] = []
    started = time.perf_counter()
    for script, arguments in CASES:
        print(f"\n{'=' * 18} {script} {'=' * 18}", flush=True)
        command = [sys.executable, str(ROOT / "src" / script), *arguments]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            failures.append(script)
    elapsed = time.perf_counter() - started
    print(f"\n完成 {len(CASES)} 个案例，用时 {elapsed:.1f}s")
    if failures:
        raise SystemExit("失败案例: " + ", ".join(failures))


if __name__ == "__main__":
    main()

