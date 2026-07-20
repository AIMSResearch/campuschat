from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str) -> None:
    print(f"\n=== Running {module} ===")
    subprocess.run([sys.executable, "-m", module], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    run_module("src.train_intent")
    run_module("src.build_retriever")
    run_module("src.evaluate")
    print("\nTraining, retrieval build, and evaluation completed.")
    print("Review artifacts/ and data/quarantine/ before deployment.")
