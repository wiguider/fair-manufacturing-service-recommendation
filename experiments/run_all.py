"""Run the complete experimental pipeline."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config


def main():
    config = load_config()
    set_seed(config["seed"])

    Path("results").mkdir(exist_ok=True)

    steps = [
        ("01 — Data Exploration", "experiments.01_data_exploration"),
        ("02 — Baseline Comparison", "experiments.02_baseline_comparison"),
        ("03 — Fair Re-ranking", "experiments.03_fair_reranking"),
        ("04 — Ablation Study", "experiments.04_ablation_study"),
    ]

    total_start = time.time()

    for step_name, module_name in steps:
        print(f"\n{'#' * 60}")
        print(f"# {step_name}")
        print(f"{'#' * 60}")

        step_start = time.time()
        try:
            # Import and run each step
            module = __import__(module_name.replace(".", "/").replace("/", "."), fromlist=["main"])
            module.main()
        except Exception as e:
            print(f"\n[ERROR] {step_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - step_start
        print(f"\n[{step_name}] completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 60}")
    print(f"# All experiments completed in {total_elapsed:.1f}s")
    print(f"# Results saved in results/")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
