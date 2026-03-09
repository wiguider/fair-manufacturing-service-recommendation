# Fair Manufacturing Service Recommendation

Research project for ACM RecSys 2026 — Short Paper.

**Research question**: How to design a manufacturing service recommendation system that is accurate and guarantees fair exposure for producers of different sizes and geographic regions?

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

```bash
python data/download.py
```

## Run experiments

```bash
# Full pipeline
python experiments/run_all.py

# Individual experiments
python experiments/01_data_exploration.py
python experiments/02_baseline_comparison.py
python experiments/03_fair_reranking.py
python experiments/04_ablation_study.py
```

## Project structure

- `src/` — Source code (data processing, models, fairness, evaluation)
- `experiments/` — Executable experiment scripts
- `configs/` — YAML configuration files
- `paper/` — LaTeX paper source
- `data/` — Raw and processed datasets (raw data gitignored)
- `results/` — Experiment outputs

## References

- Boratto et al. (2022) — Fair performance-based user recommendation in eCoaching systems, UMUAI
- Li & Starly (2024) — Building A Knowledge Graph for Manufacturing Service Discovery
- Zehlike et al. (2017) — FA*IR: A Fair Top-k Ranking Algorithm
- Singh & Joachims (2018) — Fairness of Exposure in Rankings
