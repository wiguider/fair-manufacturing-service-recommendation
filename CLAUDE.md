# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for ACM RecSys 2026 short paper. Builds a manufacturing service recommendation system with fairness constraints, ensuring equitable exposure for producers across company size (small/medium/large) and US geographic region (northeast/midwest/south/west).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download data (generates synthetic fallback if raw MSKG unavailable)
python data/download.py

# Run all experiments sequentially
python experiments/run_all.py

# Run individual experiment
python experiments/01_data_exploration.py
python experiments/02_baseline_comparison.py
python experiments/03_fair_reranking.py
python experiments/04_ablation_study.py

# Tests
pytest tests/ -v
pytest tests/test_metrics.py -v          # run single test file
pytest tests/test_metrics.py::TestNDCG -v # run single test class
```

## Architecture

**Data layer** (`src/data/`): Two dataset processors (MSKG knowledge graph, DataCo supply chain) that produce a unified `RecDataset` dataclass (defined in `loader.py`). `RecDataset` holds `interactions` (user_id, item_id, rating), `item_features`, and `protected_attrs` (item_id, size_group, geo_group). If raw MSKG data isn't present, `mskg_processor.py` generates a synthetic dataset that mimics its structure with realistic popularity bias toward large manufacturers.

**Models** (`src/models/`): All recommenders inherit from `BaseRecommender` (in `baselines.py`), which defines `fit()`, `predict()`, `recommend()`, and `recommend_all()`. Implementations:
- `baselines.py`: Random, Popularity, ContentBased (TF-IDF)
- `collaborative.py`: BPR and NeuMF (PyTorch, BPR loss / BCE loss)
- `graph_based.py`: LightGCN (PyTorch Geometric, `LGConv` layers)
- `fair_reranking.py`: Post-processing re-rankers — FA\*IR (binary), DetConstSort (multi-group), and the paper's proposed multi-attribute method that uses intersectional fairness deficit + relevance trade-off via `lambda_fair`

**Fairness** (`src/fairness/`): `groups.py` defines size/geo group constants and membership utilities. `metrics.py` computes DPR (demographic parity ratio), equity of exposure, expected exposure loss, and intersectional variants across `size_group × geo_group`.

**Evaluation** (`src/evaluation/`): `ranking_metrics.py` has standard IR metrics (precision@k, recall@k, NDCG@k, MRR, MAP). `statistical_tests.py` provides significance testing.

**Config**: Single YAML at `configs/default.yaml` loaded via `src/utils/config.py:load_config()`. Controls data paths, model hyperparameters, fair re-ranking parameters, and evaluation settings (top_k values, metrics list).

## Key Conventions

- Python 3.10+, uses `list[str]` and `dict[str, float]` type hints (not `List`/`Dict`)
- Experiments use `sys.path.insert(0, ...)` to add project root; imports are `from src.xxx import ...`
- All randomness seeded via `src/utils/seed.set_seed()` and `np.random.RandomState(seed)`
- Protected attributes are always a DataFrame with columns: `item_id`, `size_group`, `geo_group`
- Fairness metrics return lower-is-better values (except DPR where 1.0 = perfect parity)
- Results go to `results/` directory; raw data in `data/raw/` (gitignored)
- LaTeX paper source in `paper/`
