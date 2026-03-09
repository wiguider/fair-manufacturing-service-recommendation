# Integration Plan: Strengthening the RecSys 2026 Submission

Based on analysis of RecSys 2024-2025 accepted papers, this plan addresses four key recommendations to make the submission more competitive.

---

## Current State Assessment

- **Both datasets fall back to synthetic data** — `data/raw/mskg/` and `data/raw/dataco/` are empty. The entire experimental evaluation runs on synthetic data, which significantly weakens the contribution.
- **No modern baselines** beyond LightGCN (2020). No LLM-based or recent GNN model.
- **Related work** does not cite or differentiate from "Intersectional Two-sided Fairness" (WWW '24) or "Mapping Stakeholder Needs to Multi-Sided Fairness" (RecSys '25).
- **Bug:** `experiments/03_fair_reranking.py` lines 109-110 initialize `per_user_ndcg_base` and `per_user_ndcg_rerank` but never populate them — statistical significance tests are never run.

---

## Phase 1: Dataset Acquisition and Real Data Integration (CRITICAL)

### 1.1 Acquire the DataCo Smart Supply Chain Dataset
- **Source:** UCI ML Repository (URL already configured in `data/download.py`)
- **Action:** Run `python data/download.py`, verify CSV extracted into `data/raw/dataco/`
- **Validation:** `src/data/supply_chain_processor.py` has `_process_real_dataco()` already implemented (lines 43-130). Test with actual CSV to confirm column name matching works.
- **Effort:** Low (1-2 hours)

### 1.2 Acquire MSKG Data
- **Source:** Contact Li & Starly (arXiv 2404.06571) for the MSKG knowledge graph
- **Expected files:** `manufacturers.csv`, `services.csv`, `certifications.csv`, `relationships.csv`
- **Code:** `src/data/mskg_processor.py` `load_mskg_from_json()` already handles multiple file format variations
- **Fallback:** If MSKG unavailable within 2 weeks, construct a dataset from Open Supply Hub (https://opensupplyhub.org/) with real manufacturer locations and capabilities
- **New file (if fallback needed):** `src/data/opensupplyhub_processor.py`
- **Effort:** Medium (depends on data availability — contact authors immediately)

### 1.3 Fallback: Third Dataset Option
- If MSKG is unobtainable, use Open Supply Hub or Amazon MFG product reviews
- New processor must produce a `RecDataset` with `item_id`, `size_group`, `geo_group`
- **Effort:** Medium (4-8 hours)

---

## Phase 2: Add Modern Baselines (HIGH)

### 2.1 SentenceBERT Recommender (LLM-based baseline)
- **New file:** `src/models/llm_recommender.py`
- **Approach:** Use pre-trained `all-MiniLM-L6-v2` to encode manufacturer service descriptions, build user profiles as mean embeddings of interacted items, score via cosine similarity
- **Interface:** Extend `BaseRecommender` from `src/models/baselines.py` — implement `fit()` and `predict()`
- **Dependency:** Add `sentence-transformers>=2.2.0` to `requirements.txt`
- **Config:** Add `llm_recommender` section to `configs/default.yaml`
- **Effort:** Medium (4-6 hours)

### 2.2 UltraGCN (Modern GNN baseline)
- **New file:** `src/models/gnn_advanced.py`
- **Why UltraGCN:** Approximates infinite-layer GCN without explicit message passing — architecturally very different from LightGCN, which strengthens the "model-agnostic re-ranking" argument
- **Interface:** Extend `BaseRecommender`, reuse bipartite graph construction from `LightGCNRecommender._build_edge_index()`
- **Fallback:** If too complex, use NeuMF (already implemented) paired with SentenceBERT as the two "additional" baselines
- **Effort:** Medium-High (6-10 hours)

### 2.3 Integrate into Experiment Pipeline
- **Modify:** `experiments/02_baseline_comparison.py` — add new models to `models` dict (line ~109)
- **Modify:** `experiments/03_fair_reranking.py` — add new models to `base_models` dict (line ~34)
- **Modify:** `experiments/04_ablation_study.py` — include new base models
- **Effort:** Low (1-2 hours)

---

## Phase 3: Related Work and Differentiation (HIGH)

### 3.1 Add New References
**Modify:** `paper/references.bib` — add:

1. **Ghosh et al. (WWW 2024)** — "Intersectional Two-sided Fairness in Recommendation"
   - *Their work:* Two-sided fairness with intersectional groups using sharpness-aware loss
   - *Our difference:* (a) provider-side only, (b) manufacturing-specific group definitions, (c) greedy composite deficit algorithm vs constrained optimization

2. **RecSys 2025 paper** — "Mapping Stakeholder Needs to Multi-Sided Fairness in Candidate Recommendation"
   - *Their work:* Framework mapping stakeholder needs to fairness constraints in hiring
   - *Our difference:* (a) concrete re-ranking algorithm, not a framework, (b) manufacturing domain, (c) intersectional fairness across size x geography

3. **MANI-Rank (AIES 2022)** — Multi-attribute intersectional fairness for consensus ranking
   - *Our difference:* (a) applied to recommendation not consensus ranking, (b) incorporates relevance-fairness trade-off, (c) domain-specific evaluation

### 3.2 Revise Paper Related Work Section
**Modify:** `paper/main.tex` Section 2 (lines 69-79) — add paragraph:

> **Intersectional and Multi-Sided Fairness.** Recent work has begun addressing fairness across multiple protected attributes simultaneously. Ghosh et al. [cite] studied intersectional two-sided fairness... MANI-Rank [cite] addressed multi-attribute fairness in consensus ranking... [RecSys '25 cite] mapped stakeholder needs to fairness in hiring. Our work differs in three key ways: (1) we focus specifically on provider-side exposure fairness, (2) we define domain-specific protected groups for manufacturing (supplier size and geographic region), and (3) we propose a greedy composite-deficit re-ranking algorithm that handles multiple attributes without exponential complexity.

**Effort:** Medium (3-4 hours for reading papers and writing)

---

## Phase 4: Emphasize the Manufacturing Domain (MEDIUM-HIGH)

### 4.1 Add Domain-Specific Metrics
- **New file:** `src/fairness/domain_metrics.py`
- **Metrics to implement:**
  - **Supply Chain Concentration (HHI):** Herfindahl-Hirschman Index across recommended manufacturers — lower = more diversified supply chain
  - **Certification Coverage:** Fraction of unique certifications (ISO9001, AS9100, etc.) in top-k — higher = more capability diversity
  - **Regional Resilience Score:** Geographic entropy of recommended suppliers — higher = better resilience against regional disruptions
- **Follow existing pattern** in `src/fairness/metrics.py`: each takes `ranking`, `protected_attrs`/`item_features`, `top_k`
- **Effort:** Medium (3-5 hours)

### 4.2 Domain Analysis Experiment
- **New file:** `experiments/05_domain_analysis.py`
- Compute HHI, certification coverage, and resilience for each method (with and without re-ranking)
- Generate visualization showing fair re-ranking improves supply chain health
- **Effort:** Medium (3-5 hours)

### 4.3 Update Paper Framing
**Modify:** `paper/main.tex`:
- **Introduction (lines 50-66):** Add contribution bullet: "We identify manufacturing-specific fairness dimensions (supplier size, geographic resilience) and show they align with supply chain best practices"
- **Add subsection:** "Domain-Specific Impact Analysis" in experiments
- **Conclusion (lines 204-211):** "To our knowledge, this is the first work to apply fair ranking to manufacturing service platforms"
- **Effort:** Medium (3-4 hours)

---

## Phase 5: Experimental Improvements (MEDIUM)

### 5.1 Fix Statistical Significance Bug
- **Modify:** `experiments/03_fair_reranking.py` lines 109-111
- `per_user_ndcg_base` and `per_user_ndcg_rerank` are initialized but never populated
- Fix: collect per-user NDCG values during evaluation loop, then call `wilcoxon_test()` (already imported)
- **Effort:** Low (1-2 hours)

### 5.2 Cross-Dataset Generalization
- **Modify:** `experiments/03_fair_reranking.py`
- Add experiment: tune `lambda_fair` on one dataset, evaluate on the other
- Shows the method generalizes across manufacturing contexts
- **Effort:** Low (2-3 hours)

### 5.3 Pareto Front Visualization
- **New file or add to:** `experiments/05_domain_analysis.py`
- Plot NDCG@10 vs intersectional DPR for all methods
- Our method should be on the Pareto frontier — strong visual for a short paper
- **Effort:** Low (2-3 hours)

### 5.4 Update Ablation
- **Modify:** `experiments/04_ablation_study.py` — add new base models to show re-ranking improvement is consistent
- **Modify:** `experiments/run_all.py` — add step 05
- **Effort:** Low (1 hour)

---

## Phase 6: Paper Updates (HIGH — done last)

### 6.1 Update Results Tables
- `paper/main.tex` Table 1 (lines 152-171): Add rows for SentenceBERT, UltraGCN + re-ranked variants
- Table 2 (lines 183-197): Update ablation with real numbers
- Add Table 3: Domain-specific metrics (HHI, certification coverage, resilience)

### 6.2 Abstract and Keywords
- Mention real datasets by name
- Add keywords: "supply chain fairness", "manufacturing recommendation"

### 6.3 Page Budget
- RecSys short papers: 4 pages + references
- Compress algorithm description into pseudocode float
- Merge related work paragraphs where possible

---

## Phase 7: Testing and Reproducibility (MEDIUM)

- **New file:** `tests/test_new_models.py` — unit tests for SentenceBERT and UltraGCN
- **Modify:** `tests/test_metrics.py` — add tests for domain metrics (HHI, coverage, resilience)
- **Modify:** `requirements.txt` — add `sentence-transformers>=2.2.0`
- **Modify:** `configs/default.yaml` — add new model hyperparameters
- Verify full pipeline: `python experiments/run_all.py`

---

## Timeline

| Week | Phases | Key Deliverables |
|------|--------|-----------------|
| 1 | 1.1, 1.2 | DataCo downloaded and validated; MSKG data request sent |
| 1-2 | 2.1, 2.2 | SentenceBERT and UltraGCN implemented and tested |
| 2 | 2.3, 5.1 | New models in experiment pipeline; stats bug fixed |
| 2-3 | 3.1, 3.2 | Related work updated with citations and differentiation |
| 3 | 4.1, 4.2 | Domain metrics implemented; domain analysis experiment |
| 3-4 | 5.2, 5.3, 5.4 | Cross-dataset experiment; Pareto visualization; updated ablation |
| 4 | 1.3 (if needed) | Fallback dataset if MSKG unavailable |
| 4-5 | 6 | Paper updated with real results, new tables, tightened text |
| 5 | 7 | Tests, reproducibility, final run |

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| MSKG data unavailable | Medium | Use DataCo as primary + Open Supply Hub as secondary |
| SentenceBERT too slow | Low | Use `all-MiniLM-L6-v2` (fast); pre-compute and cache embeddings |
| Page limit exceeded | Medium | Compress algorithm into pseudocode; move details to appendix |
| Weak results on real data | Medium | If re-ranking doesn't help, frame as domain-specific finding |
| UltraGCN too complex | Low-Medium | Fall back to NeuMF (already implemented) + SentenceBERT |

---

## Files Summary

### New files to create
| File | Description |
|------|-------------|
| `src/models/llm_recommender.py` | SentenceBERT-based recommender |
| `src/models/gnn_advanced.py` | UltraGCN recommender |
| `src/fairness/domain_metrics.py` | HHI, certification coverage, resilience |
| `experiments/05_domain_analysis.py` | Domain-specific evaluation experiment |
| `tests/test_new_models.py` | Tests for new models |

### Files to modify
| File | Changes |
|------|---------|
| `experiments/02_baseline_comparison.py` | Add new models to `models` dict |
| `experiments/03_fair_reranking.py` | Add new models; fix per-user stats bug (lines 109-111) |
| `experiments/04_ablation_study.py` | Add new base models |
| `experiments/run_all.py` | Add step 05 |
| `configs/default.yaml` | Add new model hyperparameters |
| `requirements.txt` | Add `sentence-transformers` |
| `paper/main.tex` | Related work, results tables, abstract, conclusion |
| `paper/references.bib` | Add WWW '24, RecSys '25, AIES '22 citations |
| `src/data/supply_chain_processor.py` | Validate with real DataCo CSV |
| `src/data/mskg_processor.py` | Validate with real MSKG data |
| `tests/test_metrics.py` | Add domain metric tests |
