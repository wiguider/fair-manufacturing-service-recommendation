"""Step 3: Fair re-ranking experiments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.mskg_processor import process_mskg
from src.data.supply_chain_processor import process_dataco
from src.data.loader import train_val_test_split
from src.models.baselines import PopularityRecommender, ContentBasedRecommender
from src.models.collaborative import BPRRecommender
from src.models.graph_based import LightGCNRecommender
from src.models.llm_recommender import SentenceBERTRecommender
from src.models.gnn_advanced import UltraGCNRecommender
from src.models.fair_reranking import fair_rerank, multi_attribute_fair_rerank
from src.evaluation.ranking_metrics import evaluate_ranking, ndcg_at_k
from src.fairness.metrics import compute_all_fairness_metrics
from src.evaluation.statistical_tests import wilcoxon_test, bootstrap_ci


def run_fair_reranking(dataset, config: dict, output_prefix: str):
    """Run fair re-ranking on top of base recommenders."""
    seed = config["seed"]
    train_df, val_df, test_df = train_val_test_split(dataset, seed=seed)

    print(f"\nDataset: {dataset.name}")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Base recommenders to apply re-ranking on
    base_models = {
        "BPR": BPRRecommender(
            embedding_dim=config["models"]["bpr"]["embedding_dim"],
            lr=config["models"]["bpr"]["learning_rate"],
            num_epochs=config["models"]["bpr"]["num_epochs"],
            patience=config["models"]["early_stopping_patience"],
            seed=seed,
        ),
        "LightGCN": LightGCNRecommender(
            embedding_dim=config["models"]["lightgcn"]["embedding_dim"],
            num_layers=config["models"]["lightgcn"]["num_layers"],
            lr=config["models"]["lightgcn"]["learning_rate"],
            num_epochs=config["models"]["lightgcn"]["num_epochs"],
            patience=config["models"]["early_stopping_patience"],
            seed=seed,
        ),
        "SentenceBERT": SentenceBERTRecommender(
            model_name=config["models"]["llm_recommender"]["model_name"],
            batch_size=config["models"]["llm_recommender"]["batch_size"],
            normalize_embeddings=config["models"]["llm_recommender"]["normalize_embeddings"],
        ),
        "UltraGCN": UltraGCNRecommender(
            embedding_dim=config["models"]["ultragcn"]["embedding_dim"],
            ii_topk=config["models"]["ultragcn"]["ii_topk"],
            lambda_constraint=config["models"]["ultragcn"]["lambda_constraint"],
            w1=config["models"]["ultragcn"]["w1"],
            w2=config["models"]["ultragcn"]["w2"],
            w3=config["models"]["ultragcn"]["w3"],
            w4=config["models"]["ultragcn"]["w4"],
            neg_sample_ratio=config["models"]["ultragcn"]["neg_sample_ratio"],
            constraint_neg_ratio=config["models"]["ultragcn"]["constraint_neg_ratio"],
            lr=config["models"]["ultragcn"]["learning_rate"],
            weight_decay=config["models"]["ultragcn"]["weight_decay"],
            batch_size=config["models"]["ultragcn"]["batch_size"],
            num_epochs=config["models"]["ultragcn"]["num_epochs"],
            patience=config["models"]["ultragcn"]["early_stopping_patience"],
            seed=seed,
        ),
    }

    reranking_methods = ["fair", "detconstsort", "ours"]
    ks = config["evaluation"]["top_k"]
    max_k = max(ks)
    lambda_fair = config["fair_reranking"]["lambda_fair"]

    all_items = dataset.interactions["item_id"].unique()
    test_users = test_df["user_id"].unique()
    relevant_sets = {}
    for uid, group in test_df.groupby("user_id"):
        relevant_sets[uid] = set(group["item_id"].values)

    train_items_per_user = {}
    for uid, group in train_df.groupby("user_id"):
        train_items_per_user[uid] = set(group["item_id"].values)

    all_results = []

    for base_name, base_model in base_models.items():
        print(f"\n--- Base model: {base_name} ---")
        print(f"  Training {base_name}...")
        base_model.fit(train_df, dataset.item_features)

        # Generate base scores for each user
        user_scores = {}
        user_candidates = {}
        for uid in test_users:
            train_seen = train_items_per_user.get(uid, set())
            candidates = np.array([i for i in all_items if i not in train_seen])
            if len(candidates) == 0:
                continue
            scores = base_model.predict(uid, candidates)
            user_scores[uid] = scores
            user_candidates[uid] = candidates

        # Base (no re-ranking)
        base_rankings = {}
        for uid in user_scores:
            order = np.argsort(user_scores[uid])[::-1][:max_k]
            base_rankings[uid] = user_candidates[uid][order]

        base_ranking_metrics = evaluate_ranking(base_rankings, relevant_sets, ks)
        base_fair = {}
        for uid, ranking in base_rankings.items():
            uf = compute_all_fairness_metrics(ranking, dataset.protected_attrs, top_k=max_k)
            for m, v in uf.items():
                base_fair.setdefault(m, []).append(v)
        base_fair = {m: np.mean(v) for m, v in base_fair.items()}

        row = {"model": f"{base_name} (no rerank)"}
        row.update(base_ranking_metrics)
        row.update(base_fair)
        all_results.append(row)

        # Apply each re-ranking method
        sig_k = ks[0]  # use smallest k for significance test (e.g. NDCG@5)
        for method in reranking_methods:
            print(f"  Applying {method} re-ranking...")
            method_rankings = {}
            per_user_ndcg_base = []
            per_user_ndcg_rerank = []

            for uid in user_scores:
                reranked = fair_rerank(
                    scores=user_scores[uid],
                    item_ids=user_candidates[uid],
                    protected_attrs=dataset.protected_attrs,
                    method=method,
                    top_k=max_k,
                    lambda_fair=lambda_fair,
                    alpha=config["fair_reranking"]["fair_alpha"],
                )
                method_rankings[uid] = reranked

                rel = relevant_sets.get(uid, set())
                if rel:
                    per_user_ndcg_base.append(ndcg_at_k(base_rankings[uid], rel, sig_k))
                    per_user_ndcg_rerank.append(ndcg_at_k(reranked, rel, sig_k))

            # Evaluate
            method_ranking_metrics = evaluate_ranking(method_rankings, relevant_sets, ks)
            method_fair = {}
            for uid, ranking in method_rankings.items():
                uf = compute_all_fairness_metrics(ranking, dataset.protected_attrs, top_k=max_k)
                for m, v in uf.items():
                    method_fair.setdefault(m, []).append(v)
            method_fair_avg = {m: np.mean(v) for m, v in method_fair.items()}

            row = {"model": f"{base_name} + {method}"}
            row.update(method_ranking_metrics)
            row.update(method_fair_avg)
            all_results.append(row)

            # Print summary
            for k in ks:
                print(f"    NDCG@{k}: {method_ranking_metrics.get(f'ndcg@{k}', 0):.4f}")
            print(f"    Size DPR: {method_fair_avg.get('size_dpr', 0):.4f}  "
                  f"Geo DPR: {method_fair_avg.get('geo_dpr', 0):.4f}  "
                  f"Inter DPR: {method_fair_avg.get('inter_dpr', 0):.4f}")

            # Statistical significance vs base ranking
            if per_user_ndcg_base and per_user_ndcg_rerank:
                sig_result = wilcoxon_test(
                    np.array(per_user_ndcg_rerank),
                    np.array(per_user_ndcg_base),
                )
                print(f"    Wilcoxon NDCG@{sig_k}: stat={sig_result['statistic']:.4f}, "
                      f"p={sig_result['p_value']:.4f}, significant={sig_result['significant']}")

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = Path("results") / f"{output_prefix}_fair_reranking.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df


def _get_user_scores(
    model,
    train_df: pd.DataFrame,
    test_users,
    all_items,
) -> tuple[dict, dict]:
    """Return per-user candidate arrays and their predicted scores.

    Items already seen by the user in ``train_df`` are excluded from the
    candidate set, mirroring the exclusion logic used in ``run_fair_reranking``.
    """
    train_items_per_user: dict[int, set] = {}
    for uid, group in train_df.groupby("user_id"):
        train_items_per_user[uid] = set(group["item_id"].values)

    user_scores: dict[int, np.ndarray] = {}
    user_candidates: dict[int, np.ndarray] = {}
    for uid in test_users:
        train_seen = train_items_per_user.get(uid, set())
        candidates = np.array([i for i in all_items if i not in train_seen])
        if len(candidates) == 0:
            continue
        scores = model.predict(uid, candidates)
        user_scores[uid] = scores
        user_candidates[uid] = candidates

    return user_scores, user_candidates


def _tune_lambda_on_val(
    model,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    protected_attrs: pd.DataFrame,
    all_items,
    lambda_grid: list[float],
    max_k: int,
) -> float:
    """Select the ``lambda_fair`` value that maximises a fairness-utility balance on the validation set.

    The selection criterion is the harmonic mean of normalised NDCG@max_k and
    mean inter-group DPR.  Normalising NDCG keeps it commensurable with the
    [0, 1]-bounded DPR without requiring manual scaling constants.

    Returns the best lambda from ``lambda_grid``.
    """
    val_users = val_df["user_id"].unique()
    val_relevant: dict[int, set] = {}
    for uid, group in val_df.groupby("user_id"):
        val_relevant[uid] = set(group["item_id"].values)

    user_scores, user_candidates = _get_user_scores(model, train_df, val_users, all_items)

    best_lambda = lambda_grid[0]
    best_score = -float("inf")
    ndcg_per_lambda: list[float] = []
    dpr_per_lambda: list[float] = []

    for lam in lambda_grid:
        rankings: dict[int, np.ndarray] = {}
        for uid in user_scores:
            rankings[uid] = multi_attribute_fair_rerank(
                user_scores[uid],
                user_candidates[uid],
                protected_attrs,
                top_k=max_k,
                lambda_fair=lam,
            )

        ranking_metrics = evaluate_ranking(rankings, val_relevant, [max_k])
        ndcg = ranking_metrics.get(f"ndcg@{max_k}", 0.0)

        inter_dprs: list[float] = []
        for uid, ranking in rankings.items():
            fm = compute_all_fairness_metrics(ranking, protected_attrs, top_k=max_k)
            inter_dprs.append(fm.get("inter_dpr", 0.0))
        avg_dpr = float(np.mean(inter_dprs)) if inter_dprs else 0.0

        ndcg_per_lambda.append(ndcg)
        dpr_per_lambda.append(avg_dpr)

    # Normalise NDCG to [0, 1] across the grid so it is commensurable with DPR
    ndcg_arr = np.array(ndcg_per_lambda)
    ndcg_range = ndcg_arr.max() - ndcg_arr.min()
    norm_ndcg = (ndcg_arr - ndcg_arr.min()) / ndcg_range if ndcg_range > 0 else np.ones_like(ndcg_arr)

    for idx, lam in enumerate(lambda_grid):
        n = norm_ndcg[idx]
        d = dpr_per_lambda[idx]
        h = 2 * n * d / (n + d) if (n + d) > 0 else 0.0
        print(
            f"    lambda={lam:.2f}  val NDCG@{max_k}={ndcg_per_lambda[idx]:.4f}  "
            f"inter_dpr={d:.4f}  h-mean={h:.4f}"
        )
        if h > best_score:
            best_score = h
            best_lambda = lam

    print(f"  Best lambda_fair on validation: {best_lambda} (h-mean={best_score:.4f})")
    return best_lambda


def run_cross_dataset_experiment(config: dict) -> None:
    """Cross-dataset generalisation experiment (Section 5.2).

    Procedure
    ---------
    For each ordered pair (source, target) of {MSKG, DataCo}:

    1. Train BPR on the *source* dataset.
    2. Tune ``lambda_fair`` on the *source* validation split by maximising
       the harmonic mean of normalised NDCG@max_k and mean inter-group DPR.
    3. Train a fresh BPR on the *target* dataset (scoring is always
       in-domain; only the fairness hyper-parameter is transferred).
    4. Apply our multi-attribute re-ranker to the target test split under
       three conditions:

       - **Tuned lambda**: the value selected in step 2.
       - **Default lambda**: the value from ``configs/default.yaml``.
       - **No re-ranking**: plain score-sorted baseline.

    This isolates the transferability of the lambda trade-off from the
    transferability of collaborative-filtering embeddings, which are
    inherently dataset-specific.

    Results are saved to ``results/cross_dataset_generalization.csv``.
    """
    seed = config["seed"]
    ks = config["evaluation"]["top_k"]
    max_k = max(ks)
    lambda_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    default_lambda = config["fair_reranking"]["lambda_fair"]

    print("\n" + "=" * 60)
    print("Cross-Dataset Generalisation Experiment")
    print("=" * 60)

    mskg = process_mskg(mskg_dir=config["data"]["mskg_path"], seed=seed)
    dataco = process_dataco(dataco_dir=config["data"]["dataco_path"], seed=seed)
    datasets = {"MSKG": mskg, "DataCo": dataco}

    all_results: list[dict] = []

    for source_name, source_ds in datasets.items():
        for target_name, target_ds in datasets.items():
            if source_name == target_name:
                continue

            print(f"\n--- Source: {source_name}  ->  Target: {target_name} ---")

            # Step 1 & 2: train on source, tune lambda on source validation set
            src_train, src_val, _ = train_val_test_split(source_ds, seed=seed)
            src_model = BPRRecommender(
                embedding_dim=config["models"]["bpr"]["embedding_dim"],
                lr=config["models"]["bpr"]["learning_rate"],
                num_epochs=config["models"]["bpr"]["num_epochs"],
                patience=config["models"]["early_stopping_patience"],
                seed=seed,
            )
            print(f"  Training BPR on {source_name}...")
            src_model.fit(src_train, source_ds.item_features)

            print(f"  Tuning lambda_fair on {source_name} validation split...")
            src_all_items = source_ds.interactions["item_id"].unique()
            best_lambda = _tune_lambda_on_val(
                model=src_model,
                val_df=src_val,
                train_df=src_train,
                protected_attrs=source_ds.protected_attrs,
                all_items=src_all_items,
                lambda_grid=lambda_grid,
                max_k=max_k,
            )

            # Step 3: train a fresh BPR on the target dataset
            tgt_train, _, tgt_test = train_val_test_split(target_ds, seed=seed)
            tgt_model = BPRRecommender(
                embedding_dim=config["models"]["bpr"]["embedding_dim"],
                lr=config["models"]["bpr"]["learning_rate"],
                num_epochs=config["models"]["bpr"]["num_epochs"],
                patience=config["models"]["early_stopping_patience"],
                seed=seed,
            )
            print(f"  Training BPR on {target_name}...")
            tgt_model.fit(tgt_train, target_ds.item_features)

            tgt_all_items = target_ds.interactions["item_id"].unique()
            tgt_test_users = tgt_test["user_id"].unique()
            tgt_relevant: dict[int, set] = {}
            for uid, group in tgt_test.groupby("user_id"):
                tgt_relevant[uid] = set(group["item_id"].values)

            tgt_user_scores, tgt_user_candidates = _get_user_scores(
                tgt_model, tgt_train, tgt_test_users, tgt_all_items
            )

            # Step 4: evaluate under all three conditions
            conditions: list[tuple[str, float | None]] = [
                (f"Tuned on {source_name} (lambda={best_lambda})", best_lambda),
                (f"Default lambda ({default_lambda})", default_lambda),
                ("No re-ranking (base)", None),
            ]
            for label, lam in conditions:
                rankings: dict[int, np.ndarray] = {}
                for uid in tgt_user_scores:
                    scores = tgt_user_scores[uid]
                    candidates = tgt_user_candidates[uid]
                    if lam is None:
                        order = np.argsort(scores)[::-1][:max_k]
                        rankings[uid] = candidates[order]
                    else:
                        rankings[uid] = multi_attribute_fair_rerank(
                            scores,
                            candidates,
                            target_ds.protected_attrs,
                            top_k=max_k,
                            lambda_fair=lam,
                        )

                ranking_metrics = evaluate_ranking(rankings, tgt_relevant, ks)
                fair_agg: dict[str, list[float]] = {}
                for uid, ranking in rankings.items():
                    fm = compute_all_fairness_metrics(ranking, target_ds.protected_attrs, top_k=max_k)
                    for m, v in fm.items():
                        fair_agg.setdefault(m, []).append(v)
                fair_avg = {m: float(np.mean(v)) for m, v in fair_agg.items()}

                row: dict = {
                    "source": source_name,
                    "target": target_name,
                    "condition": label,
                    "tuned_lambda": best_lambda,
                    "applied_lambda": lam,
                }
                row.update(ranking_metrics)
                row.update(fair_avg)
                all_results.append(row)

                print(f"  [{label}]")
                for k in ks:
                    print(f"    NDCG@{k}: {ranking_metrics.get(f'ndcg@{k}', 0):.4f}")
                print(
                    f"    Size DPR: {fair_avg.get('size_dpr', 0):.4f}  "
                    f"Geo DPR: {fair_avg.get('geo_dpr', 0):.4f}  "
                    f"Inter DPR: {fair_avg.get('inter_dpr', 0):.4f}"
                )

    results_df = pd.DataFrame(all_results)
    output_path = Path("results") / "cross_dataset_generalization.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nCross-dataset results saved to {output_path}")


def main():
    config = load_config()
    set_seed(config["seed"])

    # MSKG
    print("=" * 60)
    print("Fair Re-ranking — MSKG Dataset")
    print("=" * 60)
    mskg = process_mskg(mskg_dir=config["data"]["mskg_path"], seed=config["seed"])
    run_fair_reranking(mskg, config, "mskg")

    # DataCo
    print("\n" + "=" * 60)
    print("Fair Re-ranking — DataCo Dataset")
    print("=" * 60)
    dataco = process_dataco(dataco_dir=config["data"]["dataco_path"], seed=config["seed"])
    run_fair_reranking(dataco, config, "dataco")

    # Cross-dataset generalisation
    run_cross_dataset_experiment(config)


if __name__ == "__main__":
    main()
