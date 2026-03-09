"""Step 4: Ablation study and sensitivity analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.mskg_processor import process_mskg
from src.data.loader import train_val_test_split
from src.models.collaborative import BPRRecommender
from src.models.graph_based import LightGCNRecommender
from src.models.fair_reranking import multi_attribute_fair_rerank, det_const_sort
from src.evaluation.ranking_metrics import evaluate_ranking
from src.fairness.metrics import compute_all_fairness_metrics


def ablation_components(dataset, config: dict):
    """Ablation: remove each component of our method and measure impact."""
    seed = config["seed"]
    train_df, val_df, test_df = train_val_test_split(dataset, seed=seed)
    ks = config["evaluation"]["top_k"]
    max_k = max(ks)

    # Use LightGCN as base
    base_model = LightGCNRecommender(
        embedding_dim=config["models"]["lightgcn"]["embedding_dim"],
        num_layers=config["models"]["lightgcn"]["num_layers"],
        lr=config["models"]["lightgcn"]["learning_rate"],
        num_epochs=config["models"]["lightgcn"]["num_epochs"],
        patience=config["models"]["early_stopping_patience"],
        seed=seed,
    )
    print("  Training LightGCN base model...")
    base_model.fit(train_df, dataset.item_features)

    all_items = dataset.interactions["item_id"].unique()
    test_users = test_df["user_id"].unique()
    relevant_sets = {}
    for uid, group in test_df.groupby("user_id"):
        relevant_sets[uid] = set(group["item_id"].values)

    train_items_per_user = {}
    for uid, group in train_df.groupby("user_id"):
        train_items_per_user[uid] = set(group["item_id"].values)

    # Get base scores
    user_data = {}
    for uid in test_users:
        train_seen = train_items_per_user.get(uid, set())
        candidates = np.array([i for i in all_items if i not in train_seen])
        if len(candidates) == 0:
            continue
        scores = base_model.predict(uid, candidates)
        user_data[uid] = (candidates, scores)

    # Ablation variants
    variants = {
        "Full method (ours)": {"attrs": ["size_group", "geo_group"], "lambda_fair": 0.5},
        "No geo fairness": {"attrs": ["size_group"], "lambda_fair": 0.5},
        "No size fairness": {"attrs": ["geo_group"], "lambda_fair": 0.5},
        "Single-attr (DetConstSort on size)": "detconstsort_size",
        "No re-ranking (base)": "none",
    }

    results = []
    for variant_name, params in variants.items():
        print(f"\n  Variant: {variant_name}")
        rankings = {}

        for uid, (candidates, scores) in user_data.items():
            if params == "none":
                order = np.argsort(scores)[::-1][:max_k]
                rankings[uid] = candidates[order]
            elif params == "detconstsort_size":
                rankings[uid] = det_const_sort(
                    scores, candidates, dataset.protected_attrs,
                    top_k=max_k, attr="size_group",
                )
            else:
                rankings[uid] = multi_attribute_fair_rerank(
                    scores, candidates, dataset.protected_attrs,
                    top_k=max_k, lambda_fair=params["lambda_fair"],
                    attrs=params["attrs"],
                )

        ranking_metrics = evaluate_ranking(rankings, relevant_sets, ks)
        fair_metrics = {}
        for uid, ranking in rankings.items():
            uf = compute_all_fairness_metrics(ranking, dataset.protected_attrs, top_k=max_k)
            for m, v in uf.items():
                fair_metrics.setdefault(m, []).append(v)
        fair_metrics = {m: np.mean(v) for m, v in fair_metrics.items()}

        row = {"variant": variant_name}
        row.update(ranking_metrics)
        row.update(fair_metrics)
        results.append(row)

        for k in ks:
            print(f"    NDCG@{k}: {ranking_metrics.get(f'ndcg@{k}', 0):.4f}")
        print(f"    Size DPR: {fair_metrics.get('size_dpr', 0):.4f}  "
              f"Geo DPR: {fair_metrics.get('geo_dpr', 0):.4f}")

    return pd.DataFrame(results)


def sensitivity_lambda(dataset, config: dict):
    """Sensitivity analysis: vary lambda_fair and measure trade-off."""
    seed = config["seed"]
    train_df, val_df, test_df = train_val_test_split(dataset, seed=seed)
    ks = config["evaluation"]["top_k"]
    max_k = max(ks)

    base_model = BPRRecommender(
        embedding_dim=config["models"]["bpr"]["embedding_dim"],
        lr=config["models"]["bpr"]["learning_rate"],
        num_epochs=config["models"]["bpr"]["num_epochs"],
        patience=config["models"]["early_stopping_patience"],
        seed=seed,
    )
    print("  Training BPR base model...")
    base_model.fit(train_df, dataset.item_features)

    all_items = dataset.interactions["item_id"].unique()
    test_users = test_df["user_id"].unique()
    relevant_sets = {}
    for uid, group in test_df.groupby("user_id"):
        relevant_sets[uid] = set(group["item_id"].values)

    train_items_per_user = {}
    for uid, group in train_df.groupby("user_id"):
        train_items_per_user[uid] = set(group["item_id"].values)

    user_data = {}
    for uid in test_users:
        train_seen = train_items_per_user.get(uid, set())
        candidates = np.array([i for i in all_items if i not in train_seen])
        if len(candidates) == 0:
            continue
        scores = base_model.predict(uid, candidates)
        user_data[uid] = (candidates, scores)

    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for lam in lambda_values:
        print(f"\n  lambda_fair = {lam}")
        rankings = {}
        for uid, (candidates, scores) in user_data.items():
            rankings[uid] = multi_attribute_fair_rerank(
                scores, candidates, dataset.protected_attrs,
                top_k=max_k, lambda_fair=lam,
            )

        ranking_metrics = evaluate_ranking(rankings, relevant_sets, ks)
        fair_metrics = {}
        for uid, ranking in rankings.items():
            uf = compute_all_fairness_metrics(ranking, dataset.protected_attrs, top_k=max_k)
            for m, v in uf.items():
                fair_metrics.setdefault(m, []).append(v)
        fair_metrics = {m: np.mean(v) for m, v in fair_metrics.items()}

        row = {"lambda_fair": lam}
        row.update(ranking_metrics)
        row.update(fair_metrics)
        results.append(row)

        print(f"    NDCG@10: {ranking_metrics.get('ndcg@10', 0):.4f}  "
              f"Size DPR: {fair_metrics.get('size_dpr', 0):.4f}")

    return pd.DataFrame(results)


def main():
    config = load_config()
    set_seed(config["seed"])

    print("=" * 60)
    print("Ablation Study — MSKG Dataset")
    print("=" * 60)

    mskg = process_mskg(mskg_dir=config["data"]["mskg_path"], seed=config["seed"])

    # Component ablation
    print("\n--- Component Ablation ---")
    ablation_df = ablation_components(mskg, config)
    ablation_df.to_csv("results/ablation_components.csv", index=False)
    print(f"\nAblation results saved to results/ablation_components.csv")

    # Lambda sensitivity
    print("\n--- Lambda Sensitivity ---")
    sensitivity_df = sensitivity_lambda(mskg, config)
    sensitivity_df.to_csv("results/sensitivity_lambda.csv", index=False)
    print(f"Sensitivity results saved to results/sensitivity_lambda.csv")


if __name__ == "__main__":
    main()
