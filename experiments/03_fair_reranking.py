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
from src.models.fair_reranking import fair_rerank
from src.evaluation.ranking_metrics import evaluate_ranking
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

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = Path("results") / f"{output_prefix}_fair_reranking.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df


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


if __name__ == "__main__":
    main()
