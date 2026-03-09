"""Step 2: Train and evaluate baseline recommendation models."""

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
from src.models.baselines import RandomRecommender, PopularityRecommender, ContentBasedRecommender
from src.models.collaborative import BPRRecommender, NeuMFRecommender
from src.models.graph_based import LightGCNRecommender
from src.evaluation.ranking_metrics import evaluate_ranking
from src.fairness.metrics import compute_all_fairness_metrics


def build_relevant_sets(test_df: pd.DataFrame) -> dict[int, set]:
    """Build per-user relevant item sets from test interactions."""
    relevant = {}
    for uid, group in test_df.groupby("user_id"):
        relevant[uid] = set(group["item_id"].values)
    return relevant


def evaluate_model(
    model_name: str,
    model,
    dataset,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Train a model and evaluate it."""
    print(f"\n--- {model_name} ---")

    # Train
    print(f"  Training {model_name}...")
    model.fit(train_df, dataset.item_features)

    # Get candidate items and test users
    all_items = dataset.interactions["item_id"].unique()
    test_users = test_df["user_id"].unique()
    relevant_sets = build_relevant_sets(test_df)

    # Build per-user training items for filtering
    train_items_per_user = {}
    for uid, group in train_df.groupby("user_id"):
        train_items_per_user[uid] = set(group["item_id"].values)

    # Generate recommendations
    ks = config["evaluation"]["top_k"]
    max_k = max(ks)

    rankings = {}
    for uid in test_users:
        # Candidate items: all items minus training items
        train_seen = train_items_per_user.get(uid, set())
        candidates = np.array([i for i in all_items if i not in train_seen])
        if len(candidates) == 0:
            continue
        ranking = model.recommend(uid, candidates, top_k=max_k)
        rankings[uid] = ranking

    # Evaluate ranking quality
    print(f"  Evaluating {model_name}...")
    ranking_results = evaluate_ranking(rankings, relevant_sets, ks)

    # Evaluate fairness (averaged over users)
    fairness_results = {}
    for uid, ranking in rankings.items():
        user_fair = compute_all_fairness_metrics(ranking, dataset.protected_attrs, top_k=max_k)
        for metric, val in user_fair.items():
            fairness_results.setdefault(metric, []).append(val)

    for metric in fairness_results:
        fairness_results[metric] = np.mean(fairness_results[metric])

    # Combine
    all_results = {"model": model_name}
    all_results.update(ranking_results)
    all_results.update(fairness_results)

    # Print summary
    for k in ks:
        print(f"  NDCG@{k}: {ranking_results.get(f'ndcg@{k}', 0):.4f}  "
              f"Prec@{k}: {ranking_results.get(f'precision@{k}', 0):.4f}")
    print(f"  MRR: {ranking_results.get('mrr', 0):.4f}  MAP: {ranking_results.get('map', 0):.4f}")
    print(f"  Size DPR: {fairness_results.get('size_dpr', 0):.4f}  "
          f"Geo DPR: {fairness_results.get('geo_dpr', 0):.4f}")

    return all_results


def run_baselines(dataset, config: dict, output_prefix: str):
    """Run all baselines on a dataset."""
    seed = config["seed"]
    train_df, val_df, test_df = train_val_test_split(dataset, seed=seed)

    print(f"\nDataset: {dataset.name}")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define models
    models = {
        "Random": RandomRecommender(seed=seed),
        "Popularity": PopularityRecommender(),
        "ContentBased": ContentBasedRecommender(max_features=config["models"]["content_based"]["max_features"]),
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

    all_results = []
    for name, model in models.items():
        result = evaluate_model(name, model, dataset, train_df, test_df, config)
        all_results.append(result)

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = Path("results") / f"{output_prefix}_baselines.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df


def main():
    config = load_config()
    set_seed(config["seed"])

    # MSKG
    print("=" * 60)
    print("MSKG Dataset")
    print("=" * 60)
    mskg = process_mskg(mskg_dir=config["data"]["mskg_path"], seed=config["seed"])
    run_baselines(mskg, config, "mskg")

    # DataCo
    print("\n" + "=" * 60)
    print("DataCo Dataset")
    print("=" * 60)
    dataco = process_dataco(dataco_dir=config["data"]["dataco_path"], seed=config["seed"])
    run_baselines(dataco, config, "dataco")


if __name__ == "__main__":
    main()
