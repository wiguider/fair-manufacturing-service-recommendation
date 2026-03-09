"""Step 5: Domain analysis — supply chain health metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.mskg_processor import process_mskg
from src.data.supply_chain_processor import process_dataco
from src.data.loader import train_val_test_split
from src.models.collaborative import BPRRecommender
from src.models.graph_based import LightGCNRecommender
from src.models.fair_reranking import fair_rerank, multi_attribute_fair_rerank
from src.evaluation.ranking_metrics import evaluate_ranking
from src.fairness.metrics import compute_all_fairness_metrics
from src.fairness.domain_metrics import compute_all_domain_metrics


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _build_methods(config: dict) -> list[dict]:
    """Return a list of method descriptors used across both datasets."""
    lambda_fair = config["fair_reranking"]["lambda_fair"]
    alpha = config["fair_reranking"]["fair_alpha"]

    return [
        {"label": "BPR (no rerank)",        "base": "bpr",      "rerank": None,          "lambda_fair": None,      "alpha": alpha},
        {"label": "BPR + FA*IR",             "base": "bpr",      "rerank": "fair",        "lambda_fair": lambda_fair, "alpha": alpha},
        {"label": "BPR + DetConstSort",      "base": "bpr",      "rerank": "detconstsort","lambda_fair": lambda_fair, "alpha": alpha},
        {"label": "BPR + Ours",              "base": "bpr",      "rerank": "ours",        "lambda_fair": lambda_fair, "alpha": alpha},
        {"label": "LightGCN (no rerank)",    "base": "lightgcn", "rerank": None,          "lambda_fair": None,      "alpha": alpha},
        {"label": "LightGCN + FA*IR",        "base": "lightgcn", "rerank": "fair",        "lambda_fair": lambda_fair, "alpha": alpha},
        {"label": "LightGCN + DetConstSort", "base": "lightgcn", "rerank": "detconstsort","lambda_fair": lambda_fair, "alpha": alpha},
        {"label": "LightGCN + Ours",         "base": "lightgcn", "rerank": "ours",        "lambda_fair": lambda_fair, "alpha": alpha},
    ]


def _make_base_model(base_key: str, config: dict, seed: int):
    if base_key == "bpr":
        return BPRRecommender(
            embedding_dim=config["models"]["bpr"]["embedding_dim"],
            lr=config["models"]["bpr"]["learning_rate"],
            num_epochs=config["models"]["bpr"]["num_epochs"],
            patience=config["models"]["early_stopping_patience"],
            seed=seed,
        )
    if base_key == "lightgcn":
        return LightGCNRecommender(
            embedding_dim=config["models"]["lightgcn"]["embedding_dim"],
            num_layers=config["models"]["lightgcn"]["num_layers"],
            lr=config["models"]["lightgcn"]["learning_rate"],
            num_epochs=config["models"]["lightgcn"]["num_epochs"],
            patience=config["models"]["early_stopping_patience"],
            seed=seed,
        )
    raise ValueError(f"Unknown base model key: {base_key}")


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_domain_analysis(dataset, config: dict) -> pd.DataFrame:
    """Train all methods, evaluate ranking + fairness + domain metrics.

    Returns a DataFrame with one row per method.
    """
    seed = config["seed"]
    train_df, val_df, test_df = train_val_test_split(dataset, seed=seed)
    ks = config["evaluation"]["top_k"]
    max_k = max(ks)

    print(f"\n  Dataset: {dataset.name}")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    all_items = dataset.interactions["item_id"].unique()
    test_users = test_df["user_id"].unique()

    relevant_sets: dict = {}
    for uid, group in test_df.groupby("user_id"):
        relevant_sets[uid] = set(group["item_id"].values)

    train_items_per_user: dict = {}
    for uid, group in train_df.groupby("user_id"):
        train_items_per_user[uid] = set(group["item_id"].values)

    methods = _build_methods(config)

    # Pre-train each unique base model once
    trained_models: dict[str, object] = {}
    base_scores_cache: dict[str, dict] = {}

    for base_key in ("bpr", "lightgcn"):
        model = _make_base_model(base_key, config, seed)
        print(f"\n  Training {base_key.upper()}...")
        model.fit(train_df, dataset.item_features)
        trained_models[base_key] = model

        user_data: dict = {}
        for uid in test_users:
            train_seen = train_items_per_user.get(uid, set())
            candidates = np.array([i for i in all_items if i not in train_seen])
            if len(candidates) == 0:
                continue
            scores = model.predict(uid, candidates)
            user_data[uid] = (candidates, scores)
        base_scores_cache[base_key] = user_data

    all_results = []

    for method in methods:
        label = method["label"]
        base_key = method["base"]
        rerank = method["rerank"]
        lam = method["lambda_fair"]
        alpha = method["alpha"]

        print(f"\n  Method: {label}")
        user_data = base_scores_cache[base_key]

        rankings: dict = {}
        for uid, (candidates, scores) in user_data.items():
            if rerank is None:
                order = np.argsort(scores)[::-1][:max_k]
                rankings[uid] = candidates[order]
            else:
                rankings[uid] = fair_rerank(
                    scores=scores,
                    item_ids=candidates,
                    protected_attrs=dataset.protected_attrs,
                    method=rerank,
                    top_k=max_k,
                    lambda_fair=lam,
                    alpha=alpha,
                )

        # Ranking metrics
        ranking_metrics = evaluate_ranking(rankings, relevant_sets, ks)

        # Fairness metrics (averaged over users)
        fair_accum: dict = {}
        for uid, ranking in rankings.items():
            uf = compute_all_fairness_metrics(ranking, dataset.protected_attrs, top_k=max_k)
            for m, v in uf.items():
                fair_accum.setdefault(m, []).append(v)
        fair_metrics = {m: float(np.mean(v)) for m, v in fair_accum.items()}

        # Domain metrics (averaged over users)
        domain_accum: dict = {}
        for uid, ranking in rankings.items():
            dm = compute_all_domain_metrics(
                ranking,
                dataset.item_features,
                dataset.protected_attrs,
                top_k=max_k,
            )
            for m, v in dm.items():
                domain_accum.setdefault(m, []).append(v)
        domain_metrics = {m: float(np.mean(v)) for m, v in domain_accum.items()}

        row = {"method": label}
        row.update(ranking_metrics)
        row.update(fair_metrics)
        row.update(domain_metrics)
        all_results.append(row)

        print(f"    NDCG@10:   {ranking_metrics.get('ndcg@10', 0):.4f}")
        print(f"    Inter DPR: {fair_metrics.get('inter_dpr', 0):.4f}")
        print(f"    HHI:       {domain_metrics.get('hhi', 0):.4f}")
        print(f"    Cert cov:  {domain_metrics.get('cert_coverage', 0):.4f}")
        print(f"    Resilience:{domain_metrics.get('regional_resilience', 0):.4f}")

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _method_style(label: str) -> tuple[str, str]:
    """Return (color, marker) for a method label."""
    color_map = {
        "BPR (no rerank)":        "#1f77b4",
        "BPR + FA*IR":            "#aec7e8",
        "BPR + DetConstSort":     "#ffbb78",
        "BPR + Ours":             "#d62728",
        "LightGCN (no rerank)":   "#2ca02c",
        "LightGCN + FA*IR":       "#98df8a",
        "LightGCN + DetConstSort":"#ff9896",
        "LightGCN + Ours":        "#9467bd",
    }
    marker_map = {
        "no rerank":   "o",
        "FA*IR":       "s",
        "DetConstSort":"^",
        "Ours":        "D",
    }
    color = color_map.get(label, "#333333")
    marker = "o"
    for key, m in marker_map.items():
        if key in label:
            marker = m
            break
    return color, marker


def plot_domain_metrics(results_df: pd.DataFrame, output_path: str, dataset_name: str) -> None:
    """Grouped bar chart: HHI (lower=better), cert coverage, and resilience."""
    metrics = ["hhi", "cert_coverage", "regional_resilience"]
    labels_human = ["HHI (lower=better)", "Cert Coverage (higher=better)", "Resilience (higher=better)"]

    methods = results_df["method"].tolist()
    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    bar_width = 0.8 / n_methods
    offsets = (np.arange(n_methods) - n_methods / 2.0 + 0.5) * bar_width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        row = results_df[results_df["method"] == method].iloc[0]
        values = [row.get(m, 0.0) for m in metrics]
        color, _ = _method_style(method)
        bars = ax.bar(x + offsets[i], values, bar_width * 0.9, label=method, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_human, fontsize=10)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Domain Metrics by Method — {dataset_name}")
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved domain metrics chart to {output_path}")


def plot_pareto_front(results_df: pd.DataFrame, output_path: str, dataset_name: str) -> None:
    """Scatter plot: NDCG@10 (x) vs intersectional DPR (y) for all methods.

    Better methods sit toward the top-right (high relevance, high fairness).
    """
    if "ndcg@10" not in results_df.columns or "inter_dpr" not in results_df.columns:
        print("  Skipping Pareto plot — required columns missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in results_df.iterrows():
        label = row["method"]
        x = row["ndcg@10"]
        y = row["inter_dpr"]
        color, marker = _method_style(label)
        ax.scatter(x, y, color=color, marker=marker, s=120, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color=color,
        )

    # Shade the "ideal" corner (upper-right)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.annotate(
        "Ideal\nregion",
        xy=(xlim[1], ylim[1]),
        xytext=(xlim[1] - 0.15 * (xlim[1] - xlim[0]),
                ylim[1] - 0.12 * (ylim[1] - ylim[0])),
        fontsize=8,
        color="gray",
        style="italic",
    )

    ax.set_xlabel("NDCG@10 (higher = better relevance)", fontsize=11)
    ax.set_ylabel("Intersectional DPR (higher = more fair)", fontsize=11)
    ax.set_title(f"Pareto Front: Relevance vs Fairness — {dataset_name}", fontsize=12)
    ax.grid(linestyle="--", alpha=0.4)

    # Legend: one entry per unique marker shape
    shape_legend = {
        "No re-rank": mpatches.Patch(color="gray", label="No re-rank (circle)"),
        "FA*IR":      mpatches.Patch(color="gray", label="FA*IR (square)"),
        "DetConstSort": mpatches.Patch(color="gray", label="DetConstSort (triangle)"),
        "Ours":       mpatches.Patch(color="gray", label="Ours (diamond)"),
    }
    # Use colored patches for base models
    bpr_patch = mpatches.Patch(color="#1f77b4", label="BPR family")
    lgcn_patch = mpatches.Patch(color="#2ca02c", label="LightGCN family")
    ax.legend(handles=[bpr_patch, lgcn_patch], loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved Pareto front plot to {output_path}")


def print_summary_table(results_df: pd.DataFrame, dataset_name: str) -> None:
    """Pretty-print key columns to stdout."""
    cols = ["method", "ndcg@10", "inter_dpr", "hhi", "cert_coverage", "regional_resilience"]
    available = [c for c in cols if c in results_df.columns]
    print(f"\n  === Domain Analysis Summary — {dataset_name} ===")
    print(results_df[available].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MSKG
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Domain Analysis — MSKG Dataset")
    print("=" * 60)

    mskg = process_mskg(mskg_dir=config["data"]["mskg_path"], seed=config["seed"])
    mskg_results = run_domain_analysis(mskg, config)

    mskg_results.to_csv(results_dir / "mskg_domain_analysis.csv", index=False)
    print(f"\n  Results saved to results/mskg_domain_analysis.csv")

    print_summary_table(mskg_results, "MSKG")

    plot_domain_metrics(
        mskg_results,
        str(results_dir / "mskg_domain_metrics.png"),
        "MSKG",
    )
    plot_pareto_front(
        mskg_results,
        str(results_dir / "mskg_pareto_front.png"),
        "MSKG",
    )

    # ------------------------------------------------------------------
    # DataCo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Domain Analysis — DataCo Dataset")
    print("=" * 60)

    dataco = process_dataco(dataco_dir=config["data"]["dataco_path"], seed=config["seed"])
    dataco_results = run_domain_analysis(dataco, config)

    dataco_results.to_csv(results_dir / "dataco_domain_analysis.csv", index=False)
    print(f"\n  Results saved to results/dataco_domain_analysis.csv")

    print_summary_table(dataco_results, "DataCo")

    plot_domain_metrics(
        dataco_results,
        str(results_dir / "dataco_domain_metrics.png"),
        "DataCo",
    )
    plot_pareto_front(
        dataco_results,
        str(results_dir / "dataco_pareto_front.png"),
        "DataCo",
    )


if __name__ == "__main__":
    main()
