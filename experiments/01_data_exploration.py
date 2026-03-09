"""Step 1: Exploratory Data Analysis of the datasets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.mskg_processor import process_mskg
from src.data.supply_chain_processor import process_dataco


def explore_dataset(dataset, output_dir: Path):
    """Run EDA on a single dataset and save figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 50}")
    print(dataset.summary())
    print(f"{'=' * 50}")

    interactions = dataset.interactions
    protected = dataset.protected_attrs

    # 1. Interaction distribution per user
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    user_counts = interactions.groupby("user_id").size()
    axes[0].hist(user_counts, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Number of interactions")
    axes[0].set_ylabel("Number of users")
    axes[0].set_title("User interaction distribution")

    item_counts = interactions.groupby("item_id").size()
    axes[1].hist(item_counts, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("Number of interactions")
    axes[1].set_ylabel("Number of items")
    axes[1].set_title("Item interaction distribution")

    plt.tight_layout()
    plt.savefig(output_dir / "interaction_distribution.pdf", bbox_inches="tight")
    plt.close()

    # 2. Protected attribute distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "size_group" in protected.columns:
        size_dist = protected["size_group"].value_counts()
        axes[0].bar(size_dist.index, size_dist.values, color=["#2196F3", "#FF9800", "#F44336"])
        axes[0].set_title("Manufacturer size distribution")
        axes[0].set_ylabel("Count")

    if "geo_group" in protected.columns:
        geo_dist = protected["geo_group"].value_counts()
        axes[1].bar(geo_dist.index, geo_dist.values, color=sns.color_palette("Set2", len(geo_dist)))
        axes[1].set_title("Geographic distribution")
        axes[1].set_ylabel("Count")
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "protected_attribute_distribution.pdf", bbox_inches="tight")
    plt.close()

    # 3. Popularity bias analysis: interaction count by size group
    if "size_group" in protected.columns:
        item_interactions = interactions.groupby("item_id").size().reset_index(name="num_interactions")
        merged = item_interactions.merge(protected[["item_id", "size_group"]], on="item_id", how="left")

        fig, ax = plt.subplots(figsize=(8, 5))
        for group in ["small", "medium", "large"]:
            subset = merged[merged["size_group"] == group]["num_interactions"]
            if len(subset) > 0:
                ax.hist(subset, bins=30, alpha=0.5, label=f"{group} (n={len(subset)})")

        ax.set_xlabel("Number of interactions")
        ax.set_ylabel("Number of manufacturers")
        ax.set_title("Popularity bias by manufacturer size")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "popularity_bias.pdf", bbox_inches="tight")
        plt.close()

        # Print bias statistics
        print("\nPopularity bias by size group:")
        for group in ["small", "medium", "large"]:
            subset = merged[merged["size_group"] == group]["num_interactions"]
            if len(subset) > 0:
                print(f"  {group}: mean={subset.mean():.1f}, median={subset.median():.1f}, "
                      f"count={len(subset)}")

    # 4. Summary statistics
    stats = {
        "dataset": dataset.name,
        "num_users": dataset.num_users,
        "num_items": dataset.num_items,
        "num_interactions": dataset.num_interactions,
        "density": dataset.density,
        "avg_interactions_per_user": user_counts.mean(),
        "avg_interactions_per_item": item_counts.mean(),
    }
    pd.DataFrame([stats]).to_csv(output_dir / "summary_stats.csv", index=False)
    print(f"\nSummary stats saved to {output_dir / 'summary_stats.csv'}")


def main():
    config = load_config()
    set_seed(config["seed"])

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Process MSKG
    print("Processing MSKG dataset...")
    mskg = process_mskg(
        mskg_dir=config["data"]["mskg_path"],
        seed=config["seed"],
    )
    explore_dataset(mskg, results_dir / "eda" / "mskg")

    # Process DataCo
    print("\nProcessing DataCo dataset...")
    dataco = process_dataco(
        dataco_dir=config["data"]["dataco_path"],
        seed=config["seed"],
    )
    explore_dataset(dataco, results_dir / "eda" / "dataco")

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
