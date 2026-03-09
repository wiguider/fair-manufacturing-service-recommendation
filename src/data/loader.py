"""Generic data loading utilities for recommendation datasets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class RecDataset:
    """Standard format for recommendation datasets."""
    interactions: pd.DataFrame  # columns: user_id, item_id, rating
    item_features: pd.DataFrame  # columns: item_id, + feature columns
    protected_attrs: pd.DataFrame  # columns: item_id, size_group, geo_group
    user_features: Optional[pd.DataFrame] = None
    name: str = "unknown"

    @property
    def num_users(self) -> int:
        return self.interactions["user_id"].nunique()

    @property
    def num_items(self) -> int:
        return self.interactions["item_id"].nunique()

    @property
    def num_interactions(self) -> int:
        return len(self.interactions)

    @property
    def density(self) -> float:
        return self.num_interactions / (self.num_users * self.num_items)

    def get_interaction_matrix(self) -> csr_matrix:
        users = self.interactions["user_id"].values
        items = self.interactions["item_id"].values
        ratings = self.interactions["rating"].values if "rating" in self.interactions else np.ones(len(users))
        return csr_matrix(
            (ratings, (users, items)),
            shape=(self.num_users, self.num_items),
        )

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.name}",
            f"  Users: {self.num_users}",
            f"  Items: {self.num_items}",
            f"  Interactions: {self.num_interactions}",
            f"  Density: {self.density:.6f}",
        ]
        if self.protected_attrs is not None:
            for col in ["size_group", "geo_group"]:
                if col in self.protected_attrs.columns:
                    dist = self.protected_attrs[col].value_counts().to_dict()
                    lines.append(f"  {col} distribution: {dist}")
        return "\n".join(lines)


def train_val_test_split(
    dataset: RecDataset, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-one-out style split per user: last interaction = test, second-to-last = val."""
    df = dataset.interactions.copy()

    # Sort by user and assign a rank within each user (random for implicit)
    rng = np.random.RandomState(seed)
    df["_rand"] = rng.rand(len(df))
    df = df.sort_values(["user_id", "_rand"])
    df["_rank"] = df.groupby("user_id").cumcount(ascending=False)

    test = df[df["_rank"] == 0].drop(columns=["_rand", "_rank"])
    val = df[df["_rank"] == 1].drop(columns=["_rand", "_rank"])
    train = df[df["_rank"] >= 2].drop(columns=["_rand", "_rank"])

    # Users with fewer than 3 interactions: put everything in train
    sparse_users = df.groupby("user_id").size()
    sparse_users = sparse_users[sparse_users < 3].index
    if len(sparse_users) > 0:
        train = pd.concat([train, val[val["user_id"].isin(sparse_users)], test[test["user_id"].isin(sparse_users)]])
        val = val[~val["user_id"].isin(sparse_users)]
        test = test[~test["user_id"].isin(sparse_users)]

    return train, val, test
