"""Protected group definitions for manufacturing fairness."""

import pandas as pd
import numpy as np


# Size groups based on employee count thresholds
SIZE_GROUPS = {
    "small": {"label": "Small (1-49)", "min_employees": 0, "max_employees": 50},
    "medium": {"label": "Medium (50-499)", "min_employees": 50, "max_employees": 500},
    "large": {"label": "Large (500+)", "min_employees": 500, "max_employees": float("inf")},
}

# Geographic groups (US Census regions)
GEO_GROUPS = {
    "northeast": {"label": "Northeast US"},
    "midwest": {"label": "Midwest US"},
    "south": {"label": "South US"},
    "west": {"label": "West US"},
    "unknown": {"label": "Unknown/International"},
}


def get_group_distribution(protected_attrs: pd.DataFrame, attr: str) -> dict[str, float]:
    """Get the proportion of items in each group."""
    return protected_attrs[attr].value_counts(normalize=True).to_dict()


def get_intersectional_groups(
    protected_attrs: pd.DataFrame,
    attrs: list[str] = None,
) -> pd.Series:
    """Create intersectional group labels (e.g., 'small_northeast')."""
    if attrs is None:
        attrs = ["size_group", "geo_group"]
    attrs = [a for a in attrs if a in protected_attrs.columns]
    return protected_attrs[attrs].astype(str).agg("_".join, axis=1)


def get_group_membership(
    item_ids: np.ndarray,
    protected_attrs: pd.DataFrame,
    attr: str,
) -> dict[str, np.ndarray]:
    """Return a dict mapping group -> array of item_ids in that group."""
    attr_map = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
    groups = {}
    for iid in item_ids:
        g = attr_map.get(iid, "unknown")
        groups.setdefault(g, []).append(iid)
    return {g: np.array(ids) for g, ids in groups.items()}


def compute_group_proportions_in_ranking(
    ranking: np.ndarray,
    protected_attrs: pd.DataFrame,
    attr: str,
) -> dict[str, float]:
    """Compute the proportion of each group in a ranking."""
    attr_map = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
    groups = [attr_map.get(iid, "unknown") for iid in ranking]
    total = len(groups)
    if total == 0:
        return {}
    counts = pd.Series(groups).value_counts(normalize=True).to_dict()
    return counts
