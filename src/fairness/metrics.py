"""Fairness metrics for ranking evaluation.

References:
- DPR: Demographic Parity Ratio
- Equity of Exposure (Diaz et al., 2020)
- Expected Exposure Loss (Singh & Joachims, 2018)
"""

import numpy as np
import pandas as pd

from src.fairness.groups import get_group_distribution


def demographic_parity_ratio(
    ranking: np.ndarray,
    protected_attrs: pd.DataFrame,
    attr: str = "size_group",
    top_k: int = None,
) -> float:
    """Demographic Parity Ratio: min(group_proportion) / max(group_proportion) in ranking.

    A value of 1.0 means perfect parity; lower values indicate more disparity.
    """
    if top_k is not None:
        ranking = ranking[:top_k]

    attr_map = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
    groups = [attr_map.get(iid, "unknown") for iid in ranking]

    if len(groups) == 0:
        return 0.0

    counts = pd.Series(groups).value_counts(normalize=True)
    if len(counts) <= 1:
        return 1.0

    return counts.min() / counts.max()


def equity_of_exposure(
    ranking: np.ndarray,
    protected_attrs: pd.DataFrame,
    attr: str = "size_group",
    top_k: int = None,
) -> float:
    """Equity of Exposure (Diaz et al., 2020).

    Measures how equally exposure is distributed across groups relative
    to their target proportions. Lower is more fair.
    Returns the L2 distance between actual and target exposure distributions.
    """
    if top_k is not None:
        ranking = ranking[:top_k]

    k = len(ranking)
    if k == 0:
        return 0.0

    # Position-based exposure: 1/log2(rank+1)
    exposure_weights = 1.0 / np.log2(np.arange(1, k + 1) + 1)

    attr_map = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
    target_dist = get_group_distribution(protected_attrs, attr)

    # Compute group exposure
    group_exposure = {}
    total_exposure = exposure_weights.sum()

    for pos, iid in enumerate(ranking):
        g = attr_map.get(iid, "unknown")
        group_exposure[g] = group_exposure.get(g, 0.0) + exposure_weights[pos]

    # Normalize
    for g in group_exposure:
        group_exposure[g] /= total_exposure

    # L2 distance from target
    all_groups = set(list(target_dist.keys()) + list(group_exposure.keys()))
    diff_sq = 0.0
    for g in all_groups:
        actual = group_exposure.get(g, 0.0)
        target = target_dist.get(g, 0.0)
        diff_sq += (actual - target) ** 2

    return np.sqrt(diff_sq)


def expected_exposure_loss(
    ranking: np.ndarray,
    protected_attrs: pd.DataFrame,
    attr: str = "size_group",
    top_k: int = None,
) -> float:
    """Expected Exposure Loss (Singh & Joachims, 2018).

    Measures the expected squared deviation of group exposure from
    their merit-proportional target. Lower is more fair.
    """
    if top_k is not None:
        ranking = ranking[:top_k]

    k = len(ranking)
    if k == 0:
        return 0.0

    # Exposure at each position
    exposure_weights = 1.0 / np.log2(np.arange(1, k + 1) + 1)

    attr_map = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
    target_dist = get_group_distribution(protected_attrs, attr)

    # Compute per-group exposure
    group_exposure = {}
    for pos, iid in enumerate(ranking):
        g = attr_map.get(iid, "unknown")
        group_exposure[g] = group_exposure.get(g, 0.0) + exposure_weights[pos]

    total_exposure = exposure_weights.sum()

    # Target exposure per group (proportional to group size)
    all_groups = set(list(target_dist.keys()) + list(group_exposure.keys()))
    loss = 0.0
    for g in all_groups:
        actual = group_exposure.get(g, 0.0)
        target = target_dist.get(g, 0.0) * total_exposure
        loss += (actual - target) ** 2

    return loss / len(all_groups) if all_groups else 0.0


def intersectional_fairness(
    ranking: np.ndarray,
    protected_attrs: pd.DataFrame,
    attrs: list[str] = None,
    top_k: int = None,
) -> dict[str, float]:
    """Compute fairness metrics for intersectional groups (size x geo)."""
    if attrs is None:
        attrs = ["size_group", "geo_group"]
    if top_k is not None:
        ranking = ranking[:top_k]

    k = len(ranking)
    if k == 0:
        return {"inter_dpr": 0.0, "inter_eoe": 0.0}

    # Build intersectional group
    inter = protected_attrs.copy()
    valid_attrs = [a for a in attrs if a in inter.columns]
    inter["inter_group"] = inter[valid_attrs].astype(str).agg("_".join, axis=1)

    # Use existing metrics with intersectional group
    inter_attrs = inter[["item_id", "inter_group"]].rename(columns={"inter_group": "inter"})

    return {
        "inter_dpr": demographic_parity_ratio(ranking, inter_attrs, attr="inter", top_k=None),
        "inter_eoe": equity_of_exposure(ranking, inter_attrs, attr="inter", top_k=None),
        "inter_eel": expected_exposure_loss(ranking, inter_attrs, attr="inter", top_k=None),
    }


def compute_all_fairness_metrics(
    ranking: np.ndarray,
    protected_attrs: pd.DataFrame,
    top_k: int = None,
) -> dict[str, float]:
    """Compute all fairness metrics at once."""
    results = {}

    for attr in ["size_group", "geo_group"]:
        if attr not in protected_attrs.columns:
            continue
        prefix = attr.replace("_group", "")
        results[f"{prefix}_dpr"] = demographic_parity_ratio(ranking, protected_attrs, attr, top_k)
        results[f"{prefix}_eoe"] = equity_of_exposure(ranking, protected_attrs, attr, top_k)
        results[f"{prefix}_eel"] = expected_exposure_loss(ranking, protected_attrs, attr, top_k)

    # Intersectional
    inter = intersectional_fairness(ranking, protected_attrs, top_k=top_k)
    results.update(inter)

    return results
