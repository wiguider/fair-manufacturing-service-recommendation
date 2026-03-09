"""Fair re-ranking methods: FA*IR, DetConstSort, and our proposed method.

References:
- Zehlike et al. (2017) — FA*IR: A Fair Top-k Ranking Algorithm
- Geyik et al. (2019) — Fairness-Aware Ranking in Search & Recommendation
- Singh & Joachims (2018) — Fairness of Exposure in Rankings
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binom


def fair_rerank(
    scores: np.ndarray,
    item_ids: np.ndarray,
    protected_attrs: pd.DataFrame,
    method: str = "ours",
    top_k: int = 10,
    **kwargs,
) -> np.ndarray:
    """Apply fair re-ranking to a scored list of items.

    Args:
        scores: Relevance scores for each item.
        item_ids: Item IDs corresponding to scores.
        protected_attrs: DataFrame with item_id, size_group, geo_group.
        method: One of "fair", "detconstsort", "ours".
        top_k: Number of items to return.

    Returns:
        Array of item_ids in fair-reranked order.
    """
    if method == "fair":
        return fair_topk(scores, item_ids, protected_attrs, top_k, **kwargs)
    elif method == "detconstsort":
        return det_const_sort(scores, item_ids, protected_attrs, top_k, **kwargs)
    elif method == "ours":
        return multi_attribute_fair_rerank(scores, item_ids, protected_attrs, top_k, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---- FA*IR (Zehlike et al., 2017) ----

def fair_topk(
    scores: np.ndarray,
    item_ids: np.ndarray,
    protected_attrs: pd.DataFrame,
    top_k: int = 10,
    attr: str = "size_group",
    protected_value: str = "small",
    alpha: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """FA*IR algorithm for binary protected group fairness."""
    # Map item_ids to their protected attribute
    attr_map = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
    is_protected = np.array([attr_map.get(iid, "") == protected_value for iid in item_ids])

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    sorted_ids = item_ids[order]
    sorted_protected = is_protected[order]

    # Separate protected and unprotected candidates
    protected_queue = sorted_ids[sorted_protected]
    unprotected_queue = sorted_ids[~sorted_protected]

    # Proportion of protected items
    p = is_protected.sum() / len(is_protected) if len(is_protected) > 0 else 0.5

    # Compute minimum number of protected items at each position
    result = []
    p_idx, u_idx = 0, 0

    for k in range(1, top_k + 1):
        # Minimum protected items needed at position k (FA*IR constraint)
        min_protected = _fair_min_protected(k, p, alpha)
        protected_so_far = sum(1 for iid in result if attr_map.get(iid, "") == protected_value)

        if protected_so_far < min_protected and p_idx < len(protected_queue):
            result.append(protected_queue[p_idx])
            p_idx += 1
        elif u_idx < len(unprotected_queue):
            result.append(unprotected_queue[u_idx])
            u_idx += 1
        elif p_idx < len(protected_queue):
            result.append(protected_queue[p_idx])
            p_idx += 1

    return np.array(result[:top_k])


def _fair_min_protected(k: int, p: float, alpha: float) -> int:
    """Compute minimum number of protected items in top-k using inverse binomial CDF."""
    if p <= 0:
        return 0
    for m in range(k + 1):
        if binom.cdf(m, k, p) >= alpha:
            return m
    return 0


# ---- DetConstSort (Geyik et al., 2019) ----

def det_const_sort(
    scores: np.ndarray,
    item_ids: np.ndarray,
    protected_attrs: pd.DataFrame,
    top_k: int = 10,
    attr: str = "size_group",
    target_proportions: Optional[dict] = None,
    **kwargs,
) -> np.ndarray:
    """Deterministic Constrained Sorting for multi-group fairness."""
    attr_map = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
    groups = protected_attrs[attr].unique()

    # Default: proportional representation
    if target_proportions is None:
        group_counts = protected_attrs[attr].value_counts(normalize=True).to_dict()
        target_proportions = group_counts

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    sorted_ids = item_ids[order]

    # Build per-group queues sorted by score
    group_queues = {g: [] for g in groups}
    for iid in sorted_ids:
        g = attr_map.get(iid, groups[0])
        group_queues[g].append(iid)

    # Greedily fill positions
    result = []
    group_counts_so_far = {g: 0 for g in groups}

    for k in range(1, top_k + 1):
        # Find group most below target
        best_group = None
        best_deficit = -float("inf")

        for g in groups:
            if not group_queues[g]:
                continue
            target_count = target_proportions.get(g, 0) * k
            deficit = target_count - group_counts_so_far[g]
            if deficit > best_deficit:
                best_deficit = deficit
                best_group = g

        if best_group is not None and group_queues[best_group]:
            chosen = group_queues[best_group].pop(0)
            result.append(chosen)
            group_counts_so_far[best_group] += 1
        else:
            # Fallback: pick from any non-empty queue with highest score
            for iid in sorted_ids:
                if iid not in result:
                    result.append(iid)
                    g = attr_map.get(iid, groups[0])
                    group_counts_so_far[g] += 1
                    break

    return np.array(result[:top_k])


# ---- Our method: Multi-Attribute Fair Re-ranking ----

def multi_attribute_fair_rerank(
    scores: np.ndarray,
    item_ids: np.ndarray,
    protected_attrs: pd.DataFrame,
    top_k: int = 10,
    lambda_fair: float = 0.5,
    attrs: list[str] = None,
    **kwargs,
) -> np.ndarray:
    """Multi-attribute fair re-ranking (our proposed method).

    Extends DetConstSort to handle intersectional fairness across multiple
    protected attributes simultaneously (e.g., size x geography).

    The key insight is a composite fairness deficit score that considers
    multiple attributes, weighted by a fairness-relevance trade-off parameter.
    """
    if attrs is None:
        attrs = ["size_group", "geo_group"]

    # Build attribute maps
    attr_maps = {}
    group_targets = {}
    for attr in attrs:
        if attr not in protected_attrs.columns:
            continue
        attr_maps[attr] = dict(zip(protected_attrs["item_id"], protected_attrs[attr]))
        group_targets[attr] = protected_attrs[attr].value_counts(normalize=True).to_dict()

    if not attr_maps:
        # No protected attributes available, return score-based ranking
        order = np.argsort(scores)[::-1][:top_k]
        return item_ids[order]

    # Normalize scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        norm_scores = (scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(scores)

    score_map = dict(zip(item_ids, norm_scores))

    # Build intersectional groups
    inter_groups = {}
    for iid in item_ids:
        key = tuple(attr_maps[a].get(iid, "unknown") for a in attr_maps)
        inter_groups.setdefault(key, []).append(iid)

    # Target proportions for intersectional groups (product of marginals)
    inter_targets = {}
    for key in inter_groups:
        prob = 1.0
        for attr, val in zip(attr_maps.keys(), key):
            prob *= group_targets[attr].get(val, 1.0 / len(group_targets[attr]))
        inter_targets[key] = prob

    # Normalize targets
    total = sum(inter_targets.values())
    inter_targets = {k: v / total for k, v in inter_targets.items()}

    # Sort candidates within each intersectional group by score
    for key in inter_groups:
        inter_groups[key].sort(key=lambda x: score_map.get(x, 0), reverse=True)

    # Greedy selection with composite deficit
    result = []
    group_counts = {key: 0 for key in inter_groups}
    used = set()

    for k in range(1, top_k + 1):
        best_item = None
        best_combined_score = -float("inf")

        for key, candidates in inter_groups.items():
            # Find top unused candidate in this group
            candidate = None
            for c in candidates:
                if c not in used:
                    candidate = c
                    break
            if candidate is None:
                continue

            # Relevance component
            rel_score = score_map.get(candidate, 0)

            # Fairness deficit: how far below target is this group?
            target_count = inter_targets[key] * k
            deficit = target_count - group_counts[key]

            # Also compute marginal deficits
            marginal_deficit = 0
            iid_attrs = {a: attr_maps[a].get(candidate, "unknown") for a in attr_maps}
            for attr, val in iid_attrs.items():
                attr_target = group_targets[attr].get(val, 0) * k
                attr_count = sum(
                    1 for r in result
                    if attr_maps[attr].get(r, "unknown") == val
                )
                marginal_deficit += max(0, attr_target - attr_count)

            # Combined: trade-off between relevance and fairness
            fair_score = deficit + 0.5 * marginal_deficit
            combined = (1 - lambda_fair) * rel_score + lambda_fair * fair_score

            if combined > best_combined_score:
                best_combined_score = combined
                best_item = candidate
                best_key = key

        if best_item is not None:
            result.append(best_item)
            used.add(best_item)
            group_counts[best_key] += 1
        else:
            break

    # If we didn't fill top_k, add remaining by score
    if len(result) < top_k:
        remaining = [(score_map.get(iid, 0), iid) for iid in item_ids if iid not in used]
        remaining.sort(reverse=True)
        for _, iid in remaining:
            if len(result) >= top_k:
                break
            result.append(iid)

    return np.array(result[:top_k])
